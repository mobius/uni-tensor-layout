"""Heterogeneous dataprep + SpMV → dense GEMM pipeline (Phase 2 M2 / W3.1).

Pipeline:
  1. Build CSR from sparse pattern (host) OR load user CSR
  2. Irregular phase: CSR SpMV  y = A_sp @ x   (host; optional Phi scale on dense)
  3. Dense phase: multi-VE GEMM  C = Y @ B  with auto PlacementPlan + PowerCap

This is intentionally not pure-GEMM: SpMV is memory-irregular; GEMM is compute dense.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.partition.cost_model import PlacementChoice, choose_best_placement
from uni_cute_tensor.partition.multi_device import PlacementPlan
from uni_cute_tensor.partition.runner import PlanRunResult, execute_plan
from uni_cute_tensor.power import PowerCap
from uni_cute_tensor.runtime.timeline import get_timeline, timeline_scope


@dataclass
class CSRMatrix:
    """Minimal CSR (float64 values, int32 indices)."""

    nrows: int
    ncols: int
    indptr: np.ndarray  # (nrows+1,) int32
    indices: np.ndarray  # (nnz,) int32
    data: np.ndarray  # (nnz,) float64

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    def to_dense(self) -> np.ndarray:
        a = np.zeros((self.nrows, self.ncols), dtype=np.float64)
        for i in range(self.nrows):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                a[i, self.indices[p]] = self.data[p]
        return a


def random_csr(
    nrows: int,
    ncols: int,
    density: float = 0.05,
    *,
    rng: Optional[np.random.Generator] = None,
) -> CSRMatrix:
    """Generate a random CSR with approximate density."""
    rng = rng or np.random.default_rng(0)
    density = float(np.clip(density, 1e-6, 1.0))
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for _ in range(nrows):
        # expected nnz per row
        k = max(1, int(rng.binomial(ncols, density)))
        cols = rng.choice(ncols, size=min(k, ncols), replace=False)
        cols.sort()
        for c in cols:
            indices.append(int(c))
            data.append(float(rng.standard_normal()))
        indptr.append(len(indices))
    return CSRMatrix(
        nrows=nrows,
        ncols=ncols,
        indptr=np.asarray(indptr, dtype=np.int32),
        indices=np.asarray(indices, dtype=np.int32),
        data=np.asarray(data, dtype=np.float64),
    )


def csr_spmv(csr: CSRMatrix, x: np.ndarray) -> np.ndarray:
    """y = A_sp @ x  (host CSR SpMV)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim == 1:
        if x.shape[0] != csr.ncols:
            raise ValueError("x length mismatch")
        y = np.zeros(csr.nrows, dtype=np.float64)
        for i in range(csr.nrows):
            s = 0.0
            for p in range(csr.indptr[i], csr.indptr[i + 1]):
                s += csr.data[p] * x[csr.indices[p]]
            y[i] = s
        return y
    if x.ndim == 2:
        if x.shape[0] != csr.ncols:
            raise ValueError("X rows must equal csr.ncols")
        y = np.zeros((csr.nrows, x.shape[1]), dtype=np.float64)
        for i in range(csr.nrows):
            for p in range(csr.indptr[i], csr.indptr[i + 1]):
                y[i, :] += csr.data[p] * x[csr.indices[p], :]
        return y
    raise ValueError("x must be 1-D or 2-D")


def host_scale(a: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return alpha * a + beta


@dataclass
class SpmvDataprepResult:
    y: np.ndarray  # SpMV output (also dense left matrix for GEMM)
    c: np.ndarray  # dense GEMM result
    plan: PlacementPlan
    choice: Optional[PlacementChoice]
    wall_sec: float
    spmv_sec: float
    prep_sec: float
    gemm_sec: float
    max_abs_err: float
    status: str
    notes: list[str] = field(default_factory=list)
    host_wall_sec: float = 0.0
    speedup_vs_host: float = 0.0
    spmv_gflops: float = 0.0
    gemm_gflops: float = 0.0


def run_spmv_dataprep_pipeline(
    csr: CSRMatrix,
    x: np.ndarray,
    b_dense: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    devices: Optional[Sequence[str]] = None,
    power_cap: Optional[PowerCap] = None,
    use_phi_prep: bool = False,
    force_host_gemm: bool = False,
    compare_host: bool = True,
) -> SpmvDataprepResult:
    """End-to-end: SpMV → optional scale → auto-placed dense GEMM.

    Reference: C_ref = (alpha * (A_sp @ X) + beta) @ B
    When x is 1-D, X is treated as column vector broadcast to form tall matrix
    via outer with ones or reshape — for multi-RHS, pass x as 2-D (ncols, k_rhs).
    """
    notes: list[str] = []
    if devices is None:
        devices = ve_device_names(discover_devices())
    if power_cap is None:
        power_cap = PowerCap()

    t_all = time.perf_counter()

    # --- irregular SpMV ---
    tl = get_timeline()
    t0 = time.perf_counter()
    with tl.span("csr_spmv", "kernel", device="host") if tl else _null():
        y = csr_spmv(csr, x)
    spmv_sec = time.perf_counter() - t0
    # flops ≈ 2*nnz*nrhs
    nrhs = 1 if y.ndim == 1 else y.shape[1]
    spmv_flops = 2.0 * csr.nnz * nrhs
    spmv_gflops = spmv_flops / spmv_sec / 1e9 if spmv_sec > 0 else 0.0
    notes.append(f"spmv nnz={csr.nnz} gflops={spmv_gflops:.3f}")

    # ensure Y is 2-D for GEMM: if vector, use diag-like path — actually
    # y vector @ B needs y as (m,1) or we form dense feature matrix.
    if y.ndim == 1:
        # expand: use y as first column and generate k features via shifts (dataprep)
        # simpler: Y = y[:, None] * ones → rank-1; instead stack scaled copies for demo
        y2 = y[:, None]
    else:
        y2 = y

    # --- dataprep scale (Phi optional, else host) ---
    t0 = time.perf_counter()
    with tl.span("dataprep_scale", "host", device="phi0" if use_phi_prep else "host") if tl else _null():
        if use_phi_prep:
            try:
                from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale

                y_scaled, prep = run_phi_prep_scale(y2, alpha=alpha, beta=beta)
                notes.append(f"phi_prep status={prep.status} gflops={prep.gflops:.2f}")
                if prep.status != "pass":
                    y_scaled = host_scale(y2, alpha, beta)
                    notes.append("phi_prep_fallback_host")
            except Exception as exc:
                y_scaled = host_scale(y2, alpha, beta)
                notes.append(f"phi_prep_skip:{exc}")
        else:
            y_scaled = host_scale(y2, alpha, beta)
            notes.append("host_scale")
    prep_sec = time.perf_counter() - t0

    b_dense = np.ascontiguousarray(b_dense, dtype=np.float64)
    if y_scaled.shape[1] != b_dense.shape[0]:
        raise ValueError(
            f"inner dim Y@B mismatch {y_scaled.shape} @ {b_dense.shape}"
        )

    m, k = y_scaled.shape
    n = b_dense.shape[1]

    # --- auto placement + PowerCap ---
    launch = list(devices) if devices else ["host"]
    if use_phi_prep:
        launch = ["phi0"] + launch
    ops = {d: ("scale" if d.startswith("phi") else "dgemm") for d in launch}

    choice: Optional[PlacementChoice] = None
    gemm_res: Optional[PlanRunResult] = None
    t0 = time.perf_counter()
    if force_host_gemm or not devices:
        from uni_cute_tensor.backends.host_dgemm import host_dgemm

        with power_cap.guard(["host"], {"host": "dgemm"}):
            with tl.span("host_gemm", "kernel", device="host") if tl else _null():
                c, hr = host_dgemm(y_scaled, b_dense, backend="auto")
        plan = PlacementPlan(
            global_shape=(m, n),
            shards=[],
            strategy="row_blocks",
            k=k,
            backend="HOST_OBLAS",
        )
        gemm_res = PlanRunResult(
            c=c,
            plan=plan,
            wall_sec=0.0,
            max_abs_err=hr.max_abs_err,
            status=hr.status,
            backend="HOST_OBLAS",
            strategy="row_blocks",
            kernel_gflops_mean=hr.gflops,
        )
        notes.append("gemm=host")
    else:
        # may shrink device set under power budget
        if not power_cap.can_launch(list(devices), {d: "dgemm" for d in devices}):
            # try fewer devices
            ok_devs = []
            for i in range(len(devices), 0, -1):
                sub = list(devices[:i])
                if power_cap.can_launch(sub, {d: "dgemm" for d in sub}):
                    ok_devs = sub
                    break
            if not ok_devs:
                raise RuntimeError("PowerCap blocks all VE subsets")
            devices = ok_devs
            notes.append(f"power_shrink_devices={devices}")

        choice = choose_best_placement(
            m, n, k, list(devices), backend="VE_NLC", power_cap=power_cap
        )
        notes.append(
            f"plan strategy={choice.strategy} devs={choice.devices} "
            f"est={choice.est_total_sec:.4f}s watts={choice.power_watts:.0f}"
        )
        with power_cap.guard(choice.devices, {d: "dgemm" for d in choice.devices}):
            with tl.span("ve_gemm_auto", "kernel", device=",".join(choice.devices)) if tl else _null():
                gemm_res = execute_plan(y_scaled, b_dense, choice.plan, use_pool=True)
        notes.append(f"gemm={gemm_res.backend}/{gemm_res.strategy} {gemm_res.status}")
        plan = gemm_res.plan

    gemm_sec = time.perf_counter() - t0
    wall = time.perf_counter() - t_all

    # reference
    ref_y = csr_spmv(csr, x)
    if ref_y.ndim == 1:
        ref_y2 = ref_y[:, None]
    else:
        ref_y2 = ref_y
    ref_c = host_scale(ref_y2, alpha, beta) @ b_dense
    err = float(np.max(np.abs(gemm_res.c - ref_c)))

    host_wall = 0.0
    speedup = 0.0
    if compare_host and not force_host_gemm:
        t_h = time.perf_counter()
        _ = host_scale(ref_y2, alpha, beta) @ b_dense
        # include spmv+scale+gemm host path
        _ = csr_spmv(csr, x)
        host_wall = time.perf_counter() - t_h + spmv_sec  # approx: re-run full host
        # fairer: single timed host path
        t_h = time.perf_counter()
        y_h = csr_spmv(csr, x)
        if y_h.ndim == 1:
            y_h = y_h[:, None]
        c_h = host_scale(y_h, alpha, beta) @ b_dense
        host_wall = time.perf_counter() - t_h
        del c_h
        speedup = host_wall / wall if wall > 0 else 0.0
        notes.append(f"host_wall={host_wall:.4f}s speedup={speedup:.2f}x")

    status = "pass" if err < 1e-8 and gemm_res.status == "pass" else "fail"
    return SpmvDataprepResult(
        y=y_scaled,
        c=gemm_res.c,
        plan=plan,
        choice=choice,
        wall_sec=wall,
        spmv_sec=spmv_sec,
        prep_sec=prep_sec,
        gemm_sec=gemm_sec,
        max_abs_err=err,
        status=status,
        notes=notes,
        host_wall_sec=host_wall,
        speedup_vs_host=speedup,
        spmv_gflops=spmv_gflops,
        gemm_gflops=gemm_res.kernel_gflops_mean,
    )


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_with_timeline(
    *args: Any,
    job_id: str = "spmv-dataprep",
    **kwargs: Any,
) -> tuple[SpmvDataprepResult, Any]:
    with timeline_scope(job_id=job_id) as tl:
        res = run_spmv_dataprep_pipeline(*args, **kwargs)
    return res, tl


@dataclass
class MultibatchSpmvResult:
    batches: int
    wall_sec: float
    host_wall_sec: float
    speedup_vs_host: float
    max_abs_err: float
    status: str
    notes: list[str] = field(default_factory=list)
    throughput_batches_per_sec: float = 0.0


def run_spmv_dataprep_multibatch(
    csr: CSRMatrix,
    xs: Sequence[np.ndarray],
    bs: Sequence[np.ndarray],
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    devices: Optional[Sequence[str]] = None,
    power_cap: Optional[PowerCap] = None,
    overlap: bool = True,
) -> MultibatchSpmvResult:
    """Multi-batch: SpMV(i+1) overlaps GEMM(i) on VE pool (amortized placement).

    Host baseline is serial SpMV+scale+numpy GEMM per batch.
    """
    from concurrent.futures import ThreadPoolExecutor

    if len(xs) != len(bs) or not xs:
        raise ValueError("xs/bs length mismatch")
    if devices is None:
        devices = ve_device_names(discover_devices())
    if power_cap is None:
        power_cap = PowerCap()
    notes = [f"batches={len(xs)} overlap={overlap}"]

    # host baseline
    t_h = time.perf_counter()
    for x, b in zip(xs, bs):
        y = csr_spmv(csr, x)
        if y.ndim == 1:
            y = y[:, None]
        _ = host_scale(y, alpha, beta) @ b
    host_wall = time.perf_counter() - t_h

    if not devices:
        return MultibatchSpmvResult(
            batches=len(xs),
            wall_sec=host_wall,
            host_wall_sec=host_wall,
            speedup_vs_host=1.0,
            max_abs_err=0.0,
            status="pass",
            notes=notes + ["host_only"],
            throughput_batches_per_sec=len(xs) / host_wall if host_wall > 0 else 0.0,
        )

    # pick plan from first batch shapes
    y0 = csr_spmv(csr, xs[0])
    if y0.ndim == 1:
        y0 = y0[:, None]
    y0 = host_scale(y0, alpha, beta)
    m, k = y0.shape
    n = bs[0].shape[1]
    choice = choose_best_placement(m, n, k, list(devices), backend="VE_NLC", power_cap=power_cap)
    notes.append(f"plan={choice.strategy} devs={choice.devices}")

    max_err = 0.0
    from uni_cute_tensor.backends.ve_worker import VeWorkerPool
    from uni_cute_tensor.partition.runner import clear_shared_ve_pool, set_shared_ve_pool

    ve_ids = [int(d.replace("ve", "")) for d in choice.devices if d.startswith("ve")]
    t0 = time.perf_counter()
    with power_cap.guard(choice.devices, {d: "dgemm" for d in choice.devices}):
        pool = VeWorkerPool(ve_ids) if ve_ids else None
        if pool is not None:
            pool.start()
            set_shared_ve_pool(pool)
        try:
            if not overlap:
                for x, b in zip(xs, bs):
                    y = csr_spmv(csr, x)
                    if y.ndim == 1:
                        y = y[:, None]
                    ys = host_scale(y, alpha, beta)
                    r = execute_plan(ys, b, choice.plan, use_pool=True, check=True)
                    max_err = max(max_err, r.max_abs_err)
            else:

                def prep(x):
                    y = csr_spmv(csr, x)
                    if y.ndim == 1:
                        y = y[:, None]
                    return host_scale(y, alpha, beta)

                ys = prep(xs[0])
                for i in range(len(xs)):
                    b = bs[i]

                    def gemm_job(ys=ys, b=b):
                        # check=False during timed path; verify after against host matmul
                        r = execute_plan(ys, b, choice.plan, use_pool=True, check=False)
                        ref = ys @ b
                        err = float(np.max(np.abs(r.c - ref)))
                        st = "pass" if err < 1e-8 else "fail"
                        return err, st

                    if i + 1 < len(xs):
                        with ThreadPoolExecutor(max_workers=2) as ex:
                            f_g = ex.submit(gemm_job)
                            f_p = ex.submit(prep, xs[i + 1])
                            err, st = f_g.result()
                            ys = f_p.result()
                    else:
                        err, st = gemm_job()
                    max_err = max(max_err, err)
                    if st != "pass":
                        notes.append(f"fail_batch{i}")
        finally:
            clear_shared_ve_pool()
            if pool is not None:
                pool.stop()
    wall = time.perf_counter() - t0
    speedup = host_wall / wall if wall > 0 else 0.0
    notes.append(f"host_wall={host_wall:.4f} speedup={speedup:.2f}x")
    status = "pass" if max_err < 1e-8 else "fail"
    return MultibatchSpmvResult(
        batches=len(xs),
        wall_sec=wall,
        host_wall_sec=host_wall,
        speedup_vs_host=speedup,
        max_abs_err=max_err,
        status=status,
        notes=notes,
        throughput_batches_per_sec=len(xs) / wall if wall > 0 else 0.0,
    )
