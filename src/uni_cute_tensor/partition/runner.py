"""Execute a PlacementPlan for GEMM C=A@B (W2.4 plan → runner)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from uni_cute_tensor.partition.multi_device import PlacementPlan


@dataclass
class PlanRunResult:
    c: np.ndarray
    plan: PlacementPlan
    wall_sec: float
    max_abs_err: float
    status: str
    backend: str
    strategy: str
    shard_notes: list[str] = field(default_factory=list)
    kernel_gflops_mean: float = 0.0


def _host_execute(a: np.ndarray, b: np.ndarray, plan: PlacementPlan) -> tuple[np.ndarray, list[str], float]:
    """Host execution of any strategy (correctness + fallback)."""
    m, n = plan.global_shape
    c = np.zeros((m, n), dtype=np.float64)
    notes: list[str] = []
    t_kern = 0.0
    if plan.strategy == "row_blocks":
        for s in plan.shards:
            t0 = time.perf_counter()
            c[s.row_start : s.row_end, :] = a[s.row_start : s.row_end, :] @ b
            t_kern += time.perf_counter() - t0
            notes.append(f"{s.device}:row[{s.row_start}:{s.row_end}]")
    elif plan.strategy == "col_blocks":
        for s in plan.shards:
            t0 = time.perf_counter()
            c[:, s.col_start : s.col_end] = a @ b[:, s.col_start : s.col_end]
            t_kern += time.perf_counter() - t0
            notes.append(f"{s.device}:col[{s.col_start}:{s.col_end}]")
    elif plan.strategy == "k_split":
        for s in plan.shards:
            ke = s.k_end if s.k_end > 0 else plan.k
            t0 = time.perf_counter()
            c += a[:, s.k_start : ke] @ b[s.k_start : ke, :]
            t_kern += time.perf_counter() - t0
            notes.append(f"{s.device}:k[{s.k_start}:{ke}]")
    else:
        raise ValueError(f"unsupported strategy {plan.strategy}")
    flops = 2.0 * m * n * max(plan.k, a.shape[1])
    gflops = flops / t_kern / 1e9 if t_kern > 0 else 0.0
    return c, notes, gflops


def _ve_execute(
    a: np.ndarray,
    b: np.ndarray,
    plan: PlacementPlan,
    *,
    use_pool: bool = True,
) -> tuple[np.ndarray, list[str], float]:
    """VE execution via worker pool or oneshot; supports row/col/k strategies."""
    import shutil
    import tempfile
    from pathlib import Path

    from uni_cute_tensor.backends.ve_dgemm import (
        VeRunResult,
        _parse_elapsed,
        _parse_gflops,
        _read_output,
        _write_combined,
        run_ve_dgemm_shard,
    )
    from uni_cute_tensor.backends.ve_worker import VeWorkerPool

    m, n = plan.global_shape
    c = np.zeros((m, n), dtype=np.float64)
    notes: list[str] = []
    gflops_list: list[float] = []

    ve_ids = []
    for s in plan.shards:
        if s.device.startswith("ve") and s.device[2:].isdigit():
            ve_ids.append(int(s.device[2:]))
    ve_ids = sorted(set(ve_ids))
    pool: Optional[VeWorkerPool] = None
    work_dir: Optional[Path] = None
    own_pool = False
    if use_pool and ve_ids:
        if (
            _REUSE_POOL is not None
            and all(v in getattr(_REUSE_POOL, "ve_ids", []) for v in ve_ids)
        ):
            pool = _REUSE_POOL
        else:
            pool = VeWorkerPool(ve_ids)
            pool.start()
            own_pool = True
        work_dir = Path(tempfile.mkdtemp(prefix="cct_plan_run_"))

    def _shard_gemm(a_sh: np.ndarray, b_sh: np.ndarray, ve_id: int) -> tuple[np.ndarray, VeRunResult]:
        if pool is not None and work_dir is not None:
            sub = work_dir / f"ve{ve_id}_{id(a_sh)}"
            sub.mkdir(parents=True, exist_ok=True)
            in_path = (sub / "in.bin").resolve()
            out_path = (sub / "out.bin").resolve()
            _write_combined(in_path, a_sh, b_sh)
            t0 = time.perf_counter()
            log = pool.run_combined(ve_id, in_path, out_path)
            wall = time.perf_counter() - t0
            c_sh = _read_output(out_path)
            err = float(np.max(np.abs(c_sh - a_sh @ b_sh)))
            res = VeRunResult(
                device=f"ve{ve_id}",
                ve_id=ve_id,
                m=a_sh.shape[0],
                n=b_sh.shape[1],
                k=a_sh.shape[1],
                gflops=_parse_gflops(log),
                elapsed_sec=_parse_elapsed(log) or wall,
                checksum=0.0,
                max_abs_err=err,
                stdout=log,
                stderr="",
                status="pass" if err < 1e-8 else "fail",
                mode="combined-pooled",
            )
            return c_sh, res
        return run_ve_dgemm_shard(a_sh, b_sh, ve_id=ve_id)

    try:
        if plan.strategy == "row_blocks":

            def _one_row(s):
                ve_id = int(s.device.replace("ve", ""))
                a_sh = a[s.row_start : s.row_end, :]
                c_sh, res = _shard_gemm(a_sh, b, ve_id)
                return s, c_sh, res

            with ThreadPoolExecutor(max_workers=max(len(plan.shards), 1)) as ex:
                futs = [ex.submit(_one_row, s) for s in plan.shards]
                for fut in as_completed(futs):
                    s, c_sh, res = fut.result()
                    c[s.row_start : s.row_end, :] = c_sh
                    gflops_list.append(res.gflops)
                    notes.append(f"{s.device}:{res.status}:{res.mode}")

        elif plan.strategy == "col_blocks":

            def _one_col(s):
                ve_id = int(s.device.replace("ve", ""))
                b_sh = b[:, s.col_start : s.col_end]
                c_sh, res = _shard_gemm(a, b_sh, ve_id)
                return s, c_sh, res

            with ThreadPoolExecutor(max_workers=max(len(plan.shards), 1)) as ex:
                futs = [ex.submit(_one_col, s) for s in plan.shards]
                for fut in as_completed(futs):
                    s, c_sh, res = fut.result()
                    c[:, s.col_start : s.col_end] = c_sh
                    gflops_list.append(res.gflops)
                    notes.append(f"{s.device}:{res.status}:{res.mode}")

        elif plan.strategy == "k_split":
            partials: list[np.ndarray] = [np.zeros((m, n)) for _ in plan.shards]

            def _one_k(idx_s):
                idx, s = idx_s
                ve_id = int(s.device.replace("ve", ""))
                ke = s.k_end if s.k_end > 0 else plan.k
                a_sh = a[:, s.k_start : ke]
                b_sh = b[s.k_start : ke, :]
                c_sh, res = _shard_gemm(a_sh, b_sh, ve_id)
                return idx, s, c_sh, res

            with ThreadPoolExecutor(max_workers=max(len(plan.shards), 1)) as ex:
                futs = [ex.submit(_one_k, (i, s)) for i, s in enumerate(plan.shards)]
                for fut in as_completed(futs):
                    idx, s, c_sh, res = fut.result()
                    partials[idx] = c_sh
                    gflops_list.append(res.gflops)
                    notes.append(
                        f"{s.device}:k[{s.k_start}:{s.k_end or plan.k}]:{res.status}"
                    )
            for p in partials:
                c += p
        else:
            raise ValueError(f"unsupported strategy {plan.strategy}")
    finally:
        if own_pool and pool is not None:
            pool.stop()
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    mean_g = sum(gflops_list) / max(len(gflops_list), 1)
    return c, notes, mean_g


# Optional process-wide VE pool reuse (multibatch / service mode)
_REUSE_POOL = None
_REUSE_POOL_IDS: list[int] = []


def set_shared_ve_pool(pool) -> None:
    """Install a shared VeWorkerPool for execute_plan (caller owns lifecycle)."""
    global _REUSE_POOL, _REUSE_POOL_IDS
    _REUSE_POOL = pool
    _REUSE_POOL_IDS = list(getattr(pool, "ve_ids", []) or [])


def clear_shared_ve_pool() -> None:
    global _REUSE_POOL, _REUSE_POOL_IDS
    _REUSE_POOL = None
    _REUSE_POOL_IDS = []


def execute_plan(
    a: np.ndarray,
    b: np.ndarray,
    plan: PlacementPlan,
    *,
    force_host: bool = False,
    use_pool: bool = True,
    check: bool = True,
) -> PlanRunResult:
    """Run GEMM according to plan.backend / plan.strategy."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    if plan.global_shape != (m, n):
        raise ValueError(f"plan C shape {plan.global_shape} != {(m, n)}")
    if plan.k == 0:
        plan.k = k

    backend = plan.backend
    t0 = time.perf_counter()
    if force_host or backend in ("HOST_OBLAS", "HOST_AVX512", "HOST_REF") or not plan.shards:
        if backend == "HOST_OBLAS" and plan.strategy == "row_blocks" and len(plan.shards) <= 1:
            from uni_cute_tensor.backends.host_dgemm import host_dgemm

            c, r = host_dgemm(a, b, backend="openblas")
            notes = [f"host_openblas:{r.status}"]
            gflops = r.gflops
        else:
            c, notes, gflops = _host_execute(a, b, plan)
            backend = backend if backend.startswith("HOST") else "HOST_REF"
    elif backend == "VE_NLC":
        try:
            # single-device: prefer AVEO when no shared worker pool is installed
            if (
                _REUSE_POOL is None
                and len(plan.shards) == 1
                and plan.strategy in ("row_blocks", "col_blocks")
                and plan.shards[0].device.startswith("ve")
            ):
                try:
                    from uni_cute_tensor.backends.ve_aveo import (
                        AveoSessionPool,
                        aveo_available,
                    )

                    if aveo_available():
                        ve_id = int(plan.shards[0].device.replace("ve", ""))
                        s = plan.shards[0]
                        with AveoSessionPool([ve_id]) as ap:
                            if plan.strategy == "row_blocks":
                                a_sh = a[s.row_start : s.row_end, :]
                                c_sh, r = ap.dgemm(ve_id, a_sh, b)
                                c = np.zeros((m, n), dtype=np.float64)
                                c[s.row_start : s.row_end, :] = c_sh
                            else:
                                b_sh = b[:, s.col_start : s.col_end]
                                c_sh, r = ap.dgemm(ve_id, a, b_sh)
                                c = np.zeros((m, n), dtype=np.float64)
                                c[:, s.col_start : s.col_end] = c_sh
                            notes = [f"aveo:{r.status}:{r.mode}"]
                            gflops = r.gflops
                        wall = time.perf_counter() - t0
                        err = float(np.max(np.abs(c - (a @ b)))) if check else 0.0
                        status = "pass" if (not check or err < 1e-8) else "fail"
                        return PlanRunResult(
                            c=c,
                            plan=plan,
                            wall_sec=wall,
                            max_abs_err=err,
                            status=status,
                            backend="VE_AVEO",
                            strategy=plan.strategy,
                            shard_notes=notes,
                            kernel_gflops_mean=gflops,
                        )
                except Exception as aveo_exc:
                    notes_pre = [f"aveo_skip:{aveo_exc}"]
                else:
                    notes_pre = []
            else:
                notes_pre = []
            c, notes, gflops = _ve_execute(a, b, plan, use_pool=use_pool)
            notes = notes_pre + notes
        except Exception as exc:
            # degrade to host for correctness paths
            c, notes, gflops = _host_execute(a, b, plan)
            notes = [f"ve_fallback:{exc}"] + notes
            backend = "HOST_REF"
    elif backend == "PHI_MKL":
        # single-device phi for whole matrix; multi-shard → host sim
        if len(plan.shards) == 1:
            from uni_cute_tensor.backends.phi_dgemm import run_phi_dgemm

            c, r = run_phi_dgemm(a, b, backend="mkl")
            notes = [f"phi:{r.status}"]
            gflops = r.gflops
        else:
            c, notes, gflops = _host_execute(a, b, plan)
            notes = ["phi_multi_shard_host"] + notes
            backend = "HOST_REF"
    else:
        c, notes, gflops = _host_execute(a, b, plan)

    wall = time.perf_counter() - t0
    err = 0.0
    if check:
        err = float(np.max(np.abs(c - (a @ b))))
    status = "pass" if (not check or err < 1e-8) else "fail"
    return PlanRunResult(
        c=c,
        plan=plan,
        wall_sec=wall,
        max_abs_err=err,
        status=status,
        backend=backend,
        strategy=plan.strategy,
        shard_notes=notes,
        kernel_gflops_mean=gflops,
    )


def execute_auto(
    a: np.ndarray,
    b: np.ndarray,
    devices: Sequence[str],
    *,
    power_cap: Any = None,
    backend: str = "VE_NLC",
    **choose_kwargs: Any,
) -> tuple[PlanRunResult, Any]:
    """choose_best_placement + execute_plan."""
    from uni_cute_tensor.partition.cost_model import choose_best_placement

    m, k = a.shape
    n = b.shape[1]
    choice = choose_best_placement(
        m, n, k, devices, backend=backend, power_cap=power_cap, **choose_kwargs
    )
    result = execute_plan(a, b, choice.plan)
    return result, choice
