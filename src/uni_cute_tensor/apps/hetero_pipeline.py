"""Heterogeneous pipeline: Phi preprocess → multi-VE DGEMM.

Supports serial, worker-pool, AVEO multi-VE, and multi-batch double-buffering.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale
from uni_cute_tensor.backends.phi_worker import PhiWorker
from uni_cute_tensor.backends.ve_aveo import multi_ve_aveo_dgemm
from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm
from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.power import PowerCap


@dataclass
class HeteroPipelineResult:
    max_abs_err: float
    wall_sec: float
    phi_gflops: float
    ve_kernel_gflops_mean: float
    status: str
    notes: list[str] = field(default_factory=list)
    batches: int = 1
    throughput_batches_per_sec: float = 0.0


def _ve_gemm(a_scaled, b, devices, *, mode: str, pool: Optional[VeWorkerPool]):
    if mode == "aveo":
        c, plan, ve_res = multi_ve_aveo_dgemm(a_scaled, b, devices)
        mean = sum(r.gflops for r in ve_res) / max(len(ve_res), 1)
        ok = all(r.status == "pass" for r in ve_res)
        return c, plan, mean, ok, "aveo"
    if mode == "pool" and pool is not None:
        c, plan, ve_res = multi_ve_layout_dgemm_pooled(
            a_scaled, b, devices, pool, share_b=True
        )
        mean = sum(r.gflops for r in ve_res) / max(len(ve_res), 1)
        ok = all(r.status == "pass" for r in ve_res)
        return c, plan, mean, ok, "pool"
    c, plan, ve_res = multi_ve_layout_dgemm(
        a_scaled, b, devices, parallel=True, share_b=True
    )
    mean = sum(r.gflops for r in ve_res) / max(len(ve_res), 1)
    ok = all(r.status == "pass" for r in ve_res)
    return c, plan, mean, ok, "oneshot"


def run_hetero_phi_ve_gemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float = 1.1,
    beta: float = 0.01,
    devices: Optional[Sequence[str]] = None,
    use_worker_pool: bool = True,
    use_aveo: bool = False,
    phi_threads: int = 244,
) -> tuple[np.ndarray, HeteroPipelineResult]:
    """Single-batch hetero GEMM."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if devices is None:
        devices = ve_device_names(discover_devices())
    if not devices:
        raise RuntimeError("no VE devices")

    t0 = time.perf_counter()
    notes: list[str] = []
    a_scaled, prep = run_phi_prep_scale(a, alpha=alpha, beta=beta, threads=phi_threads)
    notes.append(f"phi_prep status={prep.status} gflops={prep.gflops:.2f}")
    if prep.status != "pass":
        raise RuntimeError(f"phi prep failed err={prep.max_abs_err}")

    mode = "aveo" if use_aveo else ("pool" if use_worker_pool else "oneshot")
    pool = None
    if mode == "pool":
        ve_ids = [int(d.replace("ve", "")) for d in devices]
        pool = VeWorkerPool(ve_ids)
        pool.start()
    try:
        c, plan, ve_mean, ok, used = _ve_gemm(
            a_scaled, b, devices, mode=mode, pool=pool
        )
        notes.append(f"ve_mode={used} shards={len(plan.shards)}")
    finally:
        if pool is not None:
            pool.stop()

    wall = time.perf_counter() - t0
    ref = (alpha * a + beta) @ b
    err = float(np.max(np.abs(c - ref)))
    status = "pass" if err < 1e-8 and ok else "fail"
    return c, HeteroPipelineResult(
        max_abs_err=err,
        wall_sec=wall,
        phi_gflops=prep.gflops,
        ve_kernel_gflops_mean=ve_mean,
        status=status,
        notes=notes,
        batches=1,
        throughput_batches_per_sec=1.0 / wall if wall > 0 else 0.0,
    )


def run_hetero_multibatch(
    batches_a: Sequence[np.ndarray],
    batches_b: Sequence[np.ndarray],
    *,
    alpha: float = 1.05,
    beta: float = 0.0,
    devices: Optional[Sequence[str]] = None,
    overlap: bool = True,
    use_aveo: bool = False,
    use_phi_worker: bool = True,
    phi_threads: int = 120,
    power_cap: Optional[PowerCap] = None,
) -> HeteroPipelineResult:
    """Multi-batch pipeline; optional Phi worker + Phi||VE overlap."""
    if len(batches_a) != len(batches_b) or not batches_a:
        raise ValueError("batches_a/b must be non-empty and same length")
    if devices is None:
        devices = ve_device_names(discover_devices())
    if not devices:
        raise RuntimeError("no VE")

    n_batch = len(batches_a)
    max_err = 0.0
    notes: list[str] = [
        f"batches={n_batch} overlap={overlap} aveo={use_aveo} "
        f"phi_worker={use_phi_worker}"
    ]
    t0 = time.perf_counter()

    mode = "aveo" if use_aveo else "pool"
    pool = None
    phi_w: Optional[PhiWorker] = None
    ve_ids = [int(d.replace("ve", "")) for d in devices]
    launch_devs = ["phi0"] + list(devices)
    ops = {d: ("scale" if d.startswith("phi") else "dgemm") for d in launch_devs}

    if power_cap is None:
        power_cap = PowerCap()
    if not power_cap.can_launch(launch_devs, ops):
        raise RuntimeError(
            f"PowerCap refused hetero launch backend={power_cap.backend} "
            f"limit={power_cap.effective_limit:.0f}W"
        )
    power_cap.reserve(launch_devs, ops)
    notes.append(f"power_cap={power_cap.backend}")

    def phi_scale(a_mat):
        if phi_w is not None:
            return phi_w.scale(a_mat, alpha=alpha, beta=beta)
        return run_phi_prep_scale(a_mat, alpha=alpha, beta=beta, threads=phi_threads)

    ve_means: list[float] = []
    phi_g = 0.0

    try:
        if mode == "pool":
            pool = VeWorkerPool(ve_ids)
            pool.start()
        if use_phi_worker:
            phi_w = PhiWorker()
            phi_w.start(threads=phi_threads)

        if not overlap:
            for a, b in zip(batches_a, batches_b):
                a_s, prep = phi_scale(a)
                phi_g = prep.gflops
                c, _, ve_mean, ok, _ = _ve_gemm(
                    a_s, b, devices, mode=mode, pool=pool
                )
                ref = (alpha * a + beta) @ b
                max_err = max(max_err, float(np.max(np.abs(c - ref))))
                ve_means.append(ve_mean)
                if not ok:
                    notes.append("ve_fail")
        else:
            a_s, prep = phi_scale(batches_a[0])
            phi_g = prep.gflops
            prepped = [(a_s, batches_b[0], batches_a[0])]
            for i in range(n_batch):
                a_s_i, b_i, a_orig = prepped[0] if i == 0 else prepped.pop(0)

                def ve_job(a_s=a_s_i, b=b_i, a0=a_orig):
                    c, _, ve_mean, ok, _ = _ve_gemm(
                        a_s, b, devices, mode=mode, pool=pool
                    )
                    ref = (alpha * a0 + beta) @ b
                    err = float(np.max(np.abs(c - ref)))
                    return err, ve_mean, ok

                if i + 1 < n_batch:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        f_ve = ex.submit(ve_job)
                        f_phi = ex.submit(phi_scale, batches_a[i + 1])
                        err, ve_mean, ok = f_ve.result()
                        a_next, prep_next = f_phi.result()
                        phi_g = prep_next.gflops
                        prepped.append((a_next, batches_b[i + 1], batches_a[i + 1]))
                else:
                    err, ve_mean, ok = ve_job()
                max_err = max(max_err, err)
                ve_means.append(ve_mean)
                if not ok:
                    notes.append(f"ve_fail_batch{i}")
    finally:
        if pool is not None:
            pool.stop()
        if phi_w is not None:
            phi_w.stop()
        power_cap.release(launch_devs)

    wall = time.perf_counter() - t0
    status = "pass" if max_err < 1e-8 else "fail"
    return HeteroPipelineResult(
        max_abs_err=max_err,
        wall_sec=wall,
        phi_gflops=phi_g,
        ve_kernel_gflops_mean=sum(ve_means) / max(len(ve_means), 1),
        status=status,
        notes=notes,
        batches=n_batch,
        throughput_batches_per_sec=n_batch / wall if wall > 0 else 0.0,
    )
