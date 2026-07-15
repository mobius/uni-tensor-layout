"""Heterogeneous pipeline: Phi preprocess (scale) → multi-VE NLC DGEMM.

  A' = alpha * A + beta     (on Phi, IMCI)
  C  = A' @ B               (row-sharded multi-VE, optional worker pool)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale
from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm
from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


@dataclass
class HeteroPipelineResult:
    max_abs_err: float
    wall_sec: float
    phi_gflops: float
    ve_kernel_gflops_mean: float
    status: str
    notes: list[str] = field(default_factory=list)


def run_hetero_phi_ve_gemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float = 1.1,
    beta: float = 0.01,
    devices: Optional[Sequence[str]] = None,
    use_worker_pool: bool = True,
    phi_threads: int = 244,
) -> tuple[np.ndarray, HeteroPipelineResult]:
    """End-to-end hetero GEMM with correctness vs host reference."""
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

    if use_worker_pool:
        ve_ids = [int(d.replace("ve", "")) for d in devices]
        with VeWorkerPool(ve_ids) as pool:
            c, plan, ve_res = multi_ve_layout_dgemm_pooled(
                a_scaled, b, devices, pool, share_b=True
            )
        notes.append(f"ve_pool shards={len(plan.shards)}")
    else:
        c, plan, ve_res = multi_ve_layout_dgemm(
            a_scaled, b, devices, parallel=True, share_b=True
        )
        notes.append(f"ve_oneshot shards={len(plan.shards)}")

    wall = time.perf_counter() - t0
    ref = (alpha * a + beta) @ b
    err = float(np.max(np.abs(c - ref)))
    ve_mean = sum(r.gflops for r in ve_res) / max(len(ve_res), 1)
    status = "pass" if err < 1e-8 and all(r.status == "pass" for r in ve_res) else "fail"
    return c, HeteroPipelineResult(
        max_abs_err=err,
        wall_sec=wall,
        phi_gflops=prep.gflops,
        ve_kernel_gflops_mean=ve_mean,
        status=status,
        notes=notes,
    )
