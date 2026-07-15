#!/usr/bin/env python3
"""Demo: Phi scale preprocess → multi-VE DGEMM (worker pool)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.apps.hetero_pipeline import run_hetero_phi_ve_gemm


def main() -> int:
    rng = np.random.default_rng(7)
    m, k, n = 768, 512, 512
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    c, res = run_hetero_phi_ve_gemm(a, b, alpha=1.05, beta=-0.02, use_worker_pool=True)
    print(f"status={res.status} err={res.max_abs_err:.3e} wall={res.wall_sec:.4f}s")
    print(f"phi_gflops={res.phi_gflops:.2f} ve_mean_kernel={res.ve_kernel_gflops_mean:.2f}")
    for note in res.notes:
        print(" ", note)
    print("C shape", c.shape)
    return 0 if res.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
