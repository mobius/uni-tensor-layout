#!/usr/bin/env python3
"""Benchmark optimized Phi DGEMM across sizes (correctness + GFLOPS)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.phi_dgemm import compile_phi_dgemm, run_phi_dgemm, try_icc_license


def main() -> int:
    lic = try_icc_license()
    print(f"ICC license ok={lic.get('ok')}")
    path, tag = compile_phi_dgemm(force=True, prefer_icc=True)
    print(f"binary={path} compiler={tag}")

    sizes = [128, 256, 512, 1024, 1536, 2048]
    # non-multiple of 8 to stress padding
    sizes += [384, 1000]
    rng = np.random.default_rng(0)
    print(f"{'N':>6} {'GFLOPS':>10} {'err':>12} {'sec':>10} {'status':>8}")
    worst_err = 0.0
    best = 0.0
    for n in sizes:
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        thr = 244 if n >= 512 else 120
        _, r = run_phi_dgemm(a, b, threads=thr)
        print(
            f"{n:6d} {r.gflops:10.2f} {r.max_abs_err:12.3e} "
            f"{r.elapsed_sec:10.4f} {r.status:>8}  {r.compiler}"
        )
        if r.status != "pass":
            return 1
        worst_err = max(worst_err, r.max_abs_err)
        best = max(best, r.gflops)
    print(f"best_gflops={best:.2f} worst_err={worst_err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
