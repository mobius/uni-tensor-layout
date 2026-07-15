#!/usr/bin/env python3
"""Benchmark Phi DGEMM backends (MKL vs IMCI) across sizes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.phi_dgemm import (
    compile_phi_dgemm,
    run_phi_dgemm,
    try_icc_license,
)


def main() -> int:
    print(f"ICC license ok={try_icc_license().get('ok')}")
    backends = ["mkl", "imci"]
    sizes = [256, 512, 1024, 1536, 2048]
    rng = np.random.default_rng(0)

    for be in backends:
        try:
            path, tag = compile_phi_dgemm(force=True, backend=be)
            print(f"\n=== backend={be} binary={path.name} tag={tag} ===")
        except Exception as exc:
            print(f"\n=== backend={be} SKIP compile: {exc} ===")
            continue
        print(f"{'N':>6} {'GFLOPS':>10} {'err':>12} {'sec':>10} {'status':>8}")
        best = 0.0
        for n in sizes:
            a = rng.standard_normal((n, n))
            b = rng.standard_normal((n, n))
            thr = 244 if n >= 512 else 120
            try:
                _, r = run_phi_dgemm(a, b, threads=thr, backend=be)
            except Exception as exc:
                print(f"{n:6d} FAIL {exc}")
                return 1
            print(
                f"{n:6d} {r.gflops:10.2f} {r.max_abs_err:12.3e} "
                f"{r.elapsed_sec:10.4f} {r.status:>8}"
            )
            if r.status != "pass":
                return 1
            best = max(best, r.gflops)
        print(f"best_gflops[{be}]={best:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
