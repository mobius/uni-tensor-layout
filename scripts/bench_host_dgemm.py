#!/usr/bin/env python3
"""Benchmark host numpy vs AVX-512 OpenMP DGEMM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.host_dgemm import host_avx512_dgemm, host_numpy_dgemm


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "48")
    sizes = [256, 512, 1024, 1536, 2048]
    rng = np.random.default_rng(0)
    print(f"{'N':>6} {'numpy':>10} {'avx512':>10} {'err':>12}")
    for n in sizes:
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        _, rn = host_numpy_dgemm(a, b)
        _, ra = host_avx512_dgemm(a, b)
        print(
            f"{n:6d} {rn.gflops:10.2f} {ra.gflops:10.2f} {ra.max_abs_err:12.3e} "
            f"{ra.status}"
        )
        if ra.status != "pass":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
