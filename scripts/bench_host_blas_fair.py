#!/usr/bin/env python3
"""Fair host DGEMM comparison: numpy BLAS vs hand-written AVX-512 (multi-run median)."""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.host_dgemm import host_avx512_dgemm, host_numpy_dgemm


def _median_gflops(fn, a, b, runs: int = 5) -> tuple[float, float]:
    vals = []
    err = 0.0
    for _ in range(runs):
        _, r = fn(a, b)
        vals.append(r.gflops)
        err = max(err, r.max_abs_err)
    return statistics.median(vals), err


def main() -> int:
    threads = int(os.environ.get("OMP_NUM_THREADS", "48"))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    print(f"OMP_NUM_THREADS={threads}")
    print("numpy build:")
    try:
        np.__config__.show()
    except Exception as exc:
        print(" ", exc)
    print()

    sizes = [256, 512, 1024, 1536, 2048, 3072]
    rng = np.random.default_rng(0)
    print(f"{'N':>6} {'numpy_med':>12} {'avx512_med':>12} {'ratio':>8} {'err':>12}")
    for n in sizes:
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        # discard cold
        host_numpy_dgemm(a, b)
        host_avx512_dgemm(a, b, threads=threads)
        g_np, _ = _median_gflops(host_numpy_dgemm, a, b)
        g_ax, err = _median_gflops(
            lambda x, y: host_avx512_dgemm(x, y, threads=threads), a, b
        )
        ratio = g_ax / g_np if g_np > 0 else 0.0
        print(f"{n:6d} {g_np:12.2f} {g_ax:12.2f} {ratio:8.3f} {err:12.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
