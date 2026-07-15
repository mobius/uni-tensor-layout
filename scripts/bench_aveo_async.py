#!/usr/bin/env python3
"""AVEO session sync vs async-phase timing breakdown."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available


def main() -> int:
    if not aveo_available():
        print("AVEO unavailable")
        return 2
    rng = np.random.default_rng(0)
    print(f"{'N':>6} {'mode':>10} {'total':>10} {'h2d':>10} {'kern':>10} {'d2h':>10} {'GF':>10} {'err':>10}")
    with AveoSessionPool([1]) as pool:
        for n in (256, 512, 1024, 1536):
            a = rng.standard_normal((n, n))
            b = rng.standard_normal((n, n))
            for async_phases in (False, True):
                # warm
                pool.dgemm(1, a[:64, :64], b[:64, :64], async_phases=async_phases)
                c, r = pool.dgemm(1, a, b, async_phases=async_phases)
                print(
                    f"{n:6d} {r.mode:>10} {r.elapsed_sec:10.4f} {r.h2d_sec:10.4f} "
                    f"{r.kernel_sec:10.4f} {r.d2h_sec:10.4f} {r.gflops:10.1f} "
                    f"{r.max_abs_err:10.2e}"
                )
                if r.status != "pass":
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
