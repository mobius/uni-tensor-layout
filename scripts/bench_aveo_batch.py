#!/usr/bin/env python3
"""Compare AVEO per-batch session dgemm vs dual-buffer batch API."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available


def main() -> int:
    if not aveo_available():
        print("no aveo")
        return 2
    rng = np.random.default_rng(0)
    nbatch = 8
    print(f"{'N':>6} {'mode':>12} {'wall':>10} {'batch/s':>10} {'err':>12}")
    with AveoSessionPool([1]) as pool:
        for n in (256, 512, 1024):
            As = [rng.standard_normal((n, n)) for _ in range(nbatch)]
            Bs = [rng.standard_normal((n, n)) for _ in range(nbatch)]
            # warm
            pool.dgemm(1, As[0][:64, :64], Bs[0][:64, :64])

            t0 = time.perf_counter()
            max_err = 0.0
            for a, b in zip(As, Bs):
                c, r = pool.dgemm(1, a, b, async_phases=True)
                max_err = max(max_err, r.max_abs_err)
            wall = time.perf_counter() - t0
            print(
                f"{n:6d} {'loop-async':>12} {wall:10.4f} {nbatch/wall:10.3f} {max_err:12.3e}"
            )

            t0 = time.perf_counter()
            Cs, wall_b, err_b = pool.dgemm_batch(1, As, Bs)
            print(
                f"{n:6d} {'dual-buf':>12} {wall_b:10.4f} {nbatch/wall_b:10.3f} {err_b:12.3e}"
            )
            if max_err >= 1e-8 or err_b >= 1e-8:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
