#!/usr/bin/env python3
"""Compare multi-VE DGEMM: combined vs shared-B split mode (wall + kernel)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, ve_toolchain_available
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


def main() -> int:
    if not ve_toolchain_available():
        print("VE toolchain missing")
        return 2
    ve = ve_device_names(discover_devices())
    if len(ve) < 1:
        print("no VE")
        return 2

    sizes = [(768, 768, 768), (1536, 1024, 1024), (2048, 1536, 1536)]
    rng = np.random.default_rng(0)
    print(f"devices={ve}")
    print(f"{'shape':>20} {'mode':>10} {'wall_s':>10} {'eff_GF':>10} {'ker_GF':>10} {'err':>12}")

    for m, k, n in sizes:
        a = rng.standard_normal((m, k))
        b = rng.standard_normal((k, n))
        flops = 2.0 * m * n * k
        for share_b in (False, True):
            t0 = time.perf_counter()
            c, plan, results = multi_ve_layout_dgemm(
                a, b, ve, parallel=True, share_b=share_b
            )
            wall = time.perf_counter() - t0
            err = float(np.max(np.abs(c - (a @ b))))
            ker = sum(r.gflops for r in results) / max(len(results), 1)
            mode = "share_b" if share_b else "combined"
            print(
                f"{m}x{k}x{n:>4} {mode:>10} {wall:10.4f} {flops/wall/1e9:10.2f} "
                f"{ker:10.2f} {err:12.3e}"
            )
            if err >= 1e-8:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
