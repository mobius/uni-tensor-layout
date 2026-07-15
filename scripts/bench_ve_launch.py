#!/usr/bin/env python3
"""Compare multi-VE wall: cold ve_exec vs persistent worker pool."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, ve_toolchain_available
from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


def main() -> int:
    if not ve_toolchain_available():
        print("no VE toolchain")
        return 2
    ve = ve_device_names(discover_devices())
    if not ve:
        print("no VE")
        return 2
    ve_ids = [int(d.replace("ve", "")) for d in ve]
    sizes = [(512, 512, 512), (1024, 1024, 1024), (1536, 1024, 1024)]
    rng = np.random.default_rng(0)
    print(f"devices={ve}")
    print(f"{'shape':>18} {'mode':>12} {'wall_s':>10} {'err':>12}")

    # cold oneshot
    for m, k, n in sizes:
        a = rng.standard_normal((m, k))
        b = rng.standard_normal((k, n))
        t0 = time.perf_counter()
        c, _, _ = multi_ve_layout_dgemm(a, b, ve, share_b=True)
        wall = time.perf_counter() - t0
        err = float(np.max(np.abs(c - a @ b)))
        print(f"{m}x{k}x{n:>4} {'oneshot':>12} {wall:10.4f} {err:12.3e}")

    # pooled: start once, many jobs
    with VeWorkerPool(ve_ids) as pool:
        # warm
        a0 = rng.standard_normal((256, 256))
        b0 = rng.standard_normal((256, 256))
        multi_ve_layout_dgemm_pooled(a0, b0, ve, pool, share_b=True)
        for m, k, n in sizes:
            a = rng.standard_normal((m, k))
            b = rng.standard_normal((k, n))
            t0 = time.perf_counter()
            c, _, _ = multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)
            wall = time.perf_counter() - t0
            err = float(np.max(np.abs(c - a @ b)))
            print(f"{m}x{k}x{n:>4} {'pooled':>12} {wall:10.4f} {err:12.3e}")
            if err >= 1e-8:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
