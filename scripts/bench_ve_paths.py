#!/usr/bin/env python3
"""Compare multi-VE paths: oneshot file, worker pool, AVEO session."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_aveo import (
    aveo_available,
    multi_ve_aveo_dgemm,
    AveoSessionPool,
)
from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, ve_toolchain_available
from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


def main() -> int:
    if not ve_toolchain_available():
        print("no VE")
        return 2
    ve = ve_device_names(discover_devices())
    ve_ids = [int(d.replace("ve", "")) for d in ve]
    sizes = [(512, 512, 512), (1024, 1024, 1024)]
    rng = np.random.default_rng(0)
    print(f"devices={ve} aveo={aveo_available()}")
    print(f"{'shape':>16} {'path':>12} {'wall_s':>10} {'err':>12} {'kgflops':>10}")

    for m, k, n in sizes:
        a = rng.standard_normal((m, k))
        b = rng.standard_normal((k, n))
        ref = a @ b

        t0 = time.perf_counter()
        c, _, res = multi_ve_layout_dgemm(a, b, ve, share_b=True)
        wall = time.perf_counter() - t0
        err = float(np.max(np.abs(c - ref)))
        kg = sum(r.gflops for r in res) / max(len(res), 1)
        print(f"{m}x{k}x{n:>4} {'oneshot':>12} {wall:10.4f} {err:12.3e} {kg:10.1f}")

        with VeWorkerPool(ve_ids) as pool:
            multi_ve_layout_dgemm_pooled(
                rng.standard_normal((128, 128)),
                rng.standard_normal((128, 128)),
                ve,
                pool,
            )
            t0 = time.perf_counter()
            c, _, res = multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)
            wall = time.perf_counter() - t0
        err = float(np.max(np.abs(c - ref)))
        kg = sum(r.gflops for r in res) / max(len(res), 1)
        print(f"{m}x{k}x{n:>4} {'pool':>12} {wall:10.4f} {err:12.3e} {kg:10.1f}")

        if aveo_available():
            try:
                with AveoSessionPool(ve_ids) as pool:
                    # warm
                    multi_ve_aveo_dgemm(
                        rng.standard_normal((64, 64)),
                        rng.standard_normal((64, 64)),
                        ve,
                        pool=pool,
                    )
                    t0 = time.perf_counter()
                    c, _, res = multi_ve_aveo_dgemm(a, b, ve, pool=pool)
                    wall = time.perf_counter() - t0
                err = float(np.max(np.abs(c - ref)))
                kg = sum(r.gflops for r in res) / max(len(res), 1)
                print(f"{m}x{k}x{n:>4} {'aveo':>12} {wall:10.4f} {err:12.3e} {kg:10.1f}")
            except Exception as exc:
                print(f"{m}x{k}x{n:>4} {'aveo':>12} FAIL {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
