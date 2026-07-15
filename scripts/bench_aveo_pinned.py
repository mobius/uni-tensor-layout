#!/usr/bin/env python3
"""Compare AVEO session (alloc per call) vs pinned resident buffers (multi-GEMM)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


def main() -> int:
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    if not aveo_available():
        print("no aveo")
        return 2
    ve = ve_device_names(discover_devices())
    if not ve:
        print("no VE nodes")
        return 2
    node = int(ve[0].replace("ve", ""))
    rng = np.random.default_rng(0)
    nbatch = 16
    print(f"node=ve{node} nbatch={nbatch}")
    print(f"{'N':>6} {'mode':>12} {'wall':>10} {'batch/s':>10} {'err':>12}")
    with AveoSessionPool([node]) as pool:
        for n in (256, 512, 1024):
            As = [rng.standard_normal((n, n)) for _ in range(nbatch)]
            Bs = [rng.standard_normal((n, n)) for _ in range(nbatch)]

            pool.dgemm(node, As[0][:32, :32], Bs[0][:32, :32])
            t0 = time.perf_counter()
            err = 0.0
            for a, b in zip(As, Bs):
                _, r = pool.dgemm(node, a, b)
                err = max(err, r.max_abs_err)
            wall = time.perf_counter() - t0
            print(f"{n:6d} {'session':>12} {wall:10.4f} {nbatch/wall:10.2f} {err:12.3e}")

            pool.pin_all(n, n, n)
            pool.dgemm_pinned(node, As[0][:32, :32], Bs[0][:32, :32])
            t0 = time.perf_counter()
            err = 0.0
            for a, b in zip(As, Bs):
                _, r = pool.dgemm_pinned(node, a, b)
                err = max(err, r.max_abs_err)
            wall = time.perf_counter() - t0
            print(f"{n:6d} {'pinned':>12} {wall:10.4f} {nbatch/wall:10.2f} {err:12.3e}")
            if err >= 1e-8:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
