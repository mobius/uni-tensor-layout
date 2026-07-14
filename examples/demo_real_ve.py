#!/usr/bin/env python3
"""Real multi-VE layout-planned NLC DGEMM demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cpu_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, ve_toolchain_available
from cpu_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


def main() -> int:
    if not ve_toolchain_available():
        print("VE toolchain unavailable", file=sys.stderr)
        return 2
    ve = ve_device_names(discover_devices())
    if not ve:
        print("no VE devices", file=sys.stderr)
        return 2

    m, k, n = 768, 512, 512
    rng = np.random.default_rng(7)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    c, plan, results = multi_ve_layout_dgemm(a, b, ve, parallel=True)
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
    print("shards:", [(s.device, s.row_start, s.row_end, str(s.layout)) for s in plan.shards])
    for r in results:
        print(f"{r.device}: {r.gflops:.1f} GFLOPS err={r.max_abs_err:.3e} status={r.status}")
    print(f"assemble max_abs_err={err:.3e}")
    return 0 if err < 1e-8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
