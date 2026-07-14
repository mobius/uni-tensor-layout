#!/usr/bin/env python3
"""Demo: discover devices, partition a matrix, run host-simulated sharded GEMM."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cpu_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm
from cpu_cute_tensor.bridge.uni_adapter import discover_devices, plan_to_task_specs, ve_device_names
from cpu_cute_tensor.partition.multi_device import partition_to_devices
from cpu_cute_tensor.partition.pcie_cost import estimate_gemm_transfer_bytes, estimate_h2d_seconds


def main() -> int:
    devices = discover_devices()
    print("Devices:")
    for d in devices:
        print(f"  {d.name:8} kind={d.kind:4} online={d.online} src={d.source}")

    ve = ve_device_names(devices) or ["ve1", "ve2", "ve3"]
    m, k, n = 192, 128, 160
    plan = partition_to_devices(m, n, ve, prefer_kind="ve")
    print("\nPlacementPlan:")
    for s in plan.shards:
        print(
            f"  {s.device}: rows[{s.row_start}:{s.row_end}] "
            f"layout={s.layout} atom={s.atom_name}"
        )

    rng = np.random.default_rng(42)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    c_ref = host_dgemm_reference(a, b)
    c_sh = host_sharded_dgemm(a, b, plan)
    err = float(np.max(np.abs(c_ref - c_sh)))
    print(f"\nHost-sim sharded GEMM max_abs_err = {err:.3e}")

    xfer = estimate_gemm_transfer_bytes(m, n, k)
    print(f"Est H2D bytes={xfer['total_h2d']} ~ {estimate_h2d_seconds(xfer['total_h2d']):.4f}s")
    print("Task specs:", plan_to_task_specs(plan))
    return 0 if err < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
