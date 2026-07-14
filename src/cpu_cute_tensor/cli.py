"""CLI entry points."""

from __future__ import annotations

import json
import sys

from cpu_cute_tensor.bridge.uni_adapter import discover_devices, plan_to_task_specs, ve_device_names
from cpu_cute_tensor.hw import probe
from cpu_cute_tensor.partition.multi_device import partition_to_devices
from cpu_cute_tensor.partition.pcie_cost import estimate_gemm_transfer_bytes, estimate_h2d_seconds


def check_hw_main() -> None:
    report = probe()
    devices = discover_devices()
    print("=== cpu-cute-tensor hardware probe ===")
    print(f"host: {report.hostname} ({report.machine})")
    print(f"cpu:  {report.cpu_model}")
    print(f"flags:{' '.join(report.cpu_flags_key) or '(none key)'}")
    print(f"python: {report.python_version}")
    print(f"nvidia: {report.has_nvidia}  amx: {report.has_amx}")
    print(f"phi_online: {report.phi_online}")
    print(f"ve_nodes: {', '.join(report.ve_nodes) or '(none)'}")
    print(f"ncc: {report.ncc_version}")
    print("--- devices ---")
    for d in devices:
        print(f"  {d.name:8} kind={d.kind:4} online={d.online} src={d.source}")
    for n in report.notes:
        print(f"note: {n}")
    ok = report.layout_algebra_ok
    print(f"layout_algebra: {'OK' if ok else 'FAIL'}")
    print(f"hetero_path: {'OK' if report.hetero_ok else 'DEGRADED (host-only)'}")
    sys.exit(0 if ok else 1)


def demo_partition_main() -> None:
    devices = discover_devices()
    ve = ve_device_names(devices)
    if not ve:
        ve = ["ve1", "ve2", "ve3"]  # logical demo without hardware
        print("warning: no VE found; using logical names", ve, file=sys.stderr)
    m, n, k = 192, 128, 96
    plan = partition_to_devices(m, n, ve, prefer_kind="ve")
    specs = plan_to_task_specs(plan)
    xfer = estimate_gemm_transfer_bytes(m, n, k)
    print(json.dumps(
        {
            "plan": plan.to_dict(),
            "tasks": specs,
            "transfer": xfer,
            "est_h2d_sec": estimate_h2d_seconds(xfer["total_h2d"]),
        },
        indent=2,
    ))
