"""CLI entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uni_cute_tensor.bridge.uni_adapter import discover_devices, plan_to_task_specs, ve_device_names
from uni_cute_tensor.hw import probe
from uni_cute_tensor.partition.multi_device import partition_to_devices
from uni_cute_tensor.partition.pcie_cost import estimate_gemm_transfer_bytes, estimate_h2d_seconds


def check_hw_main() -> None:
    report = probe()
    devices = discover_devices()
    print("=== uni-cute-tensor hardware probe ===")
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


def recommend_main(argv: list[str] | None = None) -> None:
    """Print dispatch recommendation for a GEMM-shaped job."""
    from uni_cute_tensor.partition.dispatch import recommend_backend

    p = argparse.ArgumentParser(prog="uct-recommend")
    p.add_argument("-m", type=int, default=512)
    p.add_argument("-n", type=int, default=512)
    p.add_argument("-k", type=int, default=512)
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--mode", choices=("pin", "pool", "oneshot"), default="pin")
    p.add_argument("--phi", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    devices = discover_devices()
    ve = ve_device_names(devices)
    phi = any(d.kind == "phi" and d.online for d in devices)
    rec = recommend_backend(
        args.m,
        args.n,
        args.k,
        batches=args.batches,
        ve_devices=ve,
        phi_available=args.phi or phi,
        prefer_mode=args.mode,
    )
    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print("=== uct-recommend ===")
        print(f"problem: {args.m}x{args.k} @ {args.k}x{args.n}  batches={args.batches}")
        print(f"recommended: **{rec.recommended}**  confidence={rec.confidence}")
        print(f"reason: {rec.reason}")
        print(f"calibration_loaded: {rec.calibration_loaded}  ve={rec.ve_available}")
        print("--- estimates ---")
        for e in rec.estimates:
            print(f"  {e.backend:<12} wall={e.est_wall_sec:.4f}s  {'; '.join(e.notes[:2])}")
    sys.exit(0)
