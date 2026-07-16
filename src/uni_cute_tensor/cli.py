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


def _resolve_job_path(job: str) -> Path:
    job_path = Path(job)
    if job_path.is_file():
        return job_path
    candidates = [
        Path.cwd() / job,
        Path(__file__).resolve().parents[2] / "jobs" / job,
        Path(__file__).resolve().parents[2] / "jobs" / f"{job}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(job)


def run_main(argv: list[str] | None = None) -> None:
    """Run a job JSON/YAML file, or submit to uct-serve via --socket."""
    from uni_cute_tensor.runtime.job_runner import list_bundled_jobs, load_job, run_job
    from uni_cute_tensor.runtime.serve_protocol import (
        DEFAULT_SOCKET_PATH,
        client_call,
        request_health,
        request_ping,
        request_run,
        request_shutdown,
    )
    from uni_cute_tensor.runtime.session import shutdown_sessions

    p = argparse.ArgumentParser(prog="uct-run")
    p.add_argument(
        "job",
        nargs="?",
        help="Path to job .json/.yaml (or bundled name without path)",
    )
    p.add_argument("--list", action="store_true", help="List bundled jobs/")
    p.add_argument("--host-only", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write metrics.json here (default artifacts/jobs/<name>)",
    )
    p.add_argument(
        "--shutdown",
        action="store_true",
        help="Stop local sessions after run (or send shutdown to serve if --socket)",
    )
    p.add_argument(
        "--socket",
        nargs="?",
        const=DEFAULT_SOCKET_PATH,
        default=None,
        help=f"Submit to uct-serve Unix socket (default {DEFAULT_SOCKET_PATH})",
    )
    p.add_argument("--ping", action="store_true", help="Ping uct-serve (needs --socket)")
    p.add_argument("--health", action="store_true", help="Health check uct-serve")
    args = p.parse_args(argv)

    # socket control ops without job
    if args.socket and (args.ping or args.health or (args.shutdown and not args.job)):
        if args.ping:
            resp = client_call(args.socket, request_ping())
        elif args.health:
            resp = client_call(args.socket, request_health())
        else:
            resp = client_call(args.socket, request_shutdown())
        print(json.dumps(resp, indent=2))
        sys.exit(0 if resp.get("ok", False) else 1)

    if args.list or not args.job:
        jobs = list_bundled_jobs()
        print("Bundled jobs:")
        root = Path(__file__).resolve().parents[2] / "jobs"
        if not jobs and root.is_dir():
            jobs = sorted(root.glob("*.json"))
        for j in jobs:
            print(f"  {j.name}  ({j})")
        if not args.job:
            if args.list:
                sys.exit(0)
            p.error("job path required (or --list / --ping / --health)")
        sys.exit(0)

    try:
        job_path = _resolve_job_path(args.job)
    except FileNotFoundError:
        print(f"job not found: {args.job}", file=sys.stderr)
        sys.exit(2)

    out = args.out_dir
    if out is None:
        out = Path("artifacts") / "jobs" / job_path.stem

    # remote via serve
    if args.socket:
        job = load_job(job_path)
        resp = client_call(
            args.socket,
            request_run(job, id=job_path.stem, host_only=args.host_only),
        )
        if args.shutdown:
            try:
                client_call(args.socket, request_shutdown())
            except Exception:
                pass
        print("=== uct-run (via serve) ===")
        print(f"socket: {args.socket}")
        print(f"job: {job_path}")
        if not resp.get("ok"):
            print(f"error: {resp.get('error')}")
            sys.exit(1)
        result = resp.get("result") or {}
        print(f"type: {result.get('job_type')}  status: {result.get('status')}")
        print(f"backend: {result.get('backend')}  recommended: {result.get('recommended')}")
        print(f"wall: {result.get('wall_sec')}  err: {result.get('max_abs_err')}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(f"metrics: {out / 'metrics.json'}")
        sys.exit(0 if result.get("status") == "pass" else 1)

    # local run
    try:
        result = run_job(job_path, host_only=args.host_only, out_dir=out)
    finally:
        if args.shutdown:
            shutdown_sessions()

    print("=== uct-run ===")
    print(f"job: {job_path}")
    print(f"type: {result.job_type}  status: {result.status}")
    print(f"backend: {result.backend}  recommended: {result.recommended}")
    print(f"wall: {result.wall_sec:.4f}s  err: {result.max_abs_err:.3e}")
    for n in result.notes:
        print(f"  note: {n}")
    print(f"metrics: {out / 'metrics.json'}")
    thr = result.metrics.get("throughput_batches_per_sec")
    if thr is not None:
        hthr = result.metrics.get("host_dgemm_batches_per_sec")
        sp = result.metrics.get("speedup_vs_host_dgemm")
        extra = f"  host_thr={hthr:.2f}  vs_host={sp:.2f}x" if hthr is not None else ""
        othr = result.metrics.get("oneshot_batches_per_sec")
        if othr is not None:
            extra += f"  oneshot_thr={othr:.2f}  vs_oneshot={result.metrics.get('speedup_vs_oneshot', 0):.2f}x"
        print(f"thr: {thr:.2f} batches/s{extra}")
    sys.exit(0 if result.status == "pass" else 1)


def serve_main(argv: list[str] | None = None) -> None:
    """Start uct-serve Unix socket daemon (Phase 4 M2)."""
    from uni_cute_tensor.runtime.serve import run_server
    from uni_cute_tensor.runtime.serve_protocol import DEFAULT_SOCKET_PATH

    p = argparse.ArgumentParser(
        prog="uct-serve",
        description="Local Unix-socket job daemon (shared VE sessions). Not for public networks.",
    )
    p.add_argument(
        "--socket",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path (default {DEFAULT_SOCKET_PATH})",
    )
    p.add_argument("--host-only", action="store_true")
    p.add_argument(
        "--preload",
        default=None,
        help="Pre-open AVEO pin: N or M,N,K (e.g. 512 or 256,256,256)",
    )
    p.add_argument("--ve-node", type=int, default=1, help="VE node for preload pin")
    args = p.parse_args(argv)
    run_server(
        args.socket,
        host_only=args.host_only,
        preload=args.preload,
        ve_node=args.ve_node,
    )
