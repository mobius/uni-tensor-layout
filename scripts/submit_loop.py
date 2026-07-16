#!/usr/bin/env python3
"""Submit the same job N times (local or via uct-serve) and report thr.

Examples:
  python scripts/submit_loop.py --job jobs/service_dense_stream.json --n 20 --host-only
  # with serve running:
  python scripts/submit_loop.py --job jobs/service_dense_stream.json --n 20 --socket
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Loop-submit jobs for service thr")
    ap.add_argument("--job", required=True, help="job json path")
    ap.add_argument("--n", type=int, default=10, help="number of submissions")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument(
        "--socket",
        nargs="?",
        const="/tmp/uct-serve.sock",
        default=None,
        help="submit via uct-serve (default path /tmp/uct-serve.sock)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "submit_loop.json")
    args = ap.parse_args()

    from uni_cute_tensor.runtime.job_runner import load_job, run_job
    from uni_cute_tensor.runtime.serve_protocol import client_call, request_run

    job_path = Path(args.job)
    if not job_path.is_file():
        cand = ROOT / args.job
        job_path = cand if cand.is_file() else ROOT / "jobs" / args.job
    job = load_job(job_path)
    walls: list[float] = []
    ok = 0
    t_all = time.perf_counter()
    for i in range(args.n):
        t0 = time.perf_counter()
        if args.socket:
            resp = client_call(
                args.socket,
                request_run(job, id=f"loop-{i}", host_only=args.host_only),
                timeout=600,
            )
            success = bool(resp.get("ok"))
            w = float((resp.get("result") or {}).get("wall_sec") or (time.perf_counter() - t0))
        else:
            r = run_job(job, host_only=args.host_only)
            success = r.status == "pass"
            w = r.wall_sec
        walls.append(w)
        if success:
            ok += 1
        print(f"[{i+1}/{args.n}] ok={success} wall={w:.4f}s")
    total = time.perf_counter() - t_all
    summary = {
        "job": str(job_path),
        "n": args.n,
        "ok": ok,
        "socket": args.socket,
        "host_only": args.host_only,
        "total_wall_sec": total,
        "submissions_per_sec": args.n / total if total > 0 else 0.0,
        "mean_job_wall_sec": statistics.mean(walls) if walls else None,
        "median_job_wall_sec": statistics.median(walls) if walls else None,
        "p95_job_wall_sec": sorted(walls)[max(0, int(0.95 * len(walls)) - 1)] if walls else None,
        "walls": walls,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("---")
    print(
        f"ok={ok}/{args.n}  submit/s={summary['submissions_per_sec']:.2f}  "
        f"mean_job={summary['mean_job_wall_sec']:.4f}s  -> {args.out}"
    )
    return 0 if ok == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
