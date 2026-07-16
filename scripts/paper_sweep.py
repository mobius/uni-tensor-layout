#!/usr/bin/env python3
"""Paper experiment sweep: N × batches × backend × phi → metrics.jsonl + summary.md

No power metrics. Outputs under artifacts/paper/<exp_id>/.

  python scripts/paper_sweep.py --host-only --quick
  python scripts/paper_sweep.py --exp-id baseline_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller grid for smoke")
    ap.add_argument("--phi-try", action="store_true", help="also sweep phi=true (fallback ok)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os

    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    os.environ.setdefault("OMP_NUM_THREADS", "48")

    from uni_cute_tensor.runtime.job_runner import run_job
    from uni_cute_tensor.runtime.session import shutdown_sessions
    from uni_cute_tensor import __version__

    exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "artifacts" / "paper" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        sizes = [128, 256]
        batches_list = [1, 8]
    else:
        sizes = [256, 512, 768]
        batches_list = [1, 4, 16]

    backends = ["host"]
    if not args.host_only:
        backends += ["ve_pin", "oneshot"]
    phi_flags = [False, True] if args.phi_try else [False]

    # environment manifest (no secrets)
    from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
    from pathlib import Path as P

    ve = ve_device_names(discover_devices()) if not args.host_only else []
    manifest = {
        "exp_id": exp_id,
        "version": __version__,
        "seed": args.seed,
        "host_only": args.host_only,
        "ve_devices": ve,
        "phi_device": P("/dev/mic0").exists(),
        "sizes": sizes,
        "batches": batches_list,
        "backends": backends,
        "phi_flags": phi_flags,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    jsonl = out_dir / "metrics.jsonl"
    rows = []
    with jsonl.open("w", encoding="utf-8") as f:
        for n in sizes:
            for nb in batches_list:
                for be in backends:
                    for phi in phi_flags:
                        if be == "host":
                            job = {
                                "type": "dense_batch",
                                "m": n,
                                "n": n,
                                "k": n,
                                "batches": nb,
                                "backend": "host",
                                "phi": phi,
                                "compare_oneshot": False,
                                "seed": args.seed,
                            }
                            host_only = True
                        elif be == "ve_pin":
                            job = {
                                "type": "dense_batch",
                                "m": n,
                                "n": n,
                                "k": n,
                                "batches": nb,
                                "force_ve": True,
                                "phi": phi,
                                "compare_oneshot": nb >= 4,
                                "seed": args.seed,
                            }
                            host_only = False
                        else:  # oneshot approximated: force_ve + compare only
                            job = {
                                "type": "dense_batch",
                                "m": n,
                                "n": n,
                                "k": n,
                                "batches": max(nb, 3),
                                "force_ve": True,
                                "phi": phi,
                                "compare_oneshot": True,
                                "seed": args.seed,
                                "name": f"oneshot_probe_{n}_{nb}",
                            }
                            host_only = False

                        t0 = time.perf_counter()
                        try:
                            r = run_job(job, host_only=host_only or args.host_only)
                            status = r.status
                            rec = r.to_dict()
                        except Exception as exc:
                            status = "fail"
                            rec = {"error": str(exc), "status": "fail"}
                        wall = time.perf_counter() - t0
                        row = {
                            "N": n,
                            "batches": nb,
                            "backend": be,
                            "phi": phi,
                            "status": status,
                            "wall_sec": rec.get("wall_sec", wall),
                            "thr": (rec.get("metrics") or {}).get(
                                "throughput_batches_per_sec"
                            ),
                            "vs_host": (rec.get("metrics") or {}).get(
                                "speedup_vs_host_dgemm"
                            ),
                            "vs_oneshot": (rec.get("metrics") or {}).get(
                                "speedup_vs_oneshot"
                            ),
                            "prep_note": (rec.get("metrics") or {}).get("prep_note"),
                            "err": rec.get("max_abs_err"),
                            "path_backend": rec.get("backend"),
                        }
                        rows.append(row)
                        f.write(json.dumps(row) + "\n")
                        print(
                            f"N={n} B={nb} be={be} phi={phi} status={status} "
                            f"thr={row['thr']} vs_oneshot={row['vs_oneshot']}"
                        )
    shutdown_sessions()

    # summary markdown
    lines = [
        f"# Paper sweep `{exp_id}`",
        "",
        f"version={__version__}  host_only={args.host_only}  ve={ve}",
        "",
        "| N | batches | backend | phi | status | thr b/s | vs_host | vs_oneshot | prep |",
        "|--|---------:|---------|-----|--------|--------:|--------:|-----------:|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['N']} | {r['batches']} | {r['backend']} | {r['phi']} | {r['status']} | "
            f"{r['thr'] if r['thr'] is not None else '-'} | "
            f"{r['vs_host'] if r['vs_host'] is not None else '-'} | "
            f"{r['vs_oneshot'] if r['vs_oneshot'] is not None else '-'} | "
            f"{r.get('prep_note') or '-'} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- thr = batches/s of the chosen path (host_dgemm or shared AVEO pin).",
        "- vs_oneshot only meaningful for ve_pin multi-batch with compare_oneshot.",
        "- Mid-size Host OpenBLAS often wins vs_host; pin thr ≫ oneshot is the residency claim.",
        "",
        "Reproduce:",
        "```bash",
        f"python scripts/paper_sweep.py --exp-id {exp_id}"
        + (" --host-only" if args.host_only else ""),
        "```",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_dir}")
    return 0 if all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
