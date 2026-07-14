#!/usr/bin/env python3
"""Run real hardware tests: Host tile GEMM, multi-VE NLC DGEMM, Phi peak smoke.

Usage:
  source env/.venv/bin/activate
  python scripts/run_real_tests.py
  python scripts/run_real_tests.py --m 512 --k 512 --n 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.host_dgemm import host_blocked_dgemm, host_numpy_dgemm
from uni_cute_tensor.backends.phi_smoke import run_phi_peak_smoke
from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, ve_toolchain_available
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


def main() -> int:
    ap = argparse.ArgumentParser(description="Real Host/VE/Phi tests")
    ap.add_argument("--m", type=int, default=384)
    ap.add_argument("--k", type=int, default=384)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-phi", action="store_true")
    ap.add_argument("--skip-ve", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {"m": args.m, "k": args.k, "n": args.n, "cases": {}}
    failed = 0

    print("=== discover ===")
    devices = discover_devices()
    for d in devices:
        print(f"  {d.name:8} kind={d.kind:4} online={d.online} src={d.source}")
    ve = ve_device_names(devices)
    print(f"VE targets: {ve}")

    rng = np.random.default_rng(args.seed)
    a = rng.standard_normal((args.m, args.k), dtype=np.float64)
    b = rng.standard_normal((args.k, args.n), dtype=np.float64)

    print("\n=== host numpy dgemm ===")
    c_ref, host_np = host_numpy_dgemm(a, b)
    print(
        f"  status={host_np.status}  {host_np.elapsed_sec:.4f}s  "
        f"{host_np.gflops:.2f} GFLOPS"
    )
    report["cases"]["host_numpy"] = host_np.__dict__

    print("\n=== host blocked (atom tile=8) ===")
    c_blk, host_blk = host_blocked_dgemm(a, b)
    print(
        f"  status={host_blk.status}  err={host_blk.max_abs_err:.3e}  "
        f"{host_blk.elapsed_sec:.4f}s  {host_blk.gflops:.2f} GFLOPS"
    )
    report["cases"]["host_blocked"] = host_blk.__dict__
    if host_blk.status != "pass":
        failed += 1

    if not args.skip_ve:
        print("\n=== multi-VE layout NLC dgemm ===")
        if not ve_toolchain_available():
            print("  SKIP: VE toolchain unavailable")
            report["cases"]["multi_ve"] = {"status": "skip", "reason": "toolchain"}
        elif not ve:
            print("  SKIP: no VE devices")
            report["cases"]["multi_ve"] = {"status": "skip", "reason": "no devices"}
        else:
            t0 = time.perf_counter()
            c_ve, plan, ve_results = multi_ve_layout_dgemm(a, b, ve, parallel=True)
            wall = time.perf_counter() - t0
            err = float(np.max(np.abs(c_ve - c_ref)))
            flops = 2.0 * args.m * args.n * args.k
            print(f"  plan shards: {[(s.device, s.rows) for s in plan.shards]}")
            for r in ve_results:
                print(
                    f"  {r.device}: status={r.status} err={r.max_abs_err:.3e} "
                    f"kernel={r.gflops:.2f} GFLOPS elapsed={r.elapsed_sec:.4f}s"
                )
                print(f"    {r.stdout.strip().splitlines()[0] if r.stdout else ''}")
            overall = "pass" if err < 1e-8 and all(r.status == "pass" for r in ve_results) else "fail"
            print(
                f"  assemble max_abs_err={err:.3e}  wall={wall:.4f}s  "
                f"effective={flops / wall / 1e9:.2f} GFLOPS  overall={overall}"
            )
            report["cases"]["multi_ve"] = {
                "status": overall,
                "max_abs_err": err,
                "wall_sec": wall,
                "effective_gflops": flops / wall / 1e9 if wall > 0 else 0.0,
                "shards": [
                    {
                        "device": r.device,
                        "status": r.status,
                        "gflops": r.gflops,
                        "elapsed_sec": r.elapsed_sec,
                        "max_abs_err": r.max_abs_err,
                    }
                    for r in ve_results
                ],
            }
            if overall != "pass":
                failed += 1

    if not args.skip_phi:
        print("\n=== Phi peak smoke ===")
        phi = run_phi_peak_smoke()
        print(
            f"  status={phi.status}  gflops={phi.gflops:.2f}  "
            f"theory%={phi.theory_pct:.1f}  binary={phi.binary or '(none)'}"
        )
        if phi.stderr and phi.status != "pass":
            print(f"  stderr: {phi.stderr[:200]}")
        report["cases"]["phi_peak"] = {
            "status": phi.status,
            "gflops": phi.gflops,
            "theory_pct": phi.theory_pct,
            "elapsed_sec": phi.elapsed_sec,
            "binary": phi.binary,
        }
        if phi.status == "fail":
            failed += 1
        # skip does not fail the suite

    report["failed"] = failed
    print(f"\n=== summary: failed={failed} ===")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
