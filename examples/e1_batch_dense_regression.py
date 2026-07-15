#!/usr/bin/env python3
"""E1 — Batch dense regression / multi-RHS projection (terminal example).

Story
-----
Scientific batch path: column-normalize feature matrix X, then for each batch
right-hand side B compute C = X_norm @ B  (multi-RHS dense projection / design
matrix times coefficients). Prep can run on Host or Phi; dense GEMM uses auto
PlacementPlan on VE when available.

Metrics (honest)
----------------
- Correctness vs numpy
- Wall vs fixed multi-VE row plan (when ≥2 VEs)
- Multi-batch throughput (batches/s)

Usage
-----
  python examples/e1_batch_dense_regression.py
  python examples/e1_batch_dense_regression.py --host-only --batches 4 --m 256
  python examples/e1_batch_dense_regression.py --phi   # if mic + license OK
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from _common import (
    add_common_args,
    column_normalize,
    dense_gemm_auto,
    fixed_row_baseline,
    make_power_cap,
    maybe_phi_scale,
    print_kv,
    resolve_ve_devices,
    setup_ve_env,
    write_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 batch dense regression / projection")
    add_common_args(p, default_out="artifacts/examples/e1_batch_dense_regression")
    p.add_argument("--m", type=int, default=768, help="rows of X (samples)")
    p.add_argument("--k", type=int, default=512, help="features / inner dim")
    p.add_argument("--n", type=int, default=256, help="RHS columns per batch")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0, help="post-norm scale")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--compare-fixed", action="store_true", default=True)
    p.add_argument("--no-compare-fixed", action="store_false", dest="compare_fixed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_ve_env()
    from uni_cute_tensor.runtime.timeline import timeline_scope

    rng = np.random.default_rng(args.seed)
    devices = resolve_ve_devices(args.devices, host_only=args.host_only)
    if args.backend == "host":
        devices = []
    cap = make_power_cap(not args.no_power_cap)

    X = rng.standard_normal((args.m, args.k))
    batches_b = [rng.standard_normal((args.k, args.n)) for _ in range(args.batches)]

    metrics: dict = {
        "example": "e1_batch_dense_regression",
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "batches": args.batches,
        "host_only": args.host_only,
        "phi": args.phi,
        "devices_available": devices,
    }

    with timeline_scope(job_id="e1") as tl:
        t_all = time.perf_counter()
        Xn, mean, std = column_normalize(X)
        Xn, prep_note = maybe_phi_scale(Xn, use_phi=args.phi, alpha=args.alpha, beta=args.beta)

        max_err = 0.0
        walls: list[float] = []
        last = None
        for i, B in enumerate(batches_b):
            out = dense_gemm_auto(
                Xn, B, devices, power_cap=cap, backend_hint=args.backend
            )
            ref = Xn @ B
            err = float(np.max(np.abs(out["c"] - ref)))
            max_err = max(max_err, err, out["max_abs_err"])
            walls.append(out["wall_sec"])
            last = out
            if out["status"] != "pass" or err >= 1e-8:
                metrics["fail_batch"] = i
                break

        wall_total = time.perf_counter() - t_all
        thr = args.batches / wall_total if wall_total > 0 else 0.0

        # host pure numpy baseline (serial prep+gemm)
        t_h = time.perf_counter()
        Xn_h, _, _ = column_normalize(X)
        Xn_h = args.alpha * Xn_h + args.beta
        for B in batches_b:
            _ = Xn_h @ B
        host_wall = time.perf_counter() - t_h

        fixed = {"wall_sec": None, "status": "skip"}
        if args.compare_fixed and len(devices) >= 2 and last is not None:
            # single-batch fair compare on last shapes
            fixed = fixed_row_baseline(Xn, batches_b[0], devices)

        plan_path = None
        if last and last.get("plan") is not None:
            plan_path = Path(args.out_dir) / "plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            last["plan"].write_json(plan_path)

        tl_path = Path(args.out_dir) / "timeline.jsonl"
        tl_path.parent.mkdir(parents=True, exist_ok=True)
        tl.write_jsonl(tl_path)

    status = "pass" if max_err < 1e-8 and last and last["status"] == "pass" else "fail"
    metrics.update(
        {
            "status": status,
            "max_abs_err": max_err,
            "wall_total_sec": wall_total,
            "mean_batch_wall_sec": float(np.mean(walls)) if walls else None,
            "throughput_batches_per_sec": thr,
            "host_numpy_wall_sec": host_wall,
            "speedup_vs_host_numpy": (host_wall / wall_total) if wall_total > 0 else 0.0,
            "prep": prep_note,
            "gemm_backend": last["backend"] if last else None,
            "gemm_strategy": last["strategy"] if last else None,
            "gemm_devices": last["devices"] if last else None,
            "fixed_all_ve_wall_sec": fixed.get("wall_sec"),
            "fixed_all_ve_status": fixed.get("status"),
            "timeline_phases": tl.summary(),
            "plan_json": str(plan_path) if plan_path else None,
            "notes": (last or {}).get("notes", []) + [f"X mean0 std1 cols; prep={prep_note}"],
        }
    )
    if fixed.get("wall_sec") and last and last["wall_sec"] > 0:
        metrics["speedup_vs_fixed_all_ve"] = fixed["wall_sec"] / last["wall_sec"]

    mpath = write_metrics(args.out_dir, metrics)
    print_kv(
        "E1 batch dense regression",
        [
            ("status", status),
            ("shape", f"X={args.m}x{args.k}  B={args.k}x{args.n}  batches={args.batches}"),
            ("prep", prep_note),
            ("gemm", f"{metrics['gemm_backend']}/{metrics['gemm_strategy']} devs={metrics['gemm_devices']}"),
            ("wall_total", f"{wall_total:.4f}s  thr={thr:.2f} b/s"),
            ("host_numpy", f"{host_wall:.4f}s  vs_host={metrics['speedup_vs_host_numpy']:.2f}x"),
            (
                "fixed_all_ve",
                (
                    f"{fixed['wall_sec']:.4f}s  vs_fixed={metrics.get('speedup_vs_fixed_all_ve', 'n/a')}"
                    if fixed.get("wall_sec")
                    else fixed.get("status", "skip")
                ),
            ),
            ("max_abs_err", f"{max_err:.3e}"),
            ("metrics", mpath),
            ("timeline", tl_path),
        ],
        quiet=args.quiet,
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
