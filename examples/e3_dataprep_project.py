#!/usr/bin/env python3
"""E3 — Dirty feature clean → standardize → dense projection (terminal example).

Story
-----
ML/scientific dataprep: synthetic dirty matrix (NaN + outliers) is cleaned on
Host, column-standardized (Host or Phi), then projected Y = X_clean @ W with
auto-placed VE GEMM (PCA-style projection with fixed orthonormal W).

Aligns with uni hetero_dataprep narrative without full PCA eigensolve.

Usage
-----
  python examples/e3_dataprep_project.py
  python examples/e3_dataprep_project.py --host-only --m 256 --k 128 --n-out 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (
    add_common_args,
    column_normalize,
    dense_gemm_auto,
    make_power_cap,
    maybe_phi_scale,
    print_kv,
    resolve_ve_devices,
    setup_ve_env,
    write_metrics,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3 dataprep clean + project")
    add_common_args(p, default_out="artifacts/examples/e3_dataprep_project")
    p.add_argument("--m", type=int, default=1024, help="samples / rows")
    p.add_argument("--k", type=int, default=512, help="features")
    p.add_argument("--n-out", type=int, default=128, help="projection dim")
    p.add_argument("--nan-frac", type=float, default=0.02)
    p.add_argument("--outlier-frac", type=float, default=0.01)
    p.add_argument("--clip-sigma", type=float, default=6.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_ve_env()
    from uni_cute_tensor.apps.dataprep_clean import (
        clean_features,
        make_dirty_matrix,
        random_projection_matrix,
    )
    from uni_cute_tensor.runtime.timeline import timeline_scope

    rng = np.random.default_rng(args.seed)
    devices = resolve_ve_devices(args.devices, host_only=args.host_only)
    if args.backend == "host":
        devices = []
    cap = make_power_cap(not args.no_power_cap)

    dirty = make_dirty_matrix(
        args.m,
        args.k,
        rng=rng,
        nan_frac=args.nan_frac,
        outlier_frac=args.outlier_frac,
    )
    W = random_projection_matrix(args.k, args.n_out, rng=rng)

    with timeline_scope(job_id="e3") as tl:
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        with tl.span("clean", "host", device="host"):
            cleaned, cstats = clean_features(dirty, clip_sigma=args.clip_sigma)
        clean_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        with tl.span("normalize", "host", device="host"):
            Xn, _, _ = column_normalize(cleaned)
        Xn, prep_note = maybe_phi_scale(Xn, use_phi=args.phi)
        prep_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        with tl.span("project_gemm", "kernel", device=",".join(devices) or "host"):
            gemm = dense_gemm_auto(
                Xn, W, devices, power_cap=cap, backend_hint=args.backend
            )
        gemm_sec = time.perf_counter() - t0
        wall = time.perf_counter() - t_all

        Y_ref = Xn @ W
        err = float(np.max(np.abs(gemm["c"] - Y_ref)))
        # Frobenius energy: projection should preserve roughly if W thin ortho cols
        fro_x = float(np.linalg.norm(Xn, "fro"))
        fro_y = float(np.linalg.norm(gemm["c"], "fro"))
        energy_ratio = fro_y / fro_x if fro_x > 0 else 0.0

        # host e2e baseline
        t_h = time.perf_counter()
        c2, st2 = clean_features(dirty, clip_sigma=args.clip_sigma)
        xn2, _, _ = column_normalize(c2)
        _ = xn2 @ W
        host_wall = time.perf_counter() - t_h

        plan_path = None
        if gemm.get("plan") is not None:
            plan_path = Path(args.out_dir) / "plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            gemm["plan"].write_json(plan_path)
        tl_path = Path(args.out_dir) / "timeline.jsonl"
        tl_path.parent.mkdir(parents=True, exist_ok=True)
        tl.write_jsonl(tl_path)

    status = "pass" if err < 1e-8 and gemm["status"] == "pass" else "fail"
    # cleaned should have no NaN
    if np.isnan(cleaned).any():
        status = "fail"

    metrics = {
        "example": "e3_dataprep_project",
        "status": status,
        "m": args.m,
        "k": args.k,
        "n_out": args.n_out,
        "nan_filled": cstats.nan_count,
        "clipped": cstats.clipped_count,
        "prep": prep_note,
        "gemm_backend": gemm["backend"],
        "gemm_strategy": gemm["strategy"],
        "gemm_devices": gemm["devices"],
        "max_abs_err": err,
        "energy_ratio_fro": energy_ratio,
        "wall_sec": wall,
        "clean_sec": clean_sec,
        "prep_sec": prep_sec,
        "gemm_sec": gemm_sec,
        "host_e2e_wall_sec": host_wall,
        "speedup_vs_host": host_wall / wall if wall > 0 else 0.0,
        "timeline_phases": tl.summary(),
        "plan_json": str(plan_path) if plan_path else None,
        "note": "energy_ratio ~1 if W is square orthonormal; thinner W reduces it",
    }
    mpath = write_metrics(args.out_dir, metrics)
    print_kv(
        "E3 dataprep project",
        [
            ("status", status),
            ("shape", f"dirty {args.m}x{args.k} → Y {args.m}x{args.n_out}"),
            ("clean", f"nan={cstats.nan_count} clipped={cstats.clipped_count}"),
            ("prep", prep_note),
            ("gemm", f"{gemm['backend']}/{gemm['strategy']} {gemm['devices']}"),
            ("err", f"{err:.3e}  energy_fro={energy_ratio:.4f}"),
            ("wall", f"{wall:.4f}s host={host_wall:.4f}s vs={metrics['speedup_vs_host']:.2f}x"),
            ("metrics", mpath),
        ],
        quiet=args.quiet,
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
