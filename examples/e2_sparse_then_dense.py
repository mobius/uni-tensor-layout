#!/usr/bin/env python3
"""E2 — Sparse grid operator then dense post-processing (terminal example).

Story
-----
Structured 2D five-point stencil (Laplacian-style CSR) models a discrete PDE /
graph neighborhood operator. Apply multi-RHS SpMV (irregular), optional scale,
then dense GEMM post-transform on VE with auto placement — e.g. modal mixing
or multi-query projection of the sparse field.

Aligns with uni hetero_spmv *narrative*; implementation stays layout-runtime
focused (Host CSR + plan-driven dense).

Usage
-----
  python examples/e2_sparse_then_dense.py
  python examples/e2_sparse_then_dense.py --host-only --nx 32 --ny 32
  python examples/e2_sparse_then_dense.py --pattern random --density 0.02
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from _common import (
    add_common_args,
    dense_gemm_auto,
    make_power_cap,
    maybe_phi_scale,
    print_kv,
    resolve_ve_devices,
    setup_ve_env,
    write_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2 sparse SpMV then dense GEMM")
    add_common_args(p, default_out="artifacts/examples/e2_sparse_then_dense")
    p.add_argument(
        "--pattern",
        choices=("stencil5", "random"),
        default="stencil5",
        help="CSR structure (default: 2D five-point stencil)",
    )
    p.add_argument("--nx", type=int, default=48, help="grid x (stencil5)")
    p.add_argument("--ny", type=int, default=48, help="grid y (stencil5)")
    p.add_argument("--nrows", type=int, default=1024, help="random CSR rows")
    p.add_argument("--ncols", type=int, default=1024, help="random CSR cols")
    p.add_argument("--density", type=float, default=0.02)
    p.add_argument("--nrhs", type=int, default=128, help="SpMV right-hand sides")
    p.add_argument("--n_out", type=int, default=256, help="dense GEMM output cols")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--batches", type=int, default=1, help="repeat SpMV+GEMM jobs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_ve_env()
    from uni_cute_tensor.apps.spmv_dataprep import (
        csr_spmv,
        random_csr,
        stencil5_csr,
    )
    from uni_cute_tensor.runtime.timeline import timeline_scope

    rng = np.random.default_rng(args.seed)
    devices = resolve_ve_devices(args.devices, host_only=args.host_only)
    if args.backend == "host":
        devices = []
    cap = make_power_cap(not args.no_power_cap)

    if args.pattern == "stencil5":
        csr = stencil5_csr(args.nx, args.ny)
        pattern_note = f"stencil5 {args.nx}x{args.ny} n={csr.nrows} nnz={csr.nnz}"
    else:
        csr = random_csr(args.nrows, args.ncols, density=args.density, rng=rng)
        pattern_note = f"random {csr.nrows}x{csr.ncols} nnz={csr.nnz}"

    metrics: dict = {
        "example": "e2_sparse_then_dense",
        "pattern": args.pattern,
        "pattern_note": pattern_note,
        "nrhs": args.nrhs,
        "n_out": args.n_out,
        "batches": args.batches,
        "host_only": args.host_only,
        "phi": args.phi,
        "devices_available": devices,
        "nnz": csr.nnz,
    }

    max_err = 0.0
    with timeline_scope(job_id="e2") as tl:
        t_all = time.perf_counter()
        spmv_t = prep_t = gemm_t = 0.0
        last_gemm = None

        for bi in range(args.batches):
            X = rng.standard_normal((csr.ncols, args.nrhs))
            B = rng.standard_normal((args.nrhs, args.n_out))

            t0 = time.perf_counter()
            with tl.span("csr_spmv", "kernel", device="host"):
                Y = csr_spmv(csr, X)
            spmv_t += time.perf_counter() - t0

            t0 = time.perf_counter()
            with tl.span("scale", "host", device="phi0" if args.phi else "host"):
                Ys, prep_note = maybe_phi_scale(
                    Y, use_phi=args.phi, alpha=args.alpha, beta=args.beta
                )
            prep_t += time.perf_counter() - t0

            t0 = time.perf_counter()
            with tl.span("dense_gemm", "kernel", device=",".join(devices) or "host"):
                gemm = dense_gemm_auto(
                    Ys, B, devices, power_cap=cap, backend_hint=args.backend
                )
            gemm_t += time.perf_counter() - t0
            last_gemm = gemm

            # reference
            Y_ref = csr_spmv(csr, X)
            if args.alpha != 1.0 or args.beta != 0.0:
                Y_ref = args.alpha * Y_ref + args.beta
            C_ref = Y_ref @ B
            err = float(np.max(np.abs(gemm["c"] - C_ref)))
            max_err = max(max_err, err, gemm["max_abs_err"])
            if gemm["status"] != "pass" or err >= 1e-8:
                metrics["fail_batch"] = bi
                break

        wall = time.perf_counter() - t_all

        # host e2e baseline
        t_h = time.perf_counter()
        for bi in range(args.batches):
            # re-use last random only for size — regenerate for fair-ish cost
            X = rng.standard_normal((csr.ncols, args.nrhs))
            B = rng.standard_normal((args.nrhs, args.n_out))
            Y = csr_spmv(csr, X)
            Ys = args.alpha * Y + args.beta
            _ = Ys @ B
        host_wall = time.perf_counter() - t_h

        plan_path = None
        if last_gemm and last_gemm.get("plan") is not None:
            plan_path = Path(args.out_dir) / "plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            last_gemm["plan"].write_json(plan_path)

        tl_path = Path(args.out_dir) / "timeline.jsonl"
        tl_path.parent.mkdir(parents=True, exist_ok=True)
        tl.write_jsonl(tl_path)

    status = (
        "pass"
        if max_err < 1e-8 and last_gemm and last_gemm["status"] == "pass"
        else "fail"
    )
    metrics.update(
        {
            "status": status,
            "max_abs_err": max_err,
            "wall_sec": wall,
            "spmv_sec": spmv_t,
            "prep_sec": prep_t,
            "gemm_sec": gemm_t,
            "host_e2e_wall_sec": host_wall,
            "speedup_vs_host": (host_wall / wall) if wall > 0 else 0.0,
            "prep": prep_note if args.batches else "",
            "gemm_backend": last_gemm["backend"] if last_gemm else None,
            "gemm_strategy": last_gemm["strategy"] if last_gemm else None,
            "gemm_devices": last_gemm["devices"] if last_gemm else None,
            "timeline_phases": tl.summary(),
            "plan_json": str(plan_path) if plan_path else None,
            "note": (
                "speedup_vs_host may be <1 at mid-size (OpenBLAS vs PCIe); "
                "value is correct heterogeneous pipeline + reproducible plan"
            ),
        }
    )
    mpath = write_metrics(args.out_dir, metrics)
    print_kv(
        "E2 sparse then dense",
        [
            ("status", status),
            ("pattern", pattern_note),
            ("dims", f"SpMV RHS={args.nrhs}  GEMM out={args.n_out}  batches={args.batches}"),
            ("phases", f"spmv={spmv_t:.4f}s prep={prep_t:.4f}s gemm={gemm_t:.4f}s"),
            ("gemm", f"{metrics['gemm_backend']}/{metrics['gemm_strategy']} devs={metrics['gemm_devices']}"),
            ("wall", f"{wall:.4f}s  host_e2e={host_wall:.4f}s  vs_host={metrics['speedup_vs_host']:.2f}x"),
            ("max_abs_err", f"{max_err:.3e}"),
            ("metrics", mpath),
            ("timeline", tl_path),
        ],
        quiet=args.quiet,
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
