#!/usr/bin/env python3
"""E4 — Multi-task job DAG: prep → two parallel dense panels → host reduce.

Story
-----
A small job package (uni multi_task style):
  1. host_prep: column-normalize A
  2. gemm_left  || gemm_right: A_n @ B_left and A_n @ B_right (parallel when DAG allows)
  3. host_reduce: concatenate panels and check against reference

Uses uni TaskGraph when UNI_ROOT is available; otherwise local asyncio DAG.
PowerCap is applied at node granularity.

Usage
-----
  python examples/e4_job_dag.py
  python examples/e4_job_dag.py --host-only --m 128 --k 96 --n 64
  python examples/e4_job_dag.py --prefer-local   # force local DAG
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
    print_kv,
    resolve_ve_devices,
    setup_ve_env,
    write_metrics,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E4 multi-task job DAG")
    add_common_args(p, default_out="artifacts/examples/e4_job_dag")
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--k", type=int, default=384)
    p.add_argument("--n", type=int, default=256, help="total C columns (split left/right)")
    p.add_argument(
        "--prefer-local",
        action="store_true",
        help="Skip uni TaskGraph even if available",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_ve_env()
    from uni_cute_tensor.bridge.task_graph_bridge import run_task_graph

    rng = np.random.default_rng(args.seed)
    devices = resolve_ve_devices(args.devices, host_only=args.host_only)
    if args.backend == "host":
        devices = []
    cap = make_power_cap(not args.no_power_cap)

    A = rng.standard_normal((args.m, args.k))
    n_left = args.n // 2
    n_right = args.n - n_left
    B_left = rng.standard_normal((args.k, n_left))
    B_right = rng.standard_normal((args.k, n_right))

    state: dict = {}

    def do_prep():
        t0 = time.perf_counter()
        An, _, _ = column_normalize(A)
        state["An"] = An
        return {"status": "pass", "shape": list(An.shape), "wall": time.perf_counter() - t0}

    def do_gemm_left():
        An = state["An"]
        # pin each branch to a device if multiple available
        devs = devices[:1] if devices else []
        if len(devices) >= 2:
            devs = [devices[0]]
        out = dense_gemm_auto(An, B_left, devs, power_cap=cap, backend_hint=args.backend)
        state["C_left"] = out["c"]
        return {
            "status": out["status"],
            "wall": out["wall_sec"],
            "err": out["max_abs_err"],
            "backend": out["backend"],
            "devices": out["devices"],
            "strategy": out["strategy"],
        }

    def do_gemm_right():
        An = state["An"]
        devs = devices[:1] if devices else []
        if len(devices) >= 2:
            devs = [devices[1]]
        out = dense_gemm_auto(An, B_right, devs, power_cap=cap, backend_hint=args.backend)
        state["C_right"] = out["c"]
        return {
            "status": out["status"],
            "wall": out["wall_sec"],
            "err": out["max_abs_err"],
            "backend": out["backend"],
            "devices": out["devices"],
            "strategy": out["strategy"],
        }

    def do_reduce():
        t0 = time.perf_counter()
        Cl = state["C_left"]
        Cr = state["C_right"]
        C = np.concatenate([Cl, Cr], axis=1)
        An = state["An"]
        ref = An @ np.concatenate([B_left, B_right], axis=1)
        err = float(np.max(np.abs(C - ref)))
        state["C"] = C
        state["err"] = err
        return {
            "status": "pass" if err < 1e-8 else "fail",
            "wall": time.perf_counter() - t0,
            "err": err,
            "shape": list(C.shape),
        }

    left_dev = devices[0] if devices else "host"
    right_dev = devices[1] if len(devices) >= 2 else left_dev

    nodes = {
        "prep": {
            "device": "host",
            "op": "scale",
            "depends_on": [],
            "run_fn": do_prep,
            "estimated_watts": 100.0,
        },
        "gemm_left": {
            "device": left_dev,
            "op": "dgemm",
            "depends_on": ["prep"],
            "run_fn": do_gemm_left,
            "estimated_watts": 280.0,
        },
        "gemm_right": {
            "device": right_dev,
            "op": "dgemm",
            "depends_on": ["prep"],
            "run_fn": do_gemm_right,
            "estimated_watts": 280.0,
        },
        "reduce": {
            "device": "host",
            "op": "scale",
            "depends_on": ["gemm_left", "gemm_right"],
            "run_fn": do_reduce,
            "estimated_watts": 50.0,
        },
    }

    g = run_task_graph(
        nodes,
        power_cap=cap,
        prefer_uni=not args.prefer_local,
        critical_nodes=["prep", "gemm_left", "gemm_right", "reduce"],
    )

    # serial baseline (same work)
    t_s = time.perf_counter()
    An, _, _ = column_normalize(A)
    _ = An @ B_left
    _ = An @ B_right
    serial_wall = time.perf_counter() - t_s

    err = float(state.get("err", 1.0))
    status = "pass" if g.status == "pass" and err < 1e-8 else "fail"

    node_summary = {
        name: {
            "status": r.status,
            "device": r.device,
            "wall_sec": r.wall_sec,
            "payload_status": r.payload.get("status") if isinstance(r.payload, dict) else None,
            "backend": r.payload.get("backend") if isinstance(r.payload, dict) else None,
        }
        for name, r in g.results.items()
    }
    metrics = {
        "example": "e4_job_dag",
        "status": status,
        "dag_backend": g.backend,
        "dag_wall_sec": g.wall_sec,
        "serial_wall_sec": serial_wall,
        "speedup_vs_serial_host": serial_wall / g.wall_sec if g.wall_sec > 0 else 0.0,
        "max_abs_err": err,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "devices": devices,
        "nodes": node_summary,
        "notes": g.notes,
        "shape_C": list(state["C"].shape) if "C" in state else None,
    }
    mpath = write_metrics(args.out_dir, metrics)

    print_kv(
        "E4 job DAG",
        [
            ("status", status),
            ("dag", f"backend={g.backend} wall={g.wall_sec:.4f}s notes={g.notes}"),
            ("serial_host", f"{serial_wall:.4f}s vs={metrics['speedup_vs_serial_host']:.2f}x"),
            ("err", f"{err:.3e}"),
            ("graph", "prep → (gemm_left ∥ gemm_right) → reduce"),
            (
                "nodes",
                ", ".join(f"{n}:{node_summary[n]['status']}" for n in nodes),
            ),
            ("metrics", mpath),
        ],
        quiet=args.quiet,
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
