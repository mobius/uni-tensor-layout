#!/usr/bin/env python3
"""E5 — Sustained job throughput with resident session (quasi-service CLI).

Story
-----
Keep a VE session warm (AVEO pin or worker pool) and stream N jobs (or run for
T seconds). Report jobs/s and compare to cold oneshot baseline.

Usage
-----
  python examples/e5_sustained_jobs.py --jobs 20
  python examples/e5_sustained_jobs.py --seconds 8 --mode pool
  python examples/e5_sustained_jobs.py --host-only --jobs 30 --m 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (
    add_common_args,
    make_power_cap,
    print_kv,
    resolve_ve_devices,
    setup_ve_env,
    write_metrics,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E5 sustained jobs / quasi-service")
    add_common_args(p, default_out="artifacts/examples/e5_sustained_jobs")
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--jobs", type=int, default=16, help="number of jobs (if no --seconds)")
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="if >0, run for this many seconds instead of fixed --jobs",
    )
    p.add_argument(
        "--mode",
        choices=("auto", "aveo_pin", "pool", "oneshot", "host"),
        default="auto",
        help="resident path; auto prefers aveo_pin then pool",
    )
    p.add_argument(
        "--cold-compare",
        action="store_true",
        default=True,
        help="also time a few cold oneshot jobs (default on)",
    )
    p.add_argument("--no-cold-compare", action="store_false", dest="cold_compare")
    p.add_argument("--cold-jobs", type=int, default=3)
    return p.parse_args(argv)


def _pick_mode(mode: str, devices: list[str], host_only: bool) -> str:
    if host_only or mode == "host" or not devices:
        return "host"
    if mode != "auto":
        return mode
    try:
        from uni_cute_tensor.backends.ve_aveo import aveo_available

        if aveo_available():
            return "aveo_pin"
    except Exception:
        pass
    return "pool"


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_ve_env()
    rng = np.random.default_rng(args.seed)
    devices = resolve_ve_devices(args.devices, host_only=args.host_only)
    mode = _pick_mode(args.mode, devices, args.host_only)
    cap = make_power_cap(not args.no_power_cap)

    a0 = rng.standard_normal((args.m, args.k))
    b0 = rng.standard_normal((args.k, args.n))
    flops = 2.0 * args.m * args.n * args.k
    max_err = 0.0
    jobs = 0
    walls: list[float] = []

    # ---- cold oneshot baseline ----
    cold_thr = None
    cold_mean = None
    if args.cold_compare and mode != "host" and devices:
        from uni_cute_tensor.backends.ve_dgemm import run_ve_dgemm_shard

        ve_id = int(devices[0].replace("ve", ""))
        cold_ws = []
        for _ in range(args.cold_jobs):
            a = rng.standard_normal((args.m, args.k))
            b = rng.standard_normal((args.k, args.n))
            t0 = time.perf_counter()
            c, r = run_ve_dgemm_shard(a, b, ve_id=ve_id)
            w = time.perf_counter() - t0
            cold_ws.append(w)
            max_err = max(max_err, float(np.max(np.abs(c - a @ b))), r.max_abs_err)
        cold_mean = float(np.mean(cold_ws))
        cold_thr = 1.0 / cold_mean if cold_mean > 0 else 0.0

    if mode in ("aveo_pin", "oneshot"):
        launch = devices[:1] if devices else ["host"]
    elif mode == "pool":
        launch = list(devices) if devices else ["host"]
    else:
        launch = ["host"]
    ops = {d: "dgemm" for d in launch}

    t_run0 = time.perf_counter()
    try:
        if not cap.can_launch(launch, ops) and devices:
            launch = devices[:1]
            ops = {d: "dgemm" for d in launch}
        with cap.guard(launch, ops):
            if mode == "host":
                from uni_cute_tensor.backends.host_dgemm import host_dgemm

                host_dgemm(a0, b0, backend="auto")  # warm
                deadline = (
                    time.perf_counter() + args.seconds if args.seconds > 0 else None
                )
                target = args.jobs if deadline is None else 10**9
                while jobs < target:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    a = rng.standard_normal((args.m, args.k))
                    b = rng.standard_normal((args.k, args.n))
                    t0 = time.perf_counter()
                    c, r = host_dgemm(a, b, backend="auto")
                    walls.append(time.perf_counter() - t0)
                    max_err = max(max_err, r.max_abs_err)
                    jobs += 1
                    if r.status != "pass":
                        break

            elif mode == "aveo_pin":
                from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available

                if not aveo_available():
                    raise RuntimeError("AVEO unavailable")
                node = int(launch[0].replace("ve", ""))
                with AveoSessionPool([node]) as pool:
                    pool.pin_all(args.m, args.n, args.k)
                    pool.dgemm_pinned(node, a0[:32, :32], b0[:32, :32])
                    deadline = (
                        time.perf_counter() + args.seconds if args.seconds > 0 else None
                    )
                    target = args.jobs if deadline is None else 10**9
                    while jobs < target:
                        if deadline is not None and time.perf_counter() >= deadline:
                            break
                        a = rng.standard_normal((args.m, args.k))
                        b = rng.standard_normal((args.k, args.n))
                        t0 = time.perf_counter()
                        c, r = pool.dgemm_pinned(node, a, b)
                        walls.append(time.perf_counter() - t0)
                        max_err = max(max_err, r.max_abs_err)
                        jobs += 1
                        if r.status != "pass":
                            break

            elif mode == "pool":
                from uni_cute_tensor.backends.ve_worker import (
                    VeWorkerPool,
                    multi_ve_layout_dgemm_pooled,
                )

                ve_ids = [int(d.replace("ve", "")) for d in launch]
                with VeWorkerPool(ve_ids) as pool:
                    multi_ve_layout_dgemm_pooled(
                        a0[:64, :64], b0[:64, :64], launch, pool, share_b=True
                    )
                    deadline = (
                        time.perf_counter() + args.seconds if args.seconds > 0 else None
                    )
                    target = args.jobs if deadline is None else 10**9
                    while jobs < target:
                        if deadline is not None and time.perf_counter() >= deadline:
                            break
                        a = rng.standard_normal((args.m, args.k))
                        b = rng.standard_normal((args.k, args.n))
                        t0 = time.perf_counter()
                        c, _, res = multi_ve_layout_dgemm_pooled(
                            a, b, launch, pool, share_b=True
                        )
                        walls.append(time.perf_counter() - t0)
                        err = float(np.max(np.abs(c - a @ b)))
                        max_err = max(max_err, err)
                        jobs += 1
                        if err >= 1e-8:
                            break

            elif mode == "oneshot":
                from uni_cute_tensor.backends.ve_dgemm import run_ve_dgemm_shard

                ve_id = int(launch[0].replace("ve", ""))
                deadline = (
                    time.perf_counter() + args.seconds if args.seconds > 0 else None
                )
                target = args.jobs if deadline is None else 10**9
                while jobs < target:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    a = rng.standard_normal((args.m, args.k))
                    b = rng.standard_normal((args.k, args.n))
                    t0 = time.perf_counter()
                    c, r = run_ve_dgemm_shard(a, b, ve_id=ve_id)
                    walls.append(time.perf_counter() - t0)
                    max_err = max(max_err, r.max_abs_err)
                    jobs += 1
                    if r.status != "pass":
                        break
            else:
                raise ValueError(mode)
    except Exception as exc:
        metrics = {
            "example": "e5_sustained_jobs",
            "status": "fail",
            "error": str(exc),
            "mode": mode,
        }
        write_metrics(args.out_dir, metrics)
        print_kv("E5 sustained", [("status", "fail"), ("error", exc)], quiet=args.quiet)
        return 1

    wall = time.perf_counter() - t_run0
    thr = jobs / wall if wall > 0 else 0.0
    mean_job = float(np.mean(walls)) if walls else 0.0
    status = "pass" if max_err < 1e-8 and jobs > 0 else "fail"
    speedup_vs_cold = (thr / cold_thr) if cold_thr and cold_thr > 0 else None

    metrics = {
        "example": "e5_sustained_jobs",
        "status": status,
        "mode": mode,
        "devices": launch,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "jobs": jobs,
        "wall_sec": wall,
        "jobs_per_sec": thr,
        "mean_job_wall_sec": mean_job,
        "effective_gflops": jobs * flops / wall / 1e9 if wall > 0 else 0.0,
        "max_abs_err": max_err,
        "cold_mean_wall_sec": cold_mean,
        "cold_jobs_per_sec": cold_thr,
        "speedup_vs_cold_oneshot": speedup_vs_cold,
        "power_cap_backend": cap.backend,
        "power_cap_limit_w": cap.effective_limit,
        "note": "Resident session thr should beat cold oneshot when mode=aveo_pin|pool",
    }
    mpath = write_metrics(args.out_dir, metrics)
    print_kv(
        "E5 sustained jobs",
        [
            ("status", status),
            ("mode", f"{mode} devices={launch}"),
            ("jobs", f"{jobs} wall={wall:.3f}s thr={thr:.2f} j/s mean={mean_job:.4f}s"),
            ("gflops_eff", f"{metrics['effective_gflops']:.1f}"),
            (
                "vs_cold",
                (
                    f"cold_thr={cold_thr:.2f} j/s → {speedup_vs_cold:.2f}x"
                    if speedup_vs_cold is not None
                    else "n/a"
                ),
            ),
            ("max_abs_err", f"{max_err:.3e}"),
            ("metrics", mpath),
        ],
        quiet=args.quiet,
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
