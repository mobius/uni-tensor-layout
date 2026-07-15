#!/usr/bin/env python3
"""Sustained multi-VE throughput with optional power sampling."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.power import PowerCap
from uni_cute_tensor.power_sample import PowerSampler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--m", type=int, default=768)
    ap.add_argument("--k", type=int, default=768)
    ap.add_argument("--n", type=int, default=768)
    args = ap.parse_args()

    ve = ve_device_names(discover_devices())
    if not ve:
        print("no VE")
        return 2
    ve_ids = [int(d.replace("ve", "")) for d in ve]
    cap = PowerCap()
    ops = {d: "dgemm" for d in ve}
    if not cap.can_launch(list(ve), ops):
        print("PowerCap blocked")
        return 3
    cap.reserve(list(ve), ops)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((args.m, args.k))
    b = rng.standard_normal((args.k, args.n))
    flops = 2.0 * args.m * args.n * args.k

    sampler = PowerSampler(interval_sec=0.5)
    sampler.start()
    jobs = 0
    t_end = time.time() + args.seconds
    t0 = time.time()
    max_err = 0.0
    try:
        with VeWorkerPool(ve_ids) as pool:
            multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)  # warm
            while time.time() < t_end:
                c, _, res = multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)
                err = float(np.max(np.abs(c - a @ b)))
                max_err = max(max_err, err)
                jobs += 1
                if err >= 1e-8:
                    print("correctness fail")
                    return 1
    finally:
        cap.release(list(ve))
        power = sampler.stop()

    wall = time.time() - t0
    print(f"jobs={jobs} wall={wall:.2f}s jobs/s={jobs/wall:.3f}")
    print(f"effective_gflops={jobs * flops / wall / 1e9:.1f}")
    print(f"max_err={max_err:.3e}")
    print(f"power_samples={power['n']} mean_w={power['mean_w']} max_w={power['max_w']}")
    print(f"power_cap_backend={cap.backend} limit={cap.effective_limit:.0f}W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
