#!/usr/bin/env python3
"""Benchmark auto placement vs fixed row 3-VE; report prediction error."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.partition.cost_model import (
    choose_best_placement,
    estimate_gemm_placement,
    prediction_error_report,
)
from uni_cute_tensor.partition.multi_device import partition_matrix_rows
from uni_cute_tensor.partition.runner import execute_plan
from uni_cute_tensor.power import PowerCap


def main() -> int:
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    ve = ve_device_names(discover_devices())
    print(f"VE={ve}")
    if not ve:
        print("no VE — host-only smoke")
        rng = np.random.default_rng(0)
        a = rng.standard_normal((128, 96))
        b = rng.standard_normal((96, 80))
        ch = choose_best_placement(128, 80, 96, ["host"], backend="HOST_OBLAS")
        r = execute_plan(a, b, ch.plan, force_host=True)
        print(f"host auto strategy={ch.strategy} status={r.status} err={r.max_abs_err:.1e}")
        return 0 if r.status == "pass" else 1

    cap = PowerCap()
    rng = np.random.default_rng(0)
    print(
        f"{'case':<18} {'strat':<12} {'devs':<14} {'wall':>8} {'pred':>8} "
        f"{'rel_err':>8} {'status':>8}"
    )
    cases = [
        ("square512", 512, 512, 512),
        ("tall1024x256", 1024, 256, 512),  # tall: col may win
        ("wide256x1024", 256, 1024, 512),  # wide: row may win
        ("kheavy", 384, 384, 1536),  # large K
        ("square1536", 1536, 1536, 1536),  # large: multi-VE can win
    ]
    for name, m, n, k in cases:
        a = rng.standard_normal((m, k))
        b = rng.standard_normal((k, n))

        # fixed row all VE
        fixed = partition_matrix_rows(m, n, ve, k=k, backend="VE_NLC")
        t0 = time.perf_counter()
        r_fix = execute_plan(a, b, fixed, use_pool=True)
        w_fix = time.perf_counter() - t0

        # auto
        ch = choose_best_placement(m, n, k, ve, power_cap=cap, backend="VE_NLC")
        t0 = time.perf_counter()
        r_auto = execute_plan(a, b, ch.plan, use_pool=True)
        w_auto = time.perf_counter() - t0
        pred = prediction_error_report(w_auto, ch)

        devs = ",".join(ch.devices)
        print(
            f"{name:<18} {ch.strategy:<12} {devs:<14} {w_auto:8.4f} {ch.est_total_sec:8.4f} "
            f"{pred['rel_error']:8.2f} {r_auto.status:>8}"
        )
        print(
            f"{'  fixed_row':<18} {'row_blocks':<12} {','.join(ve):<14} {w_fix:8.4f} "
            f"{'':>8} {'':>8} {r_fix.status:>8}"
        )
        if r_auto.status != "pass" or r_fix.status != "pass":
            return 1
        # save plan sample
        out = ROOT / "artifacts" / f"plan_{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        ch.plan.write_json(out)

    # apples: show if auto ever beats fixed on wall for any case
    print("\nplans written under artifacts/plan_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
