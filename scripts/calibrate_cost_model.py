#!/usr/bin/env python3
"""Measure Host/VE walls and write artifacts/calibration.json for the cost model.

Usage:
  python scripts/calibrate_cost_model.py
  python scripts/calibrate_cost_model.py --host-only --out artifacts/calibration.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.partition.cost_model import (
    CalibrationSample,
    calibrate_from_samples,
    save_calibration,
    set_calibration,
    get_device_model,
)


def _measure_host(m: int, n: int, k: int, reps: int = 2) -> CalibrationSample:
    from uni_cute_tensor.backends.host_dgemm import host_dgemm

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    host_dgemm(a, b, backend="auto")  # warm
    walls = []
    gflops = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        _, r = host_dgemm(a, b, backend="auto")
        walls.append(time.perf_counter() - t0)
        gflops = r.gflops
    return CalibrationSample(
        "host", m, n, k, wall_sec=float(np.median(walls)), kernel_gflops=gflops, note="host_auto"
    )


def _measure_ve_pin(m: int, n: int, k: int, node: int, reps: int = 3) -> CalibrationSample:
    from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available

    if not aveo_available():
        raise RuntimeError("no aveo")
    rng = np.random.default_rng(1)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    walls = []
    gflops = 0.0
    with AveoSessionPool([node]) as pool:
        pool.pin_all(m, n, k)
        pool.dgemm_pinned(node, a[:32, : min(32, k)], b[: min(32, k), :32])
        for _ in range(reps):
            t0 = time.perf_counter()
            _, r = pool.dgemm_pinned(node, a, b)
            walls.append(time.perf_counter() - t0)
            gflops = r.gflops
    nbytes = 8 * (m * k + k * n + m * n)
    return CalibrationSample(
        "ve",
        m,
        n,
        k,
        wall_sec=float(np.median(walls)),
        kernel_gflops=gflops,
        transfer_bytes=nbytes,
        note="aveo_pin",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "calibration.json")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--sizes", type=str, default="256,512,768,1024")
    args = ap.parse_args()

    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    os.environ.setdefault("OMP_NUM_THREADS", "48")

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    samples: list[CalibrationSample] = []

    print("=== calibrate: host ===")
    for n in sizes:
        s = _measure_host(n, n, n)
        samples.append(s)
        print(f"  host {n}^3 wall={s.wall_sec:.4f}s gflops={s.kernel_gflops:.1f}")

    if not args.host_only:
        try:
            from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
            from uni_cute_tensor.backends.ve_aveo import aveo_available

            ve = ve_device_names(discover_devices())
            if ve and aveo_available():
                node = int(ve[0].replace("ve", ""))
                print(f"=== calibrate: ve pin node={node} ===")
                for n in sizes:
                    try:
                        s = _measure_ve_pin(n, n, n, node)
                        samples.append(s)
                        print(
                            f"  ve {n}^3 wall={s.wall_sec:.4f}s gflops={s.kernel_gflops:.1f}"
                        )
                    except Exception as exc:
                        print(f"  ve {n}^3 FAIL {exc}")
            else:
                print("=== skip VE (no devices or no AVEO) ===")
        except Exception as exc:
            print(f"=== skip VE ({exc}) ===")

    # fit against defaults first for error baseline
    set_calibration({})
    report = calibrate_from_samples(samples)
    save_calibration(report, args.out)
    print()
    print(f"wrote {args.out}")
    print(f"fitted={report.fitted}")
    print(
        f"rel_error mean={report.mean_rel_error:.3f} max={report.max_rel_error:.3f} n={report.n_eval}"
    )
    for kind in report.fitted:
        md = get_device_model(kind)
        print(
            f"  {kind}: peak={md.peak_gflops:.1f}GF launch={md.launch_overhead_sec:.4f}s pcie={md.pcie_gbps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
