#!/usr/bin/env python3
"""Unified performance snapshot table for Host / Phi / multi-VE / hetero."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.apps.hetero_pipeline import run_hetero_multibatch
from uni_cute_tensor.backends.host_dgemm import host_dgemm
from uni_cute_tensor.backends.phi_dgemm import run_phi_dgemm, try_icc_license
from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available, multi_ve_aveo_dgemm
from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, ve_toolchain_available
from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.power import PowerCap


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "48")
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    print("=== uni-cute-tensor performance summary ===")
    print(f"ICC license: {try_icc_license().get('ok')}")
    print(f"AVEO: {aveo_available()}  VE toolchain: {ve_toolchain_available()}")
    cap = PowerCap()
    print(f"PowerCap backend={cap.backend} limit={cap.effective_limit:.0f}W")
    ve = ve_device_names(discover_devices()) if ve_toolchain_available() else []
    print(f"VE devices: {ve}")
    rows = []
    rng = np.random.default_rng(0)

    # Host
    for backend, n in (("auto", 512), ("openblas", 2048), ("avx512", 512)):
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        host_dgemm(a, b, backend=backend)  # warm
        _, r = host_dgemm(a, b, backend=backend)
        rows.append(("host", f"{backend}@{n}", f"{r.gflops:.1f} GF", r.status, f"err={r.max_abs_err:.1e}"))

    # Phi MKL
    if Path("/dev/mic0").exists() and try_icc_license().get("ok"):
        n = 1024
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        try:
            _, r = run_phi_dgemm(a, b, threads=244, backend="mkl")
            rows.append(("phi", f"mkl@{n}", f"{r.gflops:.1f} GF", r.status, f"err={r.max_abs_err:.1e}"))
        except Exception as exc:
            rows.append(("phi", "mkl", "n/a", "fail", str(exc)[:40]))

    # multi-VE paths
    if len(ve) >= 1:
        n = 1024
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        t0 = time.perf_counter()
        c, _, res = multi_ve_layout_dgemm(a, b, ve, share_b=True)
        w = time.perf_counter() - t0
        err = float(np.max(np.abs(c - a @ b)))
        kg = sum(r.gflops for r in res) / len(res)
        rows.append(("ve", f"oneshot@{n}", f"wall {w:.3f}s ker {kg:.0f}", "pass" if err < 1e-8 else "fail", f"err={err:.1e}"))

        ve_ids = [int(d.replace("ve", "")) for d in ve]
        with VeWorkerPool(ve_ids) as pool:
            multi_ve_layout_dgemm_pooled(a[:128, :128], b[:128, :128], ve, pool)
            t0 = time.perf_counter()
            c, _, res = multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)
            w = time.perf_counter() - t0
        err = float(np.max(np.abs(c - a @ b)))
        kg = sum(r.gflops for r in res) / len(res)
        rows.append(("ve", f"pool@{n}", f"wall {w:.3f}s ker {kg:.0f}", "pass" if err < 1e-8 else "fail", f"err={err:.1e}"))

        if aveo_available():
            with AveoSessionPool(ve_ids) as pool:
                multi_ve_aveo_dgemm(a[:64, :64], b[:64, :64], ve, pool=pool)
                t0 = time.perf_counter()
                c, _, res = multi_ve_aveo_dgemm(a, b, ve, pool=pool)
                w = time.perf_counter() - t0
            err = float(np.max(np.abs(c - a @ b)))
            kg = sum(r.gflops for r in res) / len(res)
            rows.append(("ve", f"aveo@{n}", f"wall {w:.3f}s ker {kg:.0f}", "pass" if err < 1e-8 else "fail", f"err={err:.1e}"))

    # hetero
    if Path("/dev/mic0").exists() and len(ve) >= 1:
        As = [rng.standard_normal((384, 256)) for _ in range(3)]
        Bs = [rng.standard_normal((256, 256)) for _ in range(3)]
        try:
            res = run_hetero_multibatch(
                As, Bs, overlap=True, use_phi_worker=True, use_aveo=False, power_cap=cap
            )
            rows.append(
                (
                    "hetero",
                    "phi_worker+pool x3",
                    f"{res.throughput_batches_per_sec:.3f} b/s",
                    res.status,
                    f"err={res.max_abs_err:.1e}",
                )
            )
        except Exception as exc:
            rows.append(("hetero", "multibatch", "n/a", "fail", str(exc)[:40]))

    print()
    print(f"{'scope':<8} {'case':<22} {'metric':<28} {'status':<8} {'note'}")
    print("-" * 90)
    for r in rows:
        print(f"{r[0]:<8} {r[1]:<22} {r[2]:<28} {r[3]:<8} {r[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
