#!/usr/bin/env python3
"""Unified performance snapshot table for Host / Phi / multi-VE / hetero / DataPlane.

Writes a markdown perf-gate under docs/impl/ when --write-doc is set (M1 default on).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
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
from uni_cute_tensor.runtime.dataplane import GemmRequest, create_dataplane


def _row(scope: str, case: str, metric: str, status: str, note: str = "") -> tuple:
    return (scope, case, metric, status, note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write-doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write docs/impl/<ts>_perf_gate.md (default: on)",
    )
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "48")
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    print("=== uni-cute-tensor performance summary ===")
    print(f"ICC license: {try_icc_license().get('ok')}")
    print(f"AVEO: {aveo_available()}  VE toolchain: {ve_toolchain_available()}")
    cap = PowerCap()
    print(f"PowerCap backend={cap.backend} limit={cap.effective_limit:.0f}W")
    ve = ve_device_names(discover_devices()) if ve_toolchain_available() else []
    print(f"VE devices: {ve}")
    rows: list[tuple] = []
    rng = np.random.default_rng(0)

    # Host
    for backend, n in (("auto", 512), ("openblas", 2048), ("avx512", 512)):
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        host_dgemm(a, b, backend=backend)  # warm
        _, r = host_dgemm(a, b, backend=backend)
        rows.append(
            _row("host", f"{backend}@{n}", f"{r.gflops:.1f} GF", r.status, f"err={r.max_abs_err:.1e}")
        )

    # Phi MKL
    if Path("/dev/mic0").exists() and try_icc_license().get("ok"):
        n = 1024
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        try:
            _, r = run_phi_dgemm(a, b, threads=244, backend="mkl")
            rows.append(
                _row("phi", f"mkl@{n}", f"{r.gflops:.1f} GF", r.status, f"err={r.max_abs_err:.1e}")
            )
        except Exception as exc:
            rows.append(_row("phi", "mkl", "n/a", "fail", str(exc)[:40]))

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
        rows.append(
            _row(
                "ve",
                f"oneshot@{n}",
                f"wall {w:.3f}s ker {kg:.0f}",
                "pass" if err < 1e-8 else "fail",
                f"err={err:.1e}",
            )
        )

        ve_ids = [int(d.replace("ve", "")) for d in ve]
        with VeWorkerPool(ve_ids) as pool:
            multi_ve_layout_dgemm_pooled(a[:128, :128], b[:128, :128], ve, pool)
            t0 = time.perf_counter()
            c, _, res = multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)
            w = time.perf_counter() - t0
        err = float(np.max(np.abs(c - a @ b)))
        kg = sum(r.gflops for r in res) / len(res)
        rows.append(
            _row(
                "ve",
                f"pool@{n}",
                f"wall {w:.3f}s ker {kg:.0f}",
                "pass" if err < 1e-8 else "fail",
                f"err={err:.1e}",
            )
        )

        if aveo_available():
            with AveoSessionPool(ve_ids) as pool:
                multi_ve_aveo_dgemm(a[:64, :64], b[:64, :64], ve, pool=pool)
                t0 = time.perf_counter()
                c, _, res = multi_ve_aveo_dgemm(a, b, ve, pool=pool)
                w = time.perf_counter() - t0
            err = float(np.max(np.abs(c - a @ b)))
            kg = sum(r.gflops for r in res) / len(res)
            rows.append(
                _row(
                    "ve",
                    f"aveo@{n}",
                    f"wall {w:.3f}s ker {kg:.0f}",
                    "pass" if err < 1e-8 else "fail",
                    f"err={err:.1e}",
                )
            )

            # pinned multi-batch vs session loop (single VE)
            nbatch, pn = 16, 512
            As = [rng.standard_normal((pn, pn)) for _ in range(nbatch)]
            Bs = [rng.standard_normal((pn, pn)) for _ in range(nbatch)]
            with AveoSessionPool([ve_ids[0]]) as pool:
                pool.dgemm(ve_ids[0], As[0][:32, :32], Bs[0][:32, :32])
                t0 = time.perf_counter()
                err_s = 0.0
                for aa, bb in zip(As, Bs):
                    _, rr = pool.dgemm(ve_ids[0], aa, bb)
                    err_s = max(err_s, rr.max_abs_err)
                wall_s = time.perf_counter() - t0
                pool.pin_all(pn, pn, pn)
                pool.dgemm_pinned(ve_ids[0], As[0][:32, :32], Bs[0][:32, :32])
                t0 = time.perf_counter()
                err_p = 0.0
                for aa, bb in zip(As, Bs):
                    _, rr = pool.dgemm_pinned(ve_ids[0], aa, bb)
                    err_p = max(err_p, rr.max_abs_err)
                wall_p = time.perf_counter() - t0
            rows.append(
                _row(
                    "ve",
                    f"session x{nbatch}@{pn}",
                    f"{nbatch / wall_s:.1f} b/s",
                    "pass" if err_s < 1e-8 else "fail",
                    f"wall={wall_s:.3f}s",
                )
            )
            rows.append(
                _row(
                    "ve",
                    f"pinned x{nbatch}@{pn}",
                    f"{nbatch / wall_p:.1f} b/s",
                    "pass" if err_p < 1e-8 else "fail",
                    f"wall={wall_p:.3f}s vs session {wall_s / wall_p:.2f}x",
                )
            )

    # DataPlane unified path (host + first VE backends)
    try:
        n = 512
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        with create_dataplane("host") as plane:
            plane.gemm(GemmRequest(a=a[:64, :64], b=b[:64, :64]))
            t0 = time.perf_counter()
            r = plane.gemm(GemmRequest(a=a, b=b))
            w = time.perf_counter() - t0
        rows.append(
            _row("dp", "host@512", f"wall {w:.3f}s {r.kernel_gflops:.0f}GF", r.status, f"err={r.max_abs_err:.1e}")
        )
        if ve:
            for be in ("file", "worker", "aveo", "aveo_pinned"):
                try:
                    if be == "aveo_pinned":
                        plane = create_dataplane(
                            "aveo_pinned", devices=ve[:1], pin_m=n, pin_n=n, pin_k=n
                        )
                    else:
                        plane = create_dataplane(be, devices=ve)
                    with plane:
                        plane.gemm(GemmRequest(a=a[:64, :64], b=b[:64, :64]))
                        t0 = time.perf_counter()
                        r = plane.gemm(GemmRequest(a=a, b=b))
                        w = time.perf_counter() - t0
                    rows.append(
                        _row(
                            "dp",
                            f"{be}@512",
                            f"wall {w:.3f}s {r.kernel_gflops:.0f}GF",
                            r.status,
                            f"err={r.max_abs_err:.1e}",
                        )
                    )
                except Exception as exc:
                    rows.append(_row("dp", be, "n/a", "fail", str(exc)[:48]))
    except Exception as exc:
        rows.append(_row("dp", "host", "n/a", "fail", str(exc)[:48]))

    # hetero
    if Path("/dev/mic0").exists() and len(ve) >= 1:
        As = [rng.standard_normal((384, 256)) for _ in range(3)]
        Bs = [rng.standard_normal((256, 256)) for _ in range(3)]
        try:
            res = run_hetero_multibatch(
                As, Bs, overlap=True, use_phi_worker=True, use_aveo=False, power_cap=cap
            )
            rows.append(
                _row(
                    "hetero",
                    "phi_worker+pool x3",
                    f"{res.throughput_batches_per_sec:.3f} b/s",
                    res.status,
                    f"err={res.max_abs_err:.1e}",
                )
            )
        except Exception as exc:
            rows.append(_row("hetero", "multibatch", "n/a", "fail", str(exc)[:40]))

    print()
    hdr = f"{'scope':<8} {'case':<24} {'metric':<28} {'status':<8} {'note'}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r[0]:<8} {r[1]:<24} {r[2]:<28} {r[3]:<8} {r[4]}")

    if args.write_doc:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ROOT / "docs" / "impl" / f"{ts}_perf_gate.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Performance gate",
            "",
            f"> Generated: {ts}",
            f"> Script: `scripts/bench_summary.py`",
            "",
            "| scope | case | metric | status | note |",
            "|-------|------|--------|--------|------|",
        ]
        for r in rows:
            lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- DataPlane backends: host | file | worker | aveo | aveo_pinned")
        lines.append("- `pinned` reuses session-resident VE buffers (no alloc/free per GEMM)")
        lines.append("- Compare `session` vs `pinned` batch/s for mid-size multi-batch PCIe path")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nwrote {path.relative_to(ROOT)}")

    failed = any(r[3] == "fail" for r in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
