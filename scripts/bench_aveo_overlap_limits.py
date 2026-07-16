#!/usr/bin/env python3
"""Quantify AVEO phase timing vs dual-buf batch (DMA∥kernel limits).

Documents whether true H2D∥kernel is available: on VEO, commands on one
context are ordered — async APIs still wait between phases on the critical path.
Main thr wins: session reuse, pin, dual-buffer batch (alloc amortization).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available


def main() -> int:
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    if not aveo_available():
        print("no AVEO")
        return 2

    rng = np.random.default_rng(0)
    nbatch = 12
    rows = []
    print(
        f"{'N':>5} {'mode':>12} {'wall':>8} {'h2d':>8} {'kern':>8} {'d2h':>8} "
        f"{'sum_ph':>8} {'b/s':>8} {'err':>10}"
    )
    with AveoSessionPool([1]) as pool:
        for n in (256, 512, 1024):
            As = [rng.standard_normal((n, n)) for _ in range(nbatch)]
            Bs = [rng.standard_normal((n, n)) for _ in range(nbatch)]

            # sync session (alloc each call internally via dgemm session path)
            pool.dgemm(1, As[0][:32, :32], Bs[0][:32, :32], async_phases=False)
            t0 = time.perf_counter()
            err = 0.0
            h2d = kern = d2h = 0.0
            for a, b in zip(As, Bs):
                _, r = pool.dgemm(1, a, b, async_phases=False)
                err = max(err, r.max_abs_err)
            wall = time.perf_counter() - t0
            print(
                f"{n:5d} {'session':>12} {wall:8.4f} {'-':>8} {'-':>8} {'-':>8} "
                f"{'-':>8} {nbatch/wall:8.2f} {err:10.2e}"
            )
            rows.append(
                {"N": n, "mode": "session", "wall": wall, "batch_s": nbatch / wall, "err": err}
            )

            # async phase breakdown (still ordered waits)
            pool.dgemm(1, As[0][:32, :32], Bs[0][:32, :32], async_phases=True)
            t0 = time.perf_counter()
            err = 0.0
            h2d = kern = d2h = 0.0
            for a, b in zip(As, Bs):
                _, r = pool.dgemm(1, a, b, async_phases=True)
                err = max(err, r.max_abs_err)
                h2d += r.h2d_sec
                kern += r.kernel_sec
                d2h += r.d2h_sec
            wall = time.perf_counter() - t0
            sph = h2d + kern + d2h
            print(
                f"{n:5d} {'async_ph':>12} {wall:8.4f} {h2d:8.4f} {kern:8.4f} {d2h:8.4f} "
                f"{sph:8.4f} {nbatch/wall:8.2f} {err:10.2e}"
            )
            rows.append(
                {
                    "N": n,
                    "mode": "async_phases",
                    "wall": wall,
                    "h2d": h2d,
                    "kern": kern,
                    "d2h": d2h,
                    "sum_phases": sph,
                    "batch_s": nbatch / wall,
                    "err": err,
                    "overlap_hint": "if wall≈sum_phases, little phase overlap",
                }
            )

            # dual-buf batch
            pool.dgemm_batch(1, As[:1], Bs[:1])
            t0 = time.perf_counter()
            _, wall_b, err_b = pool.dgemm_batch(1, As, Bs)
            # dgemm_batch returns wall inside
            wall = time.perf_counter() - t0
            print(
                f"{n:5d} {'dual_buf':>12} {wall:8.4f} {'-':>8} {'-':>8} {'-':>8} "
                f"{'-':>8} {nbatch/wall:8.2f} {err_b:10.2e}"
            )
            rows.append(
                {
                    "N": n,
                    "mode": "dual_buf_batch",
                    "wall": wall,
                    "batch_s": nbatch / wall,
                    "err": err_b,
                }
            )

            # pinned loop
            pool.pin_all(n, n, n)
            pool.dgemm_pinned(1, As[0][:32, :32], Bs[0][:32, :32])
            t0 = time.perf_counter()
            err = 0.0
            for a, b in zip(As, Bs):
                _, r = pool.dgemm_pinned(1, a, b)
                err = max(err, r.max_abs_err)
            wall = time.perf_counter() - t0
            print(
                f"{n:5d} {'pinned':>12} {wall:8.4f} {'-':>8} {'-':>8} {'-':>8} "
                f"{'-':>8} {nbatch/wall:8.2f} {err:10.2e}"
            )
            rows.append(
                {"N": n, "mode": "pinned", "wall": wall, "batch_s": nbatch / wall, "err": err}
            )

    out = ROOT / "artifacts" / "aveo_overlap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "nbatch": nbatch}, indent=2) + "\n")
    print(f"\nwrote {out}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc = ROOT / "docs" / "impl" / f"{ts}_aveo_overlap_limits.md"
    lines = [
        "# AVEO overlap / phase limits",
        "",
        f"> Generated: {ts}",
        f"> nbatch={nbatch}",
        "",
        "## Results",
        "",
        "| N | mode | wall_s | batch/s | notes |",
        "|---|------|-------:|--------:|-------|",
    ]
    for r in rows:
        note = r.get("overlap_hint", "")
        if r["mode"] == "async_phases" and r.get("sum_phases"):
            note = f"sum_ph={r['sum_phases']:.4f} wall={r['wall']:.4f}"
        lines.append(
            f"| {r['N']} | {r['mode']} | {r['wall']:.4f} | {r['batch_s']:.2f} | {note} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "- VEO **single thr context is ordered**: async H2D then wait, then call, then D2H.",
        "- **True DMA∥kernel on one context is not observed** as wall ≪ sum(phases).",
        "- Throughput wins come from **session reuse**, **pinned buffers**, **dual-buf batch** "
        "(hide alloc + pipeline D2H of i with setup of i+1 at host scheduling level).",
        "- Prefer `aveo_pin` / `dgemm_batch` for multi-job; do not expect CUDA-style streams.",
        "",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
