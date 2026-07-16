#!/usr/bin/env python3
"""Host vs VE pin/pool break-even scan + recommend_backend table.

Writes docs/impl/<ts>_breakeven.md and artifacts/breakeven.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.partition.cost_model import try_autoload_calibration
from uni_cute_tensor.partition.dispatch import recommend_backend


def _time_host(m, n, k, batches: int) -> float:
    from uni_cute_tensor.backends.host_dgemm import host_dgemm

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k))
    bs = [rng.standard_normal((k, n)) for _ in range(batches)]
    host_dgemm(a, bs[0], backend="auto")
    t0 = time.perf_counter()
    for b in bs:
        host_dgemm(a, b, backend="auto")
    return time.perf_counter() - t0


def _time_ve_pin(m, n, k, batches: int, node: int) -> float:
    from uni_cute_tensor.backends.ve_aveo import AveoSessionPool

    rng = np.random.default_rng(1)
    a = rng.standard_normal((m, k))
    bs = [rng.standard_normal((k, n)) for _ in range(batches)]
    with AveoSessionPool([node]) as pool:
        pool.pin_all(m, n, k)
        pool.dgemm_pinned(node, a[:32, :32], bs[0][:32, :32])
        t0 = time.perf_counter()
        for b in bs:
            pool.dgemm_pinned(node, a, b)
        return time.perf_counter() - t0


def _time_ve_pool(m, n, k, batches: int, devices: list[str]) -> float:
    from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled

    rng = np.random.default_rng(2)
    a = rng.standard_normal((m, k))
    bs = [rng.standard_normal((k, n)) for _ in range(batches)]
    ve_ids = [int(d.replace("ve", "")) for d in devices]
    with VeWorkerPool(ve_ids) as pool:
        multi_ve_layout_dgemm_pooled(
            a[:64, :64], bs[0][:64, :64], devices, pool, share_b=True
        )
        t0 = time.perf_counter()
        for b in bs:
            multi_ve_layout_dgemm_pooled(a, b, devices, pool, share_b=True)
        return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,512,768,1024")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--write-doc", action=argparse.BooleanOptionalAction, default=True)
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

    cal = try_autoload_calibration()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    batches_list = [int(x) for x in args.batches.split(",") if x.strip()]

    ve: list[str] = []
    node = 1
    if not args.host_only:
        try:
            from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
            from uni_cute_tensor.backends.ve_aveo import aveo_available

            ve = ve_device_names(discover_devices()) if aveo_available() else []
            if ve:
                node = int(ve[0].replace("ve", ""))
        except Exception:
            ve = []

    rows = []
    print(
        f"{'N':>5} {'B':>4} {'host':>9} {'pin':>9} {'pool':>9} "
        f"{'rec':>6} {'est_host':>9} {'est_pin':>9} {'conf':>6}"
    )
    print("-" * 80)

    for n in sizes:
        for nb in batches_list:
            rec = recommend_backend(
                n, n, n, batches=nb, ve_devices=ve[:1] if ve else [], prefer_mode="pin"
            )
            est = {e.backend: e.est_wall_sec for e in rec.estimates}
            try:
                wh = _time_host(n, n, n, nb)
            except Exception as exc:
                wh = float("nan")
                print(f"host fail {exc}")
            wp = float("nan")
            wpool = float("nan")
            if ve:
                try:
                    wp = _time_ve_pin(n, n, n, nb, node)
                except Exception as exc:
                    print(f"pin fail N={n} B={nb}: {exc}")
                if len(ve) >= 1 and nb <= 8:  # pool heavier; skip huge loops
                    try:
                        wpool = _time_ve_pool(n, n, n, nb, ve[:1])
                    except Exception as exc:
                        print(f"pool fail N={n} B={nb}: {exc}")

            winner = "host"
            meas = {"host": wh, "pin": wp, "pool": wpool}
            finite = {k: v for k, v in meas.items() if v == v and v > 0}
            if finite:
                winner = min(finite, key=finite.get)  # type: ignore

            row = {
                "N": n,
                "batches": nb,
                "wall_host": wh,
                "wall_pin": wp,
                "wall_pool": wpool,
                "measured_winner": winner,
                "recommend": rec.recommended,
                "est_host": est.get("host"),
                "est_pin": est.get("ve:pin"),
                "confidence": rec.confidence,
                "reason": rec.reason,
            }
            rows.append(row)
            print(
                f"{n:5d} {nb:4d} {wh:9.4f} {wp:9.4f} {wpool:9.4f} "
                f"{rec.recommended:>6} {est.get('host', 0):9.4f} "
                f"{est.get('ve:pin', 0) or 0:9.4f} {rec.confidence:>6}"
            )

    out_json = ROOT / "artifacts" / "breakeven.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration_loaded": cal,
        "ve": ve,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_json}")

    if args.write_doc:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc = ROOT / "docs" / "impl" / f"{ts}_breakeven.md"
        lines = [
            "# Host vs VE break-even",
            "",
            f"> Generated: {ts}",
            f"> Calibration loaded: **{cal}**",
            f"> VE devices: `{ve}`",
            "",
            "## Measured walls (seconds)",
            "",
            "| N | batches | host | AVEO pin | pool | measured winner | recommend | conf |",
            "|---|--------:|-----:|---------:|-----:|-----------------|-----------|------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['N']} | {r['batches']} | {r['wall_host']:.4f} | "
                f"{r['wall_pin'] if r['wall_pin']==r['wall_pin'] else float('nan'):.4f} | "
                f"{r['wall_pool'] if r['wall_pool']==r['wall_pool'] else float('nan'):.4f} | "
                f"{r['measured_winner']} | {r['recommend']} | {r['confidence']} |"
            )
        lines += [
            "",
            "## How to read",
            "",
            "- **host**: OpenBLAS/auto Host GEMM, multi-batch loop",
            "- **AVEO pin**: resident buffers, multi-batch (best VE multi-job path)",
            "- **recommend**: `recommend_backend` using cost model (calibrated if available)",
            "- Mid-size single batch often favors **host** on this machine; pin wins more as **batches** grow",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/calibrate_cost_model.py",
            "python scripts/bench_breakeven.py",
            "```",
            "",
        ]
        doc.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {doc.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
