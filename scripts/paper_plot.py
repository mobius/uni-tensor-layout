#!/usr/bin/env python3
"""Optional plots from paper_sweep metrics.jsonl (needs matplotlib)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_dir", type=Path, help="artifacts/paper/<exp_id>")
    args = ap.parse_args()
    jsonl = args.paper_dir / "metrics.jsonl"
    if not jsonl.is_file():
        print(f"missing {jsonl}", file=sys.stderr)
        return 2
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install '.[viz]'", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    # thr vs batches for each N, backend=ve_pin phi=False
    by_n: dict[int, list] = defaultdict(list)
    for r in rows:
        if r.get("backend") == "ve_pin" and not r.get("phi") and r.get("thr"):
            by_n[int(r["N"])].append((int(r["batches"]), float(r["thr"])))
    if by_n:
        plt.figure(figsize=(6, 4))
        for n, pts in sorted(by_n.items()):
            pts = sorted(pts)
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"N={n}")
        plt.xlabel("batches")
        plt.ylabel("thr (batches/s)")
        plt.title("VE pin thr vs batches")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = args.paper_dir / "fig_thr_vs_batches.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print("wrote", out)

    # pin thr vs oneshot thr when both present
    pairs = []
    for r in rows:
        if r.get("backend") == "ve_pin" and r.get("vs_oneshot") and r.get("thr"):
            pairs.append((int(r["N"]), int(r["batches"]), float(r["thr"]), float(r["vs_oneshot"])))
    if pairs:
        plt.figure(figsize=(6, 4))
        xs = [f"{n}/{b}" for n, b, _, _ in pairs]
        ys = [v for *_, v in pairs]
        plt.bar(range(len(ys)), ys)
        plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
        plt.ylabel("speedup vs oneshot")
        plt.title("Resident pin / cold oneshot")
        plt.tight_layout()
        out = args.paper_dir / "fig_vs_oneshot.png"
        plt.savefig(out, dpi=120)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
