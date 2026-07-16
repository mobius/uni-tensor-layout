#!/usr/bin/env python3
"""Summarize Timeline JSONL as phase totals + ASCII bar chart (Phase 3 M2 / W4.2)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def report(events: list[dict], *, width: int = 40) -> str:
    by_phase: dict[str, float] = defaultdict(float)
    by_name: dict[str, float] = defaultdict(float)
    for e in events:
        d = float(e.get("duration_sec", 0.0) or 0.0)
        by_phase[str(e.get("phase", "other"))] += d
        by_name[str(e.get("name", "?"))] += d
    total = sum(by_phase.values()) or 1.0
    lines = ["# Timeline report", "", f"events={len(events)}  total_span_sum={total:.6f}s", ""]
    lines.append("## By phase")
    lines.append("")
    for phase, sec in sorted(by_phase.items(), key=lambda x: -x[1]):
        frac = sec / total
        bar = "#" * max(1, int(round(frac * width)))
        lines.append(f"  {phase:<12} {sec:10.6f}s  {100*frac:5.1f}%  {bar}")
    lines.append("")
    lines.append("## By name")
    lines.append("")
    for name, sec in sorted(by_name.items(), key=lambda x: -x[1])[:20]:
        frac = sec / total
        bar = "#" * max(1, int(round(frac * width)))
        lines.append(f"  {name:<20} {sec:10.6f}s  {100*frac:5.1f}%  {bar}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Timeline JSONL phase report")
    ap.add_argument("jsonl", type=Path, help="path to timeline JSONL")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=40)
    args = ap.parse_args()
    if not args.jsonl.is_file():
        print(f"not found: {args.jsonl}", file=sys.stderr)
        return 2
    text = report(load_events(args.jsonl), width=args.width)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
