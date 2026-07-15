#!/usr/bin/env python3
"""Compare hetero multi-batch serial vs overlapped Phi||VE."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.apps.hetero_pipeline import run_hetero_multibatch


def main() -> int:
    rng = np.random.default_rng(0)
    n_batch = 4
    m, k, n = 512, 384, 384
    As = [rng.standard_normal((m, k)) for _ in range(n_batch)]
    Bs = [rng.standard_normal((k, n)) for _ in range(n_batch)]

    for overlap in (False, True):
        res = run_hetero_multibatch(
            As, Bs, alpha=1.02, beta=0.0, overlap=overlap, use_aveo=False
        )
        print(
            f"overlap={overlap} status={res.status} wall={res.wall_sec:.3f}s "
            f"batch/s={res.throughput_batches_per_sec:.3f} err={res.max_abs_err:.3e}"
        )
        print(" ", res.notes)
        if res.status != "pass":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
