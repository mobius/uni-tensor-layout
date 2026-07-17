#!/usr/bin/env python3
"""Write small external .npy / CSR .npz samples for job matrix_a / csr_path demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "sample_data")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((args.n, args.n))
    b = rng.standard_normal((args.n, args.n))
    np.save(args.out / "A.npy", a)
    np.save(args.out / "B.npy", b)

    # tiny CSR (stencil) + matching X/W for sparse_dense jobs
    from uni_cute_tensor.apps.spmv_dataprep import stencil5_csr

    csr = stencil5_csr(8, 8)
    np.savez(
        args.out / "csr_stencil8.npz",
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data,
        nrows=csr.nrows,
        ncols=csr.ncols,
    )
    nrhs, n_out = 8, 16
    x = rng.standard_normal((csr.ncols, nrhs))
    w = rng.standard_normal((nrhs, n_out))
    np.save(args.out / "X.npy", x)
    np.save(args.out / "W.npy", w)
    print("wrote", args.out)
    print("  A.npy B.npy csr_stencil8.npz X.npy W.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
