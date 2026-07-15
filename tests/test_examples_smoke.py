"""L0 smoke: terminal examples host-only + stencil CSR unit checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stencil5_csr_structure():
    from uni_cute_tensor.apps.spmv_dataprep import csr_spmv, stencil5_csr

    csr = stencil5_csr(8, 8)
    assert csr.nrows == 64
    assert csr.ncols == 64
    # 5-point: interior 5 nnz, edges fewer — total nnz between 64 and 64*5
    assert 64 <= csr.nnz <= 64 * 5
    x = np.ones(64)
    y = csr_spmv(csr, x)
    # constant vector: Laplacian-like rows sum to 0 at interior if diag=4, off=-1
    # boundary not zero; just check finite
    assert y.shape == (64,)
    assert np.all(np.isfinite(y))
    dense = csr.to_dense()
    assert float(np.max(np.abs(y - dense @ x))) < 1e-12


def test_e1_host_only_cli():
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "e1_batch_dense_regression.py"),
            "--host-only",
            "--m",
            "64",
            "--k",
            "48",
            "--n",
            "32",
            "--batches",
            "2",
            "--quiet",
            "--out-dir",
            str(ROOT / "artifacts" / "examples" / "_smoke_e1"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    metrics = ROOT / "artifacts" / "examples" / "_smoke_e1" / "metrics.json"
    assert metrics.is_file()


def test_e2_host_only_cli():
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "e2_sparse_then_dense.py"),
            "--host-only",
            "--nx",
            "16",
            "--ny",
            "16",
            "--nrhs",
            "24",
            "--n_out",
            "32",
            "--quiet",
            "--out-dir",
            str(ROOT / "artifacts" / "examples" / "_smoke_e2"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ROOT / "artifacts" / "examples" / "_smoke_e2" / "metrics.json").is_file()
