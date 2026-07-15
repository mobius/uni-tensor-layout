"""L0 smoke: terminal examples host-only + unit helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _run_example(script: str, *args: str, out: str) -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / script),
            *args,
            "--quiet",
            "--out-dir",
            str(ROOT / "artifacts" / "examples" / out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=_env(),
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert (ROOT / "artifacts" / "examples" / out / "metrics.json").is_file()


def test_stencil5_csr_structure():
    from uni_cute_tensor.apps.spmv_dataprep import csr_spmv, stencil5_csr

    csr = stencil5_csr(8, 8)
    assert csr.nrows == 64
    assert 64 <= csr.nnz <= 64 * 5
    x = np.ones(64)
    y = csr_spmv(csr, x)
    assert float(np.max(np.abs(y - csr.to_dense() @ x))) < 1e-12


def test_dataprep_clean_no_nan():
    from uni_cute_tensor.apps.dataprep_clean import clean_features, make_dirty_matrix

    rng = np.random.default_rng(0)
    dirty = make_dirty_matrix(40, 12, rng=rng, nan_frac=0.05, outlier_frac=0.05)
    assert np.isnan(dirty).any()
    cleaned, st = clean_features(dirty)
    assert not np.isnan(cleaned).any()
    assert st.nan_count > 0


def test_e1_host_only_cli():
    _run_example(
        "e1_batch_dense_regression.py",
        "--host-only",
        "--m",
        "64",
        "--k",
        "48",
        "--n",
        "32",
        "--batches",
        "2",
        out="_smoke_e1",
    )


def test_e2_host_only_cli():
    _run_example(
        "e2_sparse_then_dense.py",
        "--host-only",
        "--nx",
        "16",
        "--ny",
        "16",
        "--nrhs",
        "24",
        "--n_out",
        "32",
        out="_smoke_e2",
    )


def test_e3_host_only_cli():
    _run_example(
        "e3_dataprep_project.py",
        "--host-only",
        "--m",
        "96",
        "--k",
        "64",
        "--n-out",
        "32",
        out="_smoke_e3",
    )


def test_e4_host_only_cli():
    _run_example(
        "e4_job_dag.py",
        "--host-only",
        "--prefer-local",
        "--m",
        "80",
        "--k",
        "64",
        "--n",
        "48",
        out="_smoke_e4",
    )


def test_e5_host_only_cli():
    _run_example(
        "e5_sustained_jobs.py",
        "--host-only",
        "--jobs",
        "4",
        "--m",
        "96",
        "--k",
        "96",
        "--n",
        "96",
        "--no-cold-compare",
        out="_smoke_e5",
    )
