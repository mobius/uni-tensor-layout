"""Job runner smoke (host-only)."""

from __future__ import annotations

import json
from pathlib import Path

from uni_cute_tensor.runtime.job_runner import load_job, run_job

ROOT = Path(__file__).resolve().parents[1]


def test_load_and_run_dense_batch_host(tmp_path: Path):
    job = {
        "type": "dense_batch",
        "m": 64,
        "n": 48,
        "k": 40,
        "batches": 3,
        "backend": "host",
        "host_only": True,
    }
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"
    assert r.job_type == "dense_batch"
    assert (tmp_path / "metrics.json").is_file()
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["status"] == "pass"


def test_run_bundled_dataprep_host(tmp_path: Path):
    path = ROOT / "jobs" / "dataprep.json"
    assert path.is_file()
    job = load_job(path)
    job["m"] = 80
    job["k"] = 48
    job["n_out"] = 24
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"


def test_run_sparse_dense_host(tmp_path: Path):
    job = {"type": "sparse_dense", "nx": 12, "ny": 12, "nrhs": 16, "n_out": 20}
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"
    assert r.metrics["nnz"] > 0


def test_phi_flag_falls_back_host_only(tmp_path: Path):
    job = {
        "type": "dense_batch",
        "m": 32,
        "n": 32,
        "k": 32,
        "batches": 2,
        "phi": True,
        "compare_oneshot": False,
    }
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"
    assert r.metrics.get("phi") is False or "host" in str(r.metrics.get("prep_note", ""))
    # host_only forces phi off
    assert r.metrics.get("phi") is False


def test_phi_prep_ve_gemm_type_host(tmp_path: Path):
    job = {
        "type": "phi_prep_ve_gemm",
        "m": 32,
        "n": 32,
        "k": 32,
        "batches": 2,
        "phi": True,
        "compare_oneshot": False,
    }
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"
    assert r.job_type == "phi_prep_ve_gemm"


def test_service_dense_stream_alias(tmp_path: Path):
    job = {
        "type": "service_dense_stream",
        "m": 32,
        "n": 24,
        "k": 24,
        "batches": 2,
        "backend": "host",
        "compare_oneshot": False,
    }
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"


def test_load_array_npy_and_npz(tmp_path: Path):
    import numpy as np
    from uni_cute_tensor.runtime.job_runner import load_array

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    npy = tmp_path / "a.npy"
    npz = tmp_path / "a.npz"
    np.save(npy, a)
    np.savez(npz, arr=a)
    got = load_array(npy)
    assert got.shape == (3, 4)
    assert float(np.max(np.abs(got - a))) == 0.0
    got2 = load_array(npz)
    assert got2.shape == (3, 4)


def test_load_csr_npz(tmp_path: Path):
    import numpy as np
    from uni_cute_tensor.apps.spmv_dataprep import stencil5_csr
    from uni_cute_tensor.runtime.job_runner import load_csr_npz

    csr = stencil5_csr(4, 4)
    p = tmp_path / "c.npz"
    np.savez(
        p,
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data,
        nrows=csr.nrows,
        ncols=csr.ncols,
    )
    loaded = load_csr_npz(p)
    assert loaded.nrows == 16
    assert loaded.nnz == csr.nnz


def test_external_matrix_dense_host(tmp_path: Path):
    import numpy as np

    a = np.eye(16, dtype=np.float64)
    b = np.ones((16, 8), dtype=np.float64)
    ap = tmp_path / "A.npy"
    bp = tmp_path / "B.npy"
    np.save(ap, a)
    np.save(bp, b)
    job = {
        "type": "dense_batch",
        "matrix_a": str(ap),
        "matrix_b": str(bp),
        "batches": 2,
        "backend": "host",
        "compare_oneshot": False,
        "phi": False,
    }
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"
    assert r.metrics.get("external_a") is True
    assert r.metrics.get("external_b") is True
    assert r.metrics["m"] == 16
    assert r.metrics["n"] == 8


def test_external_csr_sparse_dense_host(tmp_path: Path):
    import numpy as np
    from uni_cute_tensor.apps.spmv_dataprep import stencil5_csr

    csr = stencil5_csr(6, 6)
    cp = tmp_path / "csr.npz"
    np.savez(
        cp,
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data,
        nrows=csr.nrows,
        ncols=csr.ncols,
    )
    x = np.ones((csr.ncols, 4), dtype=np.float64)
    w = np.eye(4, 8, dtype=np.float64)
    xp = tmp_path / "x.npy"
    wp = tmp_path / "w.npy"
    np.save(xp, x)
    np.save(wp, w)
    job = {
        "type": "sparse_dense",
        "csr_path": str(cp),
        "matrix_x": str(xp),
        "matrix_w": str(wp),
        "backend": "host",
        "phi": False,
    }
    r = run_job(job, host_only=True, out_dir=tmp_path)
    assert r.status == "pass"
    assert r.metrics.get("csr_external") is True
    assert r.metrics["nnz"] == csr.nnz
