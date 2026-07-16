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
