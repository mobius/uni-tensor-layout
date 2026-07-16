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
