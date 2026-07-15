"""Phase 2 M2: PlacementPlan JSON, strategies, cost model, host runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uni_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm
from uni_cute_tensor.partition.cost_model import (
    CalibrationSample,
    calibrate_from_samples,
    choose_best_placement,
    estimate_gemm_placement,
    prediction_error_report,
)
from uni_cute_tensor.partition.multi_device import (
    PlacementPlan,
    partition_matrix_cols,
    partition_matrix_k,
    partition_matrix_rows,
)
from uni_cute_tensor.partition.runner import execute_plan
from uni_cute_tensor.power import PowerCap


def test_row_col_k_plans_correct_host():
    rng = np.random.default_rng(1)
    m, k, n = 90, 64, 80
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    ref = host_dgemm_reference(a, b)
    devices = ["ve1", "ve2", "ve3"]

    for plan in (
        partition_matrix_rows(m, n, devices, k=k),
        partition_matrix_cols(m, n, devices, k=k),
        partition_matrix_k(m, n, k, devices),
    ):
        plan.validate()
        c = host_sharded_dgemm(a, b, plan)
        assert np.allclose(c, ref), plan.strategy


def test_placement_json_roundtrip(tmp_path: Path):
    plan = partition_matrix_rows(48, 32, ["ve1", "ve2"], k=16, backend="VE_NLC")
    plan.estimated_total_sec = 0.12
    plan.estimated_transfer_sec = 0.04
    plan.estimated_compute_sec = 0.08
    plan.transfer_bytes = 12345
    plan.meta["note"] = "m2"
    path = tmp_path / "plan.json"
    plan.write_json(path)
    loaded = PlacementPlan.load_json(path)
    assert loaded.global_shape == (48, 32)
    assert loaded.k == 16
    assert loaded.backend == "VE_NLC"
    assert len(loaded.shards) == 2
    assert loaded.shards[0].dtype == "float64"
    assert loaded.shards[0].strides[0] == 1
    assert loaded.estimated_total_sec == pytest.approx(0.12)
    assert loaded.meta["note"] == "m2"
    # re-serialize stable keys
    d = json.loads(loaded.to_json())
    assert "shards" in d and "estimated_total_sec" in d


def test_choose_best_under_powercap():
    # tight budget: only 1 VE ~280W fits under 300W effective
    cap = PowerCap(psu_limit_w=320.0, safety_margin=0.9)  # ~288W
    # force local backend by clearing uni if reservation weird — local path uses estimates
    cap._uni = None
    choice = choose_best_placement(
        512, 512, 512, ["ve1", "ve2", "ve3"], power_cap=cap, backend="VE_NLC"
    )
    assert len(choice.devices) >= 1
    assert choice.plan.strategy in ("row_blocks", "col_blocks", "k_split")
    assert choice.est_total_sec > 0
    assert choice.plan.estimated_total_sec > 0


def test_calibrate_and_predict():
    samples = [
        CalibrationSample("ve", 512, 512, 512, wall_sec=0.02, kernel_gflops=1200.0),
        CalibrationSample("ve", 1024, 1024, 1024, wall_sec=0.05, kernel_gflops=1600.0),
        CalibrationSample("host", 512, 512, 512, wall_sec=0.01, kernel_gflops=200.0),
    ]
    report = calibrate_from_samples(samples)
    assert "ve" in report.fitted
    assert report.fitted["ve"]["peak_gflops"] > 0
    ch = estimate_gemm_placement(512, 512, 512, ["ve1"], strategy="row_blocks")
    err = prediction_error_report(0.02, ch)
    assert "rel_error" in err


def test_execute_plan_host_strategies():
    rng = np.random.default_rng(2)
    m, k, n = 60, 48, 40
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    for plan in (
        partition_matrix_rows(m, n, ["h0", "h1"], k=k, backend="HOST_REF"),
        partition_matrix_cols(m, n, ["h0", "h1"], k=k, backend="HOST_REF"),
        partition_matrix_k(m, n, k, ["h0", "h1"], backend="HOST_REF"),
    ):
        res = execute_plan(a, b, plan, force_host=True)
        assert res.status == "pass", plan.strategy
        assert res.max_abs_err < 1e-10


def test_spmv_pipeline_host_only():
    from uni_cute_tensor.apps.spmv_dataprep import (
        random_csr,
        run_spmv_dataprep_pipeline,
    )

    rng = np.random.default_rng(3)
    csr = random_csr(64, 48, density=0.1, rng=rng)
    x = rng.standard_normal((48, 32))
    b = rng.standard_normal((32, 24))
    res = run_spmv_dataprep_pipeline(
        csr, x, b, devices=[], force_host_gemm=True, compare_host=False, use_phi_prep=False
    )
    assert res.status == "pass"
    assert res.max_abs_err < 1e-10
