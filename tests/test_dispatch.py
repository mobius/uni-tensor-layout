"""Dispatch policy + calibration helpers (no device required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from uni_cute_tensor.partition.cost_model import (
    CalibrationSample,
    calibrate_from_samples,
    get_device_model,
    save_calibration,
    set_calibration,
    try_autoload_calibration,
)
from uni_cute_tensor.partition.dispatch import recommend_backend


def test_recommend_host_without_ve():
    rec = recommend_backend(256, 256, 256, batches=1, ve_devices=[], autoload_calibration=False)
    assert rec.recommended == "host"
    assert rec.ve_available is False
    assert any(e.backend == "host" for e in rec.estimates)


def test_recommend_with_ve_has_estimates():
    rec = recommend_backend(
        1024, 1024, 1024, batches=8, ve_devices=["ve1"], prefer_mode="pin", autoload_calibration=False
    )
    assert rec.recommended in ("host", "ve", "hetero")
    backends = {e.backend for e in rec.estimates}
    assert "host" in backends
    assert "ve:pin" in backends
    d = rec.to_dict()
    assert "reason" in d and d["batches"] == 8


def test_calibrate_from_synthetic_samples(tmp_path: Path):
    set_calibration({})
    samples = [
        CalibrationSample("host", 256, 256, 256, wall_sec=0.002, kernel_gflops=200.0),
        CalibrationSample("host", 512, 512, 512, wall_sec=0.008, kernel_gflops=350.0),
        CalibrationSample("ve", 256, 256, 256, wall_sec=0.05, kernel_gflops=800.0, transfer_bytes=8 * 3 * 256**2),
        CalibrationSample("ve", 512, 512, 512, wall_sec=0.08, kernel_gflops=1200.0, transfer_bytes=8 * 3 * 512**2),
    ]
    report = calibrate_from_samples(samples)
    assert "host" in report.fitted and "ve" in report.fitted
    assert get_device_model("host").peak_gflops > 0
    out = tmp_path / "cal.json"
    save_calibration(report, out)
    set_calibration({})
    assert try_autoload_calibration(out, force=True)
    assert get_device_model("ve").launch_overhead_sec >= 0
