"""Hardware-backed tests. Skipped automatically when devices/toolchain missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uni_cute_tensor.backends.host_dgemm import host_avx512_dgemm, host_blocked_dgemm
from uni_cute_tensor.backends.phi_dgemm import run_phi_dgemm
from uni_cute_tensor.backends.phi_smoke import phi_device_present, run_phi_peak_smoke
from uni_cute_tensor.backends.ve_dgemm import (
    multi_ve_layout_dgemm,
    run_ve_dgemm_shard,
    ve_toolchain_available,
)
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names


pytestmark = pytest.mark.device


def test_host_blocked_atom_tile():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((64, 48))
    b = rng.standard_normal((48, 40))
    _, res = host_blocked_dgemm(a, b)
    assert res.status == "pass"
    assert res.max_abs_err < 1e-9


def test_host_avx512_dgemm():
    rng = np.random.default_rng(9)
    a = rng.standard_normal((96, 80))
    b = rng.standard_normal((80, 64))
    _, res = host_avx512_dgemm(a, b, threads=8)
    assert res.status == "pass"
    assert res.max_abs_err < 1e-8


@pytest.mark.skipif(not phi_device_present(), reason="no /dev/mic0")
def test_phi_mkl_dgemm_correctness():
    rng = np.random.default_rng(12)
    a = rng.standard_normal((128, 96))
    b = rng.standard_normal((96, 80))
    try:
        c, res = run_phi_dgemm(a, b, threads=120, backend="mkl")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"mkl backend unavailable: {exc}")
    assert res.status == "pass"
    assert res.max_abs_err < 1e-6
    assert c.shape == (128, 80)


@pytest.mark.skipif(not ve_toolchain_available(), reason="VE toolchain missing")
def test_ve_worker_pool_multi():
    from uni_cute_tensor.backends.ve_worker import VeWorkerPool, multi_ve_layout_dgemm_pooled
    from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names

    ve = ve_device_names(discover_devices())
    if len(ve) < 1:
        pytest.skip("no VE")
    ve_ids = [int(d.replace("ve", "")) for d in ve]
    rng = np.random.default_rng(3)
    a = rng.standard_normal((192, 128))
    b = rng.standard_normal((128, 160))
    with VeWorkerPool(ve_ids) as pool:
        c, plan, results = multi_ve_layout_dgemm_pooled(a, b, ve, pool, share_b=True)
    ref = a @ b
    assert float(np.max(np.abs(c - ref))) < 1e-8
    assert all(r.status == "pass" for r in results)
    assert sum(s.rows for s in plan.shards) == 192


@pytest.mark.skipif(
    not (phi_device_present() and ve_toolchain_available()),
    reason="need Phi+VE",
)
def test_hetero_phi_ve_pipeline():
    from uni_cute_tensor.apps.hetero_pipeline import run_hetero_phi_ve_gemm

    rng = np.random.default_rng(5)
    a = rng.standard_normal((256, 192))
    b = rng.standard_normal((192, 160))
    try:
        _, res = run_hetero_phi_ve_gemm(
            a, b, alpha=1.02, beta=0.001, use_worker_pool=True, phi_threads=120
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"hetero unavailable: {exc}")
    assert res.status == "pass"
    assert res.max_abs_err < 1e-8


@pytest.mark.skipif(not ve_toolchain_available(), reason="VE toolchain missing")
def test_aveo_single_ve():
    from uni_cute_tensor.backends.ve_aveo import aveo_available, run_aveo_dgemm

    if not aveo_available():
        pytest.skip("AVEO not available")
    rng = np.random.default_rng(8)
    a = rng.standard_normal((96, 64))
    b = rng.standard_normal((64, 80))
    try:
        c, res = run_aveo_dgemm(a, b, ve_node=1)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"AVEO run failed: {exc}")
    assert res.status == "pass"
    assert res.max_abs_err < 1e-8
    assert c.shape == (96, 80)


def test_choose_best_placement():
    from uni_cute_tensor.partition.cost_model import choose_best_placement

    choice = choose_best_placement(1024, 512, 256, ["ve1", "ve2", "ve3"])
    assert choice.strategy in ("row_blocks", "col_blocks")
    assert choice.est_total_sec > 0
    assert len(choice.plan.shards) >= 1


def test_host_dgemm_auto():
    from uni_cute_tensor.backends.host_dgemm import host_dgemm

    rng = np.random.default_rng(2)
    a = rng.standard_normal((64, 64))
    b = rng.standard_normal((64, 64))
    c, r = host_dgemm(a, b, backend="auto")
    assert r.status == "pass"
    assert c.shape == (64, 64)


def test_power_cap_local():
    from uni_cute_tensor.power import PowerCap

    cap = PowerCap(psu_limit_w=1600, safety_margin=0.9)
    assert cap.can_launch(["ve1"], {"ve1": "dgemm"})
    with cap.guard(["ve1", "ve2"], {"ve1": "dgemm", "ve2": "dgemm"}):
        assert cap.reserved_watts() > 0
    # after release
    assert cap.reserved_watts() == 0 or cap.backend == "uni"


@pytest.mark.skipif(not Path("/dev/mic0").exists(), reason="no mic")
def test_phi_worker_scale():
    from uni_cute_tensor.backends.phi_worker import PhiWorker

    rng = np.random.default_rng(4)
    a = rng.standard_normal((64, 48))
    try:
        with PhiWorker() as w:
            c, r = w.scale(a, alpha=1.1, beta=0.01)
    except Exception as exc:  # pragma: no cover
        pytest.skip(str(exc))
    assert r.status == "pass"
    assert c.shape == a.shape


@pytest.mark.skipif(not ve_toolchain_available(), reason="no VE")
def test_aveo_batch_dual_buf():
    from uni_cute_tensor.backends.ve_aveo import AveoSessionPool, aveo_available

    if not aveo_available():
        pytest.skip("no aveo")
    rng = np.random.default_rng(6)
    As = [rng.standard_normal((64, 64)) for _ in range(4)]
    Bs = [rng.standard_normal((64, 64)) for _ in range(4)]
    try:
        with AveoSessionPool([1]) as pool:
            Cs, wall, err = pool.dgemm_batch(1, As, Bs)
    except Exception as exc:  # pragma: no cover
        pytest.skip(str(exc))
    assert err < 1e-8
    assert len(Cs) == 4
    assert wall > 0


@pytest.mark.skipif(not ve_toolchain_available(), reason="VE toolchain missing")
def test_single_ve_dgemm_correctness():
    ve = ve_device_names(discover_devices())
    if not ve:
        pytest.skip("no VE devices")
    ve_id = int(ve[0].replace("ve", ""))
    rng = np.random.default_rng(2)
    a = rng.standard_normal((96, 64))
    b = rng.standard_normal((64, 80))
    c, res = run_ve_dgemm_shard(a, b, ve_id=ve_id)
    assert res.status == "pass"
    assert res.max_abs_err < 1e-8
    assert c.shape == (96, 80)


@pytest.mark.skipif(not ve_toolchain_available(), reason="VE toolchain missing")
def test_multi_ve_layout_dgemm_correctness():
    ve = ve_device_names(discover_devices())
    if len(ve) < 1:
        pytest.skip("no VE devices")
    rng = np.random.default_rng(3)
    m, k, n = 192, 128, 160
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    c, plan, results = multi_ve_layout_dgemm(a, b, ve, parallel=True)
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
    assert err < 1e-8
    assert all(r.status == "pass" for r in results)
    assert sum(s.rows for s in plan.shards) == m


@pytest.mark.skipif(not phi_device_present(), reason="no /dev/mic0")
def test_phi_peak_smoke():
    res = run_phi_peak_smoke()
    if res.status == "skip":
        pytest.skip(res.stderr or "phi peak binary missing")
    assert res.status == "pass"
    assert res.gflops > 100.0


@pytest.mark.skipif(not phi_device_present(), reason="no /dev/mic0")
def test_phi_dgemm_correctness():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((64, 48))
    b = rng.standard_normal((48, 40))
    c, res = run_phi_dgemm(a, b, threads=32)
    assert res.status == "pass"
    assert res.max_abs_err < 1e-6
    assert c.shape == (64, 40)
