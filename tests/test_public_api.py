"""Public API surface checks for v1.0 freeze (no device required)."""

from __future__ import annotations

import uni_cute_tensor as uct


def test_version_semver_major():
    parts = uct.__version__.split(".")
    assert int(parts[0]) >= 1
    # 1.1.x after terminal examples closeout
    assert (int(parts[0]), int(parts[1])) >= (1, 0)


def test_public_all_importable():
    for name in uct.__all__:
        assert hasattr(uct, name), name
        getattr(uct, name)


def test_host_dgemm_and_dataplane():
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 24))
    b = rng.standard_normal((24, 16))
    c, r = uct.host_dgemm(a, b, backend="numpy")
    assert r.status == "pass"
    assert c.shape == (32, 16)

    with uct.create_dataplane("host") as plane:
        from uni_cute_tensor import GemmRequest

        out = plane.gemm(GemmRequest(a=a, b=b))
    assert out.status == "pass"


def test_placement_json_and_choose():
    plan = uct.partition_matrix_rows(64, 48, ["ve1", "ve2"], k=32)
    text = plan.to_json()
    loaded = uct.PlacementPlan.from_json(text)
    assert loaded.global_shape == (64, 48)
    ch = uct.choose_best_placement(64, 48, 32, ["ve1", "ve2", "ve3"])
    assert ch.plan.strategy in ("row_blocks", "col_blocks", "k_split")
    assert ch.est_total_sec > 0


def test_powercap_local():
    cap = uct.PowerCap(psu_limit_w=1600.0)
    cap._uni = None
    assert cap.can_launch(["ve1"], {"ve1": "dgemm"})
