"""Partition coverage tests."""

import numpy as np
import pytest
from tensor_layouts import size

from cpu_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm
from cpu_cute_tensor.partition.multi_device import partition_matrix_rows, partition_to_devices
from cpu_cute_tensor.partition.pcie_cost import estimate_h2d_seconds


def test_row_partition_coverage():
    plan = partition_matrix_rows(100, 64, ["ve1", "ve2", "ve3"])
    assert plan.global_shape == (100, 64)
    assert len(plan.shards) == 3
    rows = sum(s.rows for s in plan.shards)
    assert rows == 100
    # nearly equal
    assert max(s.rows for s in plan.shards) - min(s.rows for s in plan.shards) <= 1
    plan.validate()


def test_layout_size_matches_shard():
    plan = partition_matrix_rows(48, 32, ["a", "b"])
    for s in plan.shards:
        assert size(s.layout) == s.rows * s.cols


def test_empty_devices_raises():
    with pytest.raises(ValueError):
        partition_matrix_rows(10, 10, [])


def test_sharded_dgemm_matches_reference():
    rng = np.random.default_rng(0)
    m, k, n = 90, 40, 50
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    plan = partition_to_devices(m, n, ["ve1", "ve2", "ve3"])
    c_ref = host_dgemm_reference(a, b)
    c_sh = host_sharded_dgemm(a, b, plan)
    assert np.allclose(c_ref, c_sh)


def test_pcie_estimate_positive():
    t = estimate_h2d_seconds(12 * (1024**3))  # 12 GiB at 12 GB/s ~ 1s
    assert 0.5 < t < 2.0
