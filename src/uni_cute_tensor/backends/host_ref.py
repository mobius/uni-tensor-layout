"""Host reference GEMM using numpy — correctness oracle for layout plans."""

from __future__ import annotations

import numpy as np

from uni_cute_tensor.partition.multi_device import PlacementPlan


def host_dgemm_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C = A @ B for 2-D float64 arrays."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dim mismatch {a.shape} @ {b.shape}")
    return a @ b


def host_sharded_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    plan: PlacementPlan,
) -> np.ndarray:
    """Execute GEMM on host following PlacementPlan (row/col/k strategies)."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    if plan.global_shape != (m, n):
        if plan.global_shape[0] != m or plan.global_shape[1] != n:
            raise ValueError(
                f"plan shape {plan.global_shape} != C shape {(m, n)}"
            )
    c = np.zeros((m, n), dtype=np.result_type(a, b))
    if plan.strategy == "col_blocks":
        for shard in plan.shards:
            cs, ce = shard.col_start, shard.col_end
            c[:, cs:ce] = a @ b[:, cs:ce]
        return c
    if plan.strategy == "k_split":
        for shard in plan.shards:
            ke = shard.k_end if shard.k_end > 0 else (plan.k or k)
            c += a[:, shard.k_start : ke] @ b[shard.k_start : ke, :]
        return c
    # default row_blocks
    for shard in plan.shards:
        rs, re = shard.row_start, shard.row_end
        c[rs:re, :] = a[rs:re, :] @ b
    return c
