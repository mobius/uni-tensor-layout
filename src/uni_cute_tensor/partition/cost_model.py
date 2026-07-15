"""Simple placement cost model: transfer / PCIe + flops / device peak."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from uni_cute_tensor.partition.multi_device import PlacementPlan, partition_matrix_rows
from uni_cute_tensor.partition.pcie_cost import DEFAULT_PCIE_GBPS, estimate_h2d_seconds

Strategy = Literal["row_blocks", "col_blocks"]


@dataclass
class DeviceModel:
    name: str
    kind: str  # ve | phi | host
    peak_gflops: float
    online: bool = True


# Rough peaks from this machine's measured numbers
DEFAULT_MODELS = {
    "ve": DeviceModel("ve", "ve", peak_gflops=1750.0),
    "phi": DeviceModel("phi", "phi", peak_gflops=700.0),
    "host": DeviceModel("host", "host", peak_gflops=500.0),
}


@dataclass
class PlacementChoice:
    strategy: Strategy
    plan: PlacementPlan
    est_transfer_sec: float
    est_compute_sec: float
    est_total_sec: float
    transfer_bytes: int


def _col_partition(m: int, n: int, devices: Sequence[str], atom_name: str) -> PlacementPlan:
    """Column-block partition of C (and B); each device owns a column panel of C.

    Implemented by partitioning the N dimension as if it were rows of N×M view,
    then remapping shard fields.
    """
    # Reuse row splitter on n
    from uni_cute_tensor.partition.multi_device import _split_extent
    from tensor_layouts import Layout

    ranges = _split_extent(n, len(devices))
    shards = []
    for dev, (cs, ce) in zip(devices, ranges):
        cols = ce - cs
        layout = Layout((m, cols), (cols, 1))  # row-major local
        from uni_cute_tensor.partition.multi_device import Shard

        shards.append(
            Shard(
                device=dev,
                row_start=0,
                row_end=m,
                col_start=cs,
                col_end=ce,
                layout=layout,
                atom_name=atom_name,
            )
        )
    plan = PlacementPlan(global_shape=(m, n), shards=shards, strategy="col_blocks")
    # validate only for row_blocks enforces full coverage; skip strict validate
    return plan


def estimate_gemm_placement(
    m: int,
    n: int,
    k: int,
    devices: Sequence[str],
    *,
    strategy: Strategy = "row_blocks",
    dtype_bytes: int = 8,
    pcie_gbps: float = DEFAULT_PCIE_GBPS,
    peak_gflops_per_device: float = 1750.0,
) -> PlacementChoice:
    """Estimate cost for a placement strategy (GEMM C=A@B)."""
    atom = "VE_NLC_DGEMM_64x64x64_F64"
    if strategy == "row_blocks":
        plan = partition_matrix_rows(m, n, devices, atom_name=atom)
        # Each device gets A_rows×K and full B, produces C_rows×N
        # Transfer: sum over shards of A_shard + B (shared once if share_b)
        a_bytes = m * k * dtype_bytes
        b_bytes = k * n * dtype_bytes
        c_bytes = m * n * dtype_bytes
        # model share_b: B once + all A + all C back
        xfer = a_bytes + b_bytes + c_bytes
        # compute: full flops / (n_dev * peak) with imperfect balance
        flops = 2.0 * m * n * k
        n_dev = max(len(devices), 1)
        compute = flops / (n_dev * peak_gflops_per_device * 1e9)
    else:
        plan = _col_partition(m, n, list(devices), atom)
        # col split: full A once, B panels, C panels
        a_bytes = m * k * dtype_bytes
        b_bytes = k * n * dtype_bytes
        c_bytes = m * n * dtype_bytes
        xfer = a_bytes + b_bytes + c_bytes
        flops = 2.0 * m * n * k
        n_dev = max(len(devices), 1)
        compute = flops / (n_dev * peak_gflops_per_device * 1e9)

    transfer = estimate_h2d_seconds(xfer, pcie_gbps)
    # crude: transfer and compute partially overlap for large jobs → max + 0.3*min
    total = max(transfer, compute) + 0.3 * min(transfer, compute)
    return PlacementChoice(
        strategy=strategy,
        plan=plan,
        est_transfer_sec=transfer,
        est_compute_sec=compute,
        est_total_sec=total,
        transfer_bytes=xfer,
    )


def choose_best_placement(
    m: int,
    n: int,
    k: int,
    devices: Sequence[str],
    **kwargs,
) -> PlacementChoice:
    """Pick row vs col blocks by estimated total time."""
    candidates = [
        estimate_gemm_placement(m, n, k, devices, strategy="row_blocks", **kwargs),
        estimate_gemm_placement(m, n, k, devices, strategy="col_blocks", **kwargs),
    ]
    return min(candidates, key=lambda c: c.est_total_sec)
