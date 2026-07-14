"""Partition logical matrices across devices using CuTe-style shapes.

Uses contiguous row blocks by default (device-friendly for DGEMM with shared B
or column panels). Guarantees full coverage and no row overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from tensor_layouts import Layout, size


@dataclass(frozen=True)
class Shard:
    """One device-owned sub-tensor view (logical, no storage)."""

    device: str
    # Inclusive-exclusive row range in the global matrix
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    # Layout of the shard in its own (m_local, n_local) domain, column-major default
    layout: Layout
    atom_name: str = ""

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def cols(self) -> int:
        return self.col_end - self.col_start

    @property
    def nelems(self) -> int:
        return size(self.layout)


@dataclass
class PlacementPlan:
    """Full placement of a logical matrix across devices."""

    global_shape: tuple[int, int]
    shards: list[Shard] = field(default_factory=list)
    strategy: str = "row_blocks"

    def validate(self) -> None:
        m, n = self.global_shape
        if m < 0 or n < 0:
            raise ValueError("shape must be non-negative")
        covered = [0] * m
        for s in self.shards:
            if s.rows <= 0 or s.cols <= 0:
                raise ValueError(f"empty shard on {s.device}")
            if not (0 <= s.row_start < s.row_end <= m):
                raise ValueError(f"row range OOB on {s.device}")
            if not (0 <= s.col_start < s.col_end <= n):
                raise ValueError(f"col range OOB on {s.device}")
            if s.cols != n - 0 and not (s.col_start == 0 and s.col_end == n):
                # allow partial cols but still check
                pass
            for r in range(s.row_start, s.row_end):
                # only enforce non-overlap when full-width row shards
                if s.col_start == 0 and s.col_end == n:
                    covered[r] += 1
            if size(s.layout) != s.rows * s.cols:
                raise ValueError(f"layout size mismatch on {s.device}")
        # Full-width row partition must cover each row exactly once
        if self.strategy == "row_blocks":
            if any(c != 1 for c in covered):
                bad = [i for i, c in enumerate(covered) if c != 1]
                raise ValueError(f"row coverage invalid at rows {bad[:8]}...")

    def to_dict(self) -> dict:
        return {
            "global_shape": list(self.global_shape),
            "strategy": self.strategy,
            "shards": [
                {
                    "device": s.device,
                    "row_start": s.row_start,
                    "row_end": s.row_end,
                    "col_start": s.col_start,
                    "col_end": s.col_end,
                    "layout": str(s.layout),
                    "atom_name": s.atom_name,
                    "nelems": s.nelems,
                }
                for s in self.shards
            ],
        }


def _split_extent(extent: int, n_parts: int) -> list[tuple[int, int]]:
    """Split [0, extent) into n_parts nearly equal contiguous ranges."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    if extent < 0:
        raise ValueError("extent must be non-negative")
    if n_parts > extent and extent > 0:
        # more devices than rows: first `extent` devices get 1 row, rest empty skipped later
        n_parts = extent
    if extent == 0:
        return []
    base, rem = divmod(extent, n_parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n_parts):
        span = base + (1 if i < rem else 0)
        end = start + span
        if span > 0:
            ranges.append((start, end))
        start = end
    return ranges


def partition_matrix_rows(
    m: int,
    n: int,
    devices: Sequence[str],
    *,
    atom_name: str = "VE_NLC_DGEMM_64x64x64_F64",
    column_major: bool = True,
) -> PlacementPlan:
    """Row-block partition of an M×N matrix across devices.

    Each shard layout is local (rows_i, n) with column-major strides by default:
    Layout((rows_i, n), (1, rows_i)).
    """
    if not devices:
        raise ValueError("devices must be non-empty")
    ranges = _split_extent(m, len(devices))
    shards: list[Shard] = []
    for dev, (rs, re) in zip(devices, ranges):
        rows = re - rs
        if column_major:
            layout = Layout((rows, n), (1, rows))
        else:
            layout = Layout((rows, n), (n, 1))
        shards.append(
            Shard(
                device=dev,
                row_start=rs,
                row_end=re,
                col_start=0,
                col_end=n,
                layout=layout,
                atom_name=atom_name,
            )
        )
    plan = PlacementPlan(global_shape=(m, n), shards=shards, strategy="row_blocks")
    plan.validate()
    return plan


def partition_to_devices(
    m: int,
    n: int,
    devices: Sequence[str],
    *,
    prefer_kind: str = "ve",
) -> PlacementPlan:
    """Convenience: pick atom by prefer_kind and row-partition."""
    atom = {
        "ve": "VE_NLC_DGEMM_64x64x64_F64",
        "phi": "PHI_KNC_8x8x8_F64",
        "host": "HOST_AVX512_8x8x8_F64",
    }.get(prefer_kind, "VE_NLC_DGEMM_64x64x64_F64")
    return partition_matrix_rows(m, n, devices, atom_name=atom)
