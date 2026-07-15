"""Partition logical matrices across devices using CuTe-style shapes.

Strategies:
  - row_blocks: contiguous row panels of C (shared B)
  - col_blocks: contiguous column panels of C (shared A)
  - k_split:   split K; each device partial product, host reduce

PlacementPlan is JSON-serializable (W2.1) with dtype, strides, backend, costs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

from tensor_layouts import Layout, size

StrategyName = Literal["row_blocks", "col_blocks", "k_split"]
BackendName = Literal["VE_NLC", "PHI_MKL", "HOST_OBLAS", "HOST_AVX512", "HOST_REF"]

# Atom name ↔ backend binding (W2.4)
BACKEND_ATOMS: dict[str, str] = {
    "VE_NLC": "VE_NLC_DGEMM_64x64x64_F64",
    "PHI_MKL": "PHI_KNC_8x8x8_F64",
    "HOST_OBLAS": "HOST_AVX512_8x8x8_F64",
    "HOST_AVX512": "HOST_AVX512_8x8x8_F64",
    "HOST_REF": "HOST_AVX512_8x8x8_F64",
}

DTYPE_BYTES = {"float64": 8, "float32": 4, "f64": 8, "f32": 4}


@dataclass(frozen=True)
class Shard:
    """One device-owned sub-tensor view (logical, no storage)."""

    device: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    layout: Layout
    atom_name: str = ""
    # W2.1 extensions
    backend: str = "VE_NLC"
    dtype: str = "float64"
    strides: tuple[int, int] = (1, 0)  # local (row_stride, col_stride); 0 → fill later
    k_start: int = 0
    k_end: int = 0  # 0 means "full K" (set by plan.k)
    affinity: str = ""  # optional device affinity hint

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def cols(self) -> int:
        return self.col_end - self.col_start

    @property
    def nelems(self) -> int:
        return size(self.layout)

    def resolved_strides(self) -> tuple[int, int]:
        if self.strides[1] != 0 or self.strides[0] != 0:
            # if col stride is 0 but row is set, treat as incomplete
            if self.strides[0] != 0 and self.strides[1] != 0:
                return self.strides
        # column-major default: (1, rows)
        return (1, self.rows)

    def to_dict(self) -> dict[str, Any]:
        rs, cs = self.resolved_strides()
        return {
            "device": self.device,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "k_start": self.k_start,
            "k_end": self.k_end,
            "layout": str(self.layout),
            "atom_name": self.atom_name,
            "backend": self.backend,
            "dtype": self.dtype,
            "strides": [rs, cs],
            "affinity": self.affinity or self.device,
            "nelems": self.nelems,
        }


@dataclass
class PlacementPlan:
    """Full placement of a logical GEMM C=A@B across devices."""

    global_shape: tuple[int, int]  # C shape (M, N)
    shards: list[Shard] = field(default_factory=list)
    strategy: str = "row_blocks"
    # W2.1 extensions
    k: int = 0  # inner dim of GEMM
    dtype: str = "float64"
    backend: str = "VE_NLC"
    estimated_transfer_sec: float = 0.0
    estimated_compute_sec: float = 0.0
    estimated_total_sec: float = 0.0
    transfer_bytes: int = 0
    version: str = "0.2"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        m, n = self.global_shape
        if m < 0 or n < 0:
            raise ValueError("shape must be non-negative")
        if self.strategy == "row_blocks":
            covered = [0] * m
            for s in self.shards:
                if s.rows <= 0 or s.cols <= 0:
                    raise ValueError(f"empty shard on {s.device}")
                if not (0 <= s.row_start < s.row_end <= m):
                    raise ValueError(f"row range OOB on {s.device}")
                if s.col_start != 0 or s.col_end != n:
                    raise ValueError(f"row_blocks requires full-width cols on {s.device}")
                for r in range(s.row_start, s.row_end):
                    covered[r] += 1
                if size(s.layout) != s.rows * s.cols:
                    raise ValueError(f"layout size mismatch on {s.device}")
            if any(c != 1 for c in covered):
                bad = [i for i, c in enumerate(covered) if c != 1]
                raise ValueError(f"row coverage invalid at rows {bad[:8]}...")
        elif self.strategy == "col_blocks":
            covered = [0] * n
            for s in self.shards:
                if s.rows != m or s.cols <= 0:
                    raise ValueError(f"col_blocks bad rows/cols on {s.device}")
                if not (0 <= s.col_start < s.col_end <= n):
                    raise ValueError(f"col range OOB on {s.device}")
                for c in range(s.col_start, s.col_end):
                    covered[c] += 1
                if size(s.layout) != s.rows * s.cols:
                    raise ValueError(f"layout size mismatch on {s.device}")
            if any(c != 1 for c in covered):
                bad = [i for i, c in enumerate(covered) if c != 1]
                raise ValueError(f"col coverage invalid at cols {bad[:8]}...")
        elif self.strategy == "k_split":
            if self.k <= 0:
                raise ValueError("k_split requires plan.k > 0")
            covered = [0] * self.k
            for s in self.shards:
                if s.rows != m or s.cols != n:
                    raise ValueError(f"k_split shard must cover full C on {s.device}")
                ke = s.k_end if s.k_end > 0 else self.k
                if not (0 <= s.k_start < ke <= self.k):
                    raise ValueError(f"k range OOB on {s.device}")
                for ki in range(s.k_start, ke):
                    covered[ki] += 1
            if any(c != 1 for c in covered):
                bad = [i for i, c in enumerate(covered) if c != 1]
                raise ValueError(f"k coverage invalid at {bad[:8]}...")
        else:
            # permissive for unknown strategies
            for s in self.shards:
                if size(s.layout) != s.rows * s.cols:
                    raise ValueError(f"layout size mismatch on {s.device}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "global_shape": list(self.global_shape),
            "k": self.k,
            "strategy": self.strategy,
            "dtype": self.dtype,
            "backend": self.backend,
            "estimated_transfer_sec": self.estimated_transfer_sec,
            "estimated_compute_sec": self.estimated_compute_sec,
            "estimated_total_sec": self.estimated_total_sec,
            "transfer_bytes": self.transfer_bytes,
            "meta": self.meta,
            "shards": [s.to_dict() for s in self.shards],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PlacementPlan":
        shape = d.get("global_shape", [0, 0])
        m, n = int(shape[0]), int(shape[1])
        shards: list[Shard] = []
        for sd in d.get("shards", []):
            rows = int(sd["row_end"]) - int(sd["row_start"])
            cols = int(sd["col_end"]) - int(sd["col_start"])
            strides = sd.get("strides", [1, rows])
            if isinstance(strides, list):
                strides_t = (int(strides[0]), int(strides[1]) if len(strides) > 1 else rows)
            else:
                strides_t = (1, rows)
            # rebuild layout from local shape + strides
            layout = Layout((rows, cols), strides_t)
            shards.append(
                Shard(
                    device=str(sd["device"]),
                    row_start=int(sd["row_start"]),
                    row_end=int(sd["row_end"]),
                    col_start=int(sd["col_start"]),
                    col_end=int(sd["col_end"]),
                    layout=layout,
                    atom_name=str(sd.get("atom_name", "")),
                    backend=str(sd.get("backend", d.get("backend", "VE_NLC"))),
                    dtype=str(sd.get("dtype", d.get("dtype", "float64"))),
                    strides=strides_t,
                    k_start=int(sd.get("k_start", 0)),
                    k_end=int(sd.get("k_end", 0)),
                    affinity=str(sd.get("affinity", "")),
                )
            )
        plan = PlacementPlan(
            global_shape=(m, n),
            shards=shards,
            strategy=str(d.get("strategy", "row_blocks")),
            k=int(d.get("k", 0)),
            dtype=str(d.get("dtype", "float64")),
            backend=str(d.get("backend", "VE_NLC")),
            estimated_transfer_sec=float(d.get("estimated_transfer_sec", 0.0)),
            estimated_compute_sec=float(d.get("estimated_compute_sec", 0.0)),
            estimated_total_sec=float(d.get("estimated_total_sec", 0.0)),
            transfer_bytes=int(d.get("transfer_bytes", 0)),
            version=str(d.get("version", "0.2")),
            meta=dict(d.get("meta") or {}),
        )
        return plan

    @staticmethod
    def from_json(text: str) -> "PlacementPlan":
        return PlacementPlan.from_dict(json.loads(text))

    @staticmethod
    def load_json(path: Union[str, Path]) -> "PlacementPlan":
        return PlacementPlan.from_json(Path(path).read_text(encoding="utf-8"))


def _split_extent(extent: int, n_parts: int) -> list[tuple[int, int]]:
    """Split [0, extent) into n_parts nearly equal contiguous ranges."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    if extent < 0:
        raise ValueError("extent must be non-negative")
    if n_parts > extent and extent > 0:
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


def _make_shard(
    device: str,
    rs: int,
    re: int,
    cs: int,
    ce: int,
    *,
    atom_name: str,
    backend: str,
    dtype: str,
    column_major: bool = True,
    k_start: int = 0,
    k_end: int = 0,
) -> Shard:
    rows, cols = re - rs, ce - cs
    if column_major:
        layout = Layout((rows, cols), (1, rows))
        strides = (1, rows)
    else:
        layout = Layout((rows, cols), (cols, 1))
        strides = (cols, 1)
    return Shard(
        device=device,
        row_start=rs,
        row_end=re,
        col_start=cs,
        col_end=ce,
        layout=layout,
        atom_name=atom_name,
        backend=backend,
        dtype=dtype,
        strides=strides,
        k_start=k_start,
        k_end=k_end,
        affinity=device,
    )


def partition_matrix_rows(
    m: int,
    n: int,
    devices: Sequence[str],
    *,
    atom_name: str = "VE_NLC_DGEMM_64x64x64_F64",
    column_major: bool = True,
    backend: str = "VE_NLC",
    dtype: str = "float64",
    k: int = 0,
) -> PlacementPlan:
    """Row-block partition of an M×N matrix across devices."""
    if not devices:
        raise ValueError("devices must be non-empty")
    ranges = _split_extent(m, len(devices))
    shards = [
        _make_shard(
            dev,
            rs,
            re,
            0,
            n,
            atom_name=atom_name,
            backend=backend,
            dtype=dtype,
            column_major=column_major,
            k_end=k,
        )
        for dev, (rs, re) in zip(devices, ranges)
    ]
    plan = PlacementPlan(
        global_shape=(m, n),
        shards=shards,
        strategy="row_blocks",
        k=k,
        dtype=dtype,
        backend=backend,
    )
    plan.validate()
    return plan


def partition_matrix_cols(
    m: int,
    n: int,
    devices: Sequence[str],
    *,
    atom_name: str = "VE_NLC_DGEMM_64x64x64_F64",
    column_major: bool = True,
    backend: str = "VE_NLC",
    dtype: str = "float64",
    k: int = 0,
) -> PlacementPlan:
    """Column-block partition of C (and B panels)."""
    if not devices:
        raise ValueError("devices must be non-empty")
    ranges = _split_extent(n, len(devices))
    shards = [
        _make_shard(
            dev,
            0,
            m,
            cs,
            ce,
            atom_name=atom_name,
            backend=backend,
            dtype=dtype,
            column_major=column_major,
            k_end=k,
        )
        for dev, (cs, ce) in zip(devices, ranges)
    ]
    plan = PlacementPlan(
        global_shape=(m, n),
        shards=shards,
        strategy="col_blocks",
        k=k,
        dtype=dtype,
        backend=backend,
    )
    plan.validate()
    return plan


def partition_matrix_k(
    m: int,
    n: int,
    k: int,
    devices: Sequence[str],
    *,
    atom_name: str = "VE_NLC_DGEMM_64x64x64_F64",
    column_major: bool = True,
    backend: str = "VE_NLC",
    dtype: str = "float64",
) -> PlacementPlan:
    """K-split: each device owns a K panel; results reduce on host."""
    if not devices:
        raise ValueError("devices must be non-empty")
    if k <= 0:
        raise ValueError("k must be positive")
    ranges = _split_extent(k, len(devices))
    shards = [
        _make_shard(
            dev,
            0,
            m,
            0,
            n,
            atom_name=atom_name,
            backend=backend,
            dtype=dtype,
            column_major=column_major,
            k_start=ks,
            k_end=ke,
        )
        for dev, (ks, ke) in zip(devices, ranges)
    ]
    plan = PlacementPlan(
        global_shape=(m, n),
        shards=shards,
        strategy="k_split",
        k=k,
        dtype=dtype,
        backend=backend,
    )
    plan.validate()
    return plan


def partition_to_devices(
    m: int,
    n: int,
    devices: Sequence[str],
    *,
    prefer_kind: str = "ve",
    k: int = 0,
) -> PlacementPlan:
    """Convenience: pick atom/backend by prefer_kind and row-partition."""
    backend = {
        "ve": "VE_NLC",
        "phi": "PHI_MKL",
        "host": "HOST_OBLAS",
    }.get(prefer_kind, "VE_NLC")
    atom = BACKEND_ATOMS.get(backend, "VE_NLC_DGEMM_64x64x64_F64")
    return partition_matrix_rows(m, n, devices, atom_name=atom, backend=backend, k=k)
