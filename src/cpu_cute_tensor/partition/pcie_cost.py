"""Rough PCIe H2D cost model for planning (not a benchmark)."""

from __future__ import annotations

# Conservative Gen3 x16 usable payload ~12 GB/s (below theoretical 15.75)
DEFAULT_PCIE_GBPS = 12.0


def estimate_h2d_seconds(nbytes: int, gbps: float = DEFAULT_PCIE_GBPS) -> float:
    """Estimate host-to-device transfer time in seconds."""
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if gbps <= 0:
        raise ValueError("gbps must be positive")
    return (nbytes / (1024**3)) / gbps


def estimate_gemm_transfer_bytes(m: int, n: int, k: int, dtype_bytes: int = 8) -> dict[str, int]:
    """Bytes for A (m×k), B (k×n), C (m×n) FP64 by default."""
    a = m * k * dtype_bytes
    b = k * n * dtype_bytes
    c = m * n * dtype_bytes
    return {"A": a, "B": b, "C": c, "total_h2d": a + b, "total_d2h": c}
