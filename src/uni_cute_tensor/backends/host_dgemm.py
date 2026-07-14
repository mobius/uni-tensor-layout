"""Host DGEMM paths: numpy reference and optional blocked OpenMP-style Python."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from uni_cute_tensor.atoms.host_avx512 import HOST_AVX512_8x8x8_F64


@dataclass
class HostDgemmResult:
    elapsed_sec: float
    gflops: float
    max_abs_err: float
    status: str
    atom_name: str


def host_numpy_dgemm(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, HostDgemmResult]:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    t0 = time.perf_counter()
    c = a @ b
    elapsed = time.perf_counter() - t0
    m, k = a.shape
    n = b.shape[1]
    gflops = 2.0 * m * n * k / elapsed / 1e9 if elapsed > 0 else 0.0
    return c, HostDgemmResult(
        elapsed_sec=elapsed,
        gflops=gflops,
        max_abs_err=0.0,
        status="pass",
        atom_name="numpy_dgemm",
    )


def host_blocked_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    tile: int | None = None,
) -> tuple[np.ndarray, HostDgemmResult]:
    """Simple tiled GEMM whose tile size follows the Host AVX-512 atom (8).

    This is a correctness/layout exercise, not a peak kernel.
    """
    if tile is None:
        tile = HOST_AVX512_8x8x8_F64.shape_mnk[0]
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    c = np.zeros((m, n), dtype=np.float64)
    t0 = time.perf_counter()
    for i0 in range(0, m, tile):
        i1 = min(i0 + tile, m)
        for j0 in range(0, n, tile):
            j1 = min(j0 + tile, n)
            for p0 in range(0, k, tile):
                p1 = min(p0 + tile, k)
                c[i0:i1, j0:j1] += a[i0:i1, p0:p1] @ b[p0:p1, j0:j1]
    elapsed = time.perf_counter() - t0
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
    gflops = 2.0 * m * n * k / elapsed / 1e9 if elapsed > 0 else 0.0
    return c, HostDgemmResult(
        elapsed_sec=elapsed,
        gflops=gflops,
        max_abs_err=err,
        status="pass" if err < 1e-9 else "fail",
        atom_name=HOST_AVX512_8x8x8_F64.name,
    )
