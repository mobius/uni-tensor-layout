"""Host DGEMM: numpy reference, pure-Python tiles, and AVX-512 OpenMP C kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uni_cute_tensor.atoms.host_avx512 import HOST_AVX512_8x8x8_F64

_SRC = Path(__file__).resolve().parents[1] / "kernels" / "host" / "dgemm_avx512.c"
_BUILD = Path(__file__).resolve().parents[3] / "build" / "host"
_SO = _BUILD / "libhost_dgemm_avx512.so"


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
    """Pure-Python tiled GEMM (atom tile=8) for layout correctness.

    .. deprecated:: 1.0
        Teaching / correctness path only — **not** a performance baseline.
        Prefer ``host_dgemm(..., backend="openblas")`` or ``"auto"``.
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


def compile_host_avx512(*, force: bool = False) -> Path:
    """Build shared library with gcc -mavx512f -fopenmp."""
    if not _SRC.is_file():
        raise FileNotFoundError(_SRC)
    _BUILD.mkdir(parents=True, exist_ok=True)
    if _SO.is_file() and not force:
        if _SO.stat().st_mtime >= _SRC.stat().st_mtime:
            return _SO
    cmd = [
        "gcc",
        "-O3",
        "-fopenmp",
        "-mavx512f",
        "-mfma",
        "-fPIC",
        "-shared",
        "-o",
        str(_SO),
        str(_SRC),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"host avx512 compile failed:\n{r.stderr}\n{r.stdout}")
    return _SO


def _load_lib() -> ctypes.CDLL:
    so = compile_host_avx512()
    lib = ctypes.CDLL(str(so))
    lib.host_dgemm_avx512.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.host_dgemm_avx512.restype = None
    return lib


def host_avx512_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    threads: int | None = None,
) -> tuple[np.ndarray, HostDgemmResult]:
    """Native OpenMP+AVX-512 DGEMM via ctypes."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    c = np.empty((m, n), dtype=np.float64)

    if threads is not None and threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)

    lib = _load_lib()
    # warmup
    lib.host_dgemm_avx512(
        m,
        n,
        k,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    t0 = time.perf_counter()
    lib.host_dgemm_avx512(
        m,
        n,
        k,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    elapsed = time.perf_counter() - t0
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
    gflops = 2.0 * m * n * k / elapsed / 1e9 if elapsed > 0 else 0.0
    return c, HostDgemmResult(
        elapsed_sec=elapsed,
        gflops=gflops,
        max_abs_err=err,
        status="pass" if err < 1e-8 else "fail",
        atom_name="HOST_AVX512_OMP_FMA",
    )


def host_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    backend: str = "auto",
    threads: int | None = None,
    auto_threshold: int = 768,
) -> tuple[np.ndarray, HostDgemmResult]:
    """Host DGEMM with backend selection.

    backend:
      - openblas / numpy: numpy ``@`` (OpenBLAS in this env)
      - avx512: hand-written OpenMP+AVX-512
      - auto: avx512 if max(M,N,K) < auto_threshold else openblas
    """
    backend = backend.lower()
    m, k = a.shape
    n = b.shape[1]
    if backend == "auto":
        backend = "avx512" if max(m, n, k) < auto_threshold else "openblas"
    if backend in ("openblas", "numpy", "blas"):
        c, r = host_numpy_dgemm(a, b)
        r = HostDgemmResult(
            elapsed_sec=r.elapsed_sec,
            gflops=r.gflops,
            max_abs_err=r.max_abs_err,
            status=r.status,
            atom_name="openblas_numpy",
        )
        return c, r
    if backend == "avx512":
        return host_avx512_dgemm(a, b, threads=threads)
    if backend == "blocked":
        return host_blocked_dgemm(a, b)
    raise ValueError(f"unknown host backend: {backend}")
