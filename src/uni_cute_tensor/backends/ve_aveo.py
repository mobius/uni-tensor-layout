"""AVEO (libveo) DGEMM backend — memory offload without file staging."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from uni_cute_tensor.partition.multi_device import PlacementPlan, partition_to_devices

NLC_ROOT = Path(os.environ.get("NLC_ROOT", "/opt/nec/ve/nlc/3.1.0"))
NCC = Path(os.environ.get("NCC", "/opt/nec/ve/bin/ncc"))
VEO_INC = Path(os.environ.get("VEO_INC", "/opt/nec/ve/veos/include"))
VEO_LIB = Path(os.environ.get("VEO_LIB", "/opt/nec/ve/veos/lib64"))
NFORT_LIB = Path(os.environ.get("NFORT_LIB", "/opt/nec/ve/nfort/5.4.1/lib"))

_VE_LIB_SRC = Path(__file__).resolve().parents[1] / "kernels" / "ve" / "dgemm_lib.c"
_HOST_SRC = Path(__file__).resolve().parents[1] / "kernels" / "host" / "veo_dgemm_host.c"
_BUILD = Path(__file__).resolve().parents[3] / "build" / "aveo"
_VE_SO = _BUILD / "libve_dgemm.so"
# VE library must live on a path the VE process can open (prefer /tmp)
_VE_SO_RUNTIME = Path("/tmp/uni_cute_libve_dgemm.so")
_HOST_SO = _BUILD / "libhost_veo_dgemm.so"


def _ensure_ve_ld_path() -> None:
    parts = [
        str(NLC_ROOT / "lib"),
        str(NFORT_LIB),
        "/opt/nec/ve/lib",
    ]
    cur = os.environ.get("VE_LD_LIBRARY_PATH", "")
    merged = ":".join(parts + ([cur] if cur else []))
    os.environ["VE_LD_LIBRARY_PATH"] = merged


def aveo_available() -> bool:
    return (
        NCC.is_file()
        and VEO_INC.is_dir()
        and (VEO_LIB / "libveo.so").is_file()
        and _VE_LIB_SRC.is_file()
        and _HOST_SRC.is_file()
    )


def compile_aveo_stack(*, force: bool = False) -> tuple[Path, Path]:
    """Build VE .so and host VEO wrapper .so."""
    if not aveo_available():
        raise RuntimeError("AVEO stack unavailable")
    _BUILD.mkdir(parents=True, exist_ok=True)

    if force or not _VE_SO.is_file() or _VE_SO.stat().st_mtime < _VE_LIB_SRC.stat().st_mtime:
        cmd = [
            str(NCC),
            "-O3",
            "-fpic",
            "-shared",
            "-o",
            str(_VE_SO),
            str(_VE_LIB_SRC),
            f"-I{NLC_ROOT / 'include'}",
            f"-L{NLC_ROOT / 'lib'}",
            "-lcblas",
            "-lblas_openmp",
            f"-Wl,-rpath,{NLC_ROOT / 'lib'}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"VE lib compile failed:\n{r.stderr}")
        # install to /tmp for VE visibility
        _VE_SO_RUNTIME.write_bytes(_VE_SO.read_bytes())
        _VE_SO_RUNTIME.chmod(0o755)
    elif not _VE_SO_RUNTIME.is_file():
        _VE_SO_RUNTIME.write_bytes(_VE_SO.read_bytes())
        _VE_SO_RUNTIME.chmod(0o755)

    if force or not _HOST_SO.is_file() or _HOST_SO.stat().st_mtime < _HOST_SRC.stat().st_mtime:
        cmd = [
            "gcc",
            "-O2",
            "-fPIC",
            "-shared",
            "-o",
            str(_HOST_SO),
            str(_HOST_SRC),
            f"-I{VEO_INC}",
            f"-L{VEO_LIB}",
            "-lveo",
            "-lpthread",
            "-ldl",
            f"-Wl,-rpath,{VEO_LIB}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"host VEO compile failed:\n{r.stderr}")

    return _VE_SO_RUNTIME, _HOST_SO


@dataclass
class AveoRunResult:
    device: str
    ve_id: int
    m: int
    n: int
    k: int
    gflops: float
    elapsed_sec: float
    max_abs_err: float
    status: str
    mode: str = "aveo"
    h2d_sec: float = 0.0
    kernel_sec: float = 0.0
    d2h_sec: float = 0.0


def _load_host_lib() -> ctypes.CDLL:
    _, host_so = compile_aveo_stack()
    lib = ctypes.CDLL(str(host_so))
    lib.aveo_dgemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.aveo_dgemm.restype = ctypes.c_int
    lib.aveo_session_open.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.aveo_session_open.restype = ctypes.c_void_p
    lib.aveo_session_dgemm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.aveo_session_dgemm.restype = ctypes.c_int
    lib.aveo_session_dgemm_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.aveo_session_dgemm_async.restype = ctypes.c_int
    lib.aveo_session_dgemm_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.aveo_session_dgemm_batch.restype = ctypes.c_int
    lib.aveo_session_pin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.aveo_session_pin.restype = ctypes.c_int
    lib.aveo_session_dgemm_pinned.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.aveo_session_dgemm_pinned.restype = ctypes.c_int
    lib.aveo_session_close.argtypes = [ctypes.c_void_p]
    lib.aveo_session_close.restype = None
    return lib


def run_aveo_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    ve_node: int = 1,
    session: Optional[ctypes.c_void_p] = None,
    lib: Optional[ctypes.CDLL] = None,
    async_phases: bool = False,
) -> tuple[np.ndarray, AveoRunResult]:
    """C = A @ B on one VE via AVEO. ve_node is ve_exec -N style (1,2,3)."""
    _ensure_ve_ld_path()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    c = np.empty((m, n), dtype=np.float64)
    if lib is None:
        lib = _load_host_lib()
    ve_lib = str(_VE_SO_RUNTIME).encode()
    elapsed = ctypes.c_double(0.0)
    h2d = ctypes.c_double(0.0)
    kern = ctypes.c_double(0.0)
    d2h = ctypes.c_double(0.0)
    t0 = time.perf_counter()
    if session and async_phases:
        rc = lib.aveo_session_dgemm_async(
            session,
            m,
            n,
            k,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(h2d),
            ctypes.byref(kern),
            ctypes.byref(d2h),
            ctypes.byref(elapsed),
        )
        mode = "aveo-async"
    elif session:
        rc = lib.aveo_session_dgemm(
            session,
            m,
            n,
            k,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(elapsed),
        )
        mode = "aveo-session"
    else:
        rc = lib.aveo_dgemm(
            ve_node,
            m,
            n,
            k,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ve_lib,
            ctypes.byref(elapsed),
        )
        mode = "aveo-oneshot"
    wall = time.perf_counter() - t0
    el = float(elapsed.value) if elapsed.value > 0 else wall
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
    # report kernel-only GFLOPS when async phases available
    ker_t = float(kern.value) if kern.value > 0 else el
    gflops = 2.0 * m * n * k / ker_t / 1e9 if ker_t > 0 else 0.0
    return c, AveoRunResult(
        device=f"ve{ve_node}",
        ve_id=ve_node,
        m=m,
        n=n,
        k=k,
        gflops=gflops,
        elapsed_sec=el,
        max_abs_err=err,
        status="pass" if rc == 0 and err < 1e-8 else "fail",
        mode=mode,
        h2d_sec=float(h2d.value),
        kernel_sec=float(kern.value),
        d2h_sec=float(d2h.value),
    )


class AveoSessionPool:
    """One persistent AVEO proc per VE node."""

    def __init__(self, ve_nodes: Sequence[int]):
        self.ve_nodes = list(ve_nodes)
        self._lib: Optional[ctypes.CDLL] = None
        self._sessions: dict[int, ctypes.c_void_p] = {}

    def start(self) -> None:
        _ensure_ve_ld_path()
        self._lib = _load_host_lib()
        ve_lib = str(_VE_SO_RUNTIME).encode()
        for node in self.ve_nodes:
            sess = self._lib.aveo_session_open(node, ve_lib)
            if not sess:
                self.stop()
                raise RuntimeError(f"aveo_session_open({node}) failed")
            self._sessions[node] = sess

    def stop(self) -> None:
        if self._lib:
            for sess in self._sessions.values():
                self._lib.aveo_session_close(sess)
        self._sessions.clear()

    def __enter__(self) -> "AveoSessionPool":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def pin_all(self, max_m: int, max_n: int, max_k: int) -> None:
        assert self._lib is not None
        for node, sess in self._sessions.items():
            rc = self._lib.aveo_session_pin(sess, max_m, max_n, max_k)
            if rc != 0:
                raise RuntimeError(f"aveo_session_pin({node}) rc={rc}")

    def dgemm(
        self,
        ve_node: int,
        a: np.ndarray,
        b: np.ndarray,
        *,
        async_phases: bool = False,
    ):
        return run_aveo_dgemm(
            a,
            b,
            ve_node=ve_node,
            session=self._sessions[ve_node],
            lib=self._lib,
            async_phases=async_phases,
        )

    def dgemm_pinned(
        self, ve_node: int, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, AveoRunResult]:
        """GEMM using pre-pinned VE buffers (no alloc/free per call)."""
        assert self._lib is not None
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        m, k = a.shape
        n = b.shape[1]
        c = np.empty((m, n), dtype=np.float64)
        elapsed = ctypes.c_double(0.0)
        t0 = time.perf_counter()
        rc = self._lib.aveo_session_dgemm_pinned(
            self._sessions[ve_node],
            m,
            n,
            k,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(elapsed),
        )
        wall = time.perf_counter() - t0
        el = float(elapsed.value) if elapsed.value > 0 else wall
        err = float(np.max(np.abs(c - a @ b)))
        gflops = 2.0 * m * n * k / el / 1e9 if el > 0 else 0.0
        return c, AveoRunResult(
            device=f"ve{ve_node}",
            ve_id=ve_node,
            m=m,
            n=n,
            k=k,
            gflops=gflops,
            elapsed_sec=el,
            max_abs_err=err,
            status="pass" if rc == 0 and err < 1e-8 else "fail",
            mode="aveo-pinned",
        )

    def dgemm_batch(
        self,
        ve_node: int,
        batches_a: Sequence[np.ndarray],
        batches_b: Sequence[np.ndarray],
    ) -> tuple[list[np.ndarray], float, float]:
        """Multi-batch on one VE with dual VE buffers. Returns Cs, wall, max_err."""
        if not batches_a or len(batches_a) != len(batches_b):
            raise ValueError("batch length mismatch")
        lib = self._lib
        assert lib is not None
        nbatch = len(batches_a)
        m, k = batches_a[0].shape
        n = batches_b[0].shape[1]
        As = []
        Bs = []
        Cs = []
        a_ptrs = (ctypes.c_void_p * nbatch)()
        b_ptrs = (ctypes.c_void_p * nbatch)()
        c_ptrs = (ctypes.c_void_p * nbatch)()
        for i in range(nbatch):
            a = np.ascontiguousarray(batches_a[i], dtype=np.float64)
            b = np.ascontiguousarray(batches_b[i], dtype=np.float64)
            c = np.empty((m, n), dtype=np.float64)
            As.append(a)
            Bs.append(b)
            Cs.append(c)
            a_ptrs[i] = a.ctypes.data
            b_ptrs[i] = b.ctypes.data
            c_ptrs[i] = c.ctypes.data
        elapsed = ctypes.c_double(0.0)
        t0 = time.perf_counter()
        rc = lib.aveo_session_dgemm_batch(
            self._sessions[ve_node],
            nbatch,
            m,
            n,
            k,
            a_ptrs,
            b_ptrs,
            c_ptrs,
            ctypes.byref(elapsed),
        )
        wall = time.perf_counter() - t0
        if rc != 0:
            raise RuntimeError(f"aveo_session_dgemm_batch rc={rc}")
        max_err = 0.0
        for a, b, c in zip(As, Bs, Cs):
            max_err = max(max_err, float(np.max(np.abs(c - a @ b))))
        return Cs, wall, max_err


def multi_ve_aveo_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    devices: Sequence[str],
    *,
    pool: Optional[AveoSessionPool] = None,
) -> tuple[np.ndarray, PlacementPlan, list[AveoRunResult]]:
    """Row-sharded multi-VE DGEMM via AVEO."""
    m, k = a.shape
    n = b.shape[1]
    plan = partition_to_devices(m, n, list(devices), prefer_kind="ve")
    c = np.zeros((m, n), dtype=np.float64)
    results: list[AveoRunResult] = []
    own_pool = pool is None
    if pool is None:
        nodes = []
        for d in devices:
            if d.startswith("ve") and d[2:].isdigit():
                nodes.append(int(d[2:]))
        pool = AveoSessionPool(nodes)
        pool.start()

    def _one(shard):
        ve_id = int(shard.device.replace("ve", ""))
        a_sh = a[shard.row_start : shard.row_end, :]
        c_sh, res = pool.dgemm(ve_id, a_sh, b)
        return slice(shard.row_start, shard.row_end), c_sh, res

    try:
        with ThreadPoolExecutor(max_workers=len(plan.shards)) as ex:
            futs = [ex.submit(_one, s) for s in plan.shards]
            for fut in as_completed(futs):
                sl, c_sh, res = fut.result()
                c[sl, :] = c_sh
                results.append(res)
    finally:
        if own_pool:
            pool.stop()

    results.sort(key=lambda r: r.ve_id)
    return c, plan, results
