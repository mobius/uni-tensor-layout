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
    t0 = time.perf_counter()
    if session:
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
    wall = time.perf_counter() - t0
    el = float(elapsed.value) if elapsed.value > 0 else wall
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
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
        mode="aveo-session" if session else "aveo-oneshot",
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

    def dgemm(self, ve_node: int, a: np.ndarray, b: np.ndarray):
        return run_aveo_dgemm(
            a, b, ve_node=ve_node, session=self._sessions[ve_node], lib=self._lib
        )


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
