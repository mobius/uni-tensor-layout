"""Real NEC VE DGEMM backend: ncc + NLC + ve_exec + filesystem staging."""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from uni_cute_tensor.partition.multi_device import PlacementPlan, partition_to_devices

NLC_ROOT = Path(os.environ.get("NLC_ROOT", "/opt/nec/ve/nlc/3.1.0"))
VE_EXEC = Path(os.environ.get("VE_EXEC", "/opt/nec/ve/bin/ve_exec"))
NCC = Path(os.environ.get("NCC", "/opt/nec/ve/bin/ncc"))

_KERNEL_SRC = Path(__file__).resolve().parents[1] / "kernels" / "ve" / "dgemm_rect.c"
_BUILD_DIR = Path(__file__).resolve().parents[3] / "build" / "ve"
_KERNEL_BIN = _BUILD_DIR / "dgemm_rect_ve"


@dataclass
class VeRunResult:
    device: str
    ve_id: int
    m: int
    n: int
    k: int
    gflops: float
    elapsed_sec: float
    checksum: float
    max_abs_err: float
    stdout: str
    stderr: str
    status: str


def ve_toolchain_available() -> bool:
    return NCC.is_file() and VE_EXEC.is_file() and (NLC_ROOT / "include" / "cblas.h").is_file()


def compile_ve_dgemm(*, force: bool = False) -> Path:
    """Compile dgemm_rect_ve with ncc. Returns binary path."""
    if not ve_toolchain_available():
        raise RuntimeError("VE toolchain not available (ncc/ve_exec/NLC)")
    if not _KERNEL_SRC.is_file():
        raise FileNotFoundError(f"kernel source missing: {_KERNEL_SRC}")

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _KERNEL_BIN.is_file() and not force:
        if _KERNEL_BIN.stat().st_mtime >= _KERNEL_SRC.stat().st_mtime:
            return _KERNEL_BIN

    inc = NLC_ROOT / "include"
    lib = NLC_ROOT / "lib"
    cmd = [
        str(NCC),
        "-O3",
        "-fopenmp",
        "-o",
        str(_KERNEL_BIN),
        str(_KERNEL_SRC),
        f"-I{inc}",
        f"-L{lib}",
        "-lcblas",
        "-lblas_openmp",
        f"-Wl,-rpath,{lib}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ncc failed:\n{r.stderr}\n{r.stdout}")
    return _KERNEL_BIN


def _write_input(path: Path, a: np.ndarray, b: np.ndarray) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A,B must be 2-D")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    a_c = np.ascontiguousarray(a, dtype=np.float64)
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    with path.open("wb") as f:
        f.write(struct.pack("iii", m, n, k))
        f.write(a_c.tobytes(order="C"))
        f.write(b_c.tobytes(order="C"))


def _read_output(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            raise ValueError("short output header")
        m, n = struct.unpack("ii", hdr)
        data = f.read(m * n * 8)
        if len(data) != m * n * 8:
            raise ValueError("short output body")
    return np.frombuffer(data, dtype=np.float64).copy().reshape(m, n)


def _parse_gflops(text: str) -> float:
    m = re.search(r"([\d.]+)\s*GFLOPS", text)
    return float(m.group(1)) if m else 0.0


def _parse_elapsed(text: str) -> float:
    m = re.search(r"([\d.]+)s\s+[\d.]+\s*GFLOPS", text)
    return float(m.group(1)) if m else 0.0


def _parse_checksum(text: str) -> float:
    m = re.search(r"checksum=([+\-eE\d.]+)", text)
    return float(m.group(1)) if m else float("nan")


def _ve_id_from_name(device: str) -> int:
    # ve1 / ve2 / ve3 → 1/2/3
    if device.startswith("ve") and device[2:].isdigit():
        return int(device[2:])
    raise ValueError(f"not a VE device name: {device}")


def run_ve_dgemm_shard(
    a_shard: np.ndarray,
    b: np.ndarray,
    *,
    ve_id: int,
    work_dir: Optional[Path] = None,
    timeout: float = 300.0,
) -> tuple[np.ndarray, VeRunResult]:
    """Run C = A_shard @ B on one VE node."""
    binary = compile_ve_dgemm()
    m, k = a_shard.shape
    n = b.shape[1]

    own_tmp = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix=f"cct_ve{ve_id}_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    in_path = work_dir / f"in_ve{ve_id}.bin"
    out_path = work_dir / f"out_ve{ve_id}.bin"
    _write_input(in_path, a_shard, b)

    env = os.environ.copy()
    nlc_lib = str(NLC_ROOT / "lib")
    prev = env.get("VE_LD_LIBRARY_PATH", "")
    env["VE_LD_LIBRARY_PATH"] = nlc_lib if not prev else f"{nlc_lib}:{prev}"

    cmd = [str(VE_EXEC), "-N", str(ve_id), str(binary), str(in_path), str(out_path)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    wall = time.perf_counter() - t0

    if r.returncode != 0:
        result = VeRunResult(
            device=f"ve{ve_id}",
            ve_id=ve_id,
            m=m,
            n=n,
            k=k,
            gflops=0.0,
            elapsed_sec=wall,
            checksum=float("nan"),
            max_abs_err=float("inf"),
            stdout=r.stdout,
            stderr=r.stderr,
            status="fail",
        )
        if own_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"ve_exec failed on ve{ve_id}: {r.stderr or r.stdout}")

    c = _read_output(out_path)
    ref = a_shard @ b
    err = float(np.max(np.abs(c - ref)))
    result = VeRunResult(
        device=f"ve{ve_id}",
        ve_id=ve_id,
        m=m,
        n=n,
        k=k,
        gflops=_parse_gflops(r.stdout),
        elapsed_sec=_parse_elapsed(r.stdout) or wall,
        checksum=_parse_checksum(r.stdout),
        max_abs_err=err,
        stdout=r.stdout,
        stderr=r.stderr,
        status="pass" if err < 1e-8 else "fail",
    )
    if own_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)
    return c, result


def multi_ve_layout_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    devices: Sequence[str],
    *,
    work_dir: Optional[Path] = None,
    parallel: bool = True,
) -> tuple[np.ndarray, PlacementPlan, list[VeRunResult]]:
    """Row-partition A across VEs, run NLC DGEMM, assemble C.

    Uses PlacementPlan from the layout partitioner (logical rows match shards).
    """
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    if not devices:
        raise ValueError("no VE devices")

    plan = partition_to_devices(m, n, list(devices), prefer_kind="ve")
    c = np.zeros((m, n), dtype=np.float64)
    results: list[VeRunResult] = []

    own_tmp = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="cct_multi_ve_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    def _one(shard) -> tuple[slice, np.ndarray, VeRunResult]:
        ve_id = _ve_id_from_name(shard.device)
        a_sh = a[shard.row_start : shard.row_end, :]
        c_sh, res = run_ve_dgemm_shard(
            a_sh, b, ve_id=ve_id, work_dir=work_dir / shard.device
        )
        return slice(shard.row_start, shard.row_end), c_sh, res

    if parallel and len(plan.shards) > 1:
        with ThreadPoolExecutor(max_workers=len(plan.shards)) as ex:
            futs = [ex.submit(_one, s) for s in plan.shards]
            for fut in as_completed(futs):
                sl, c_sh, res = fut.result()
                c[sl, :] = c_sh
                results.append(res)
    else:
        for s in plan.shards:
            sl, c_sh, res = _one(s)
            c[sl, :] = c_sh
            results.append(res)

    # stable order by row_start
    results.sort(key=lambda r: r.ve_id)

    if own_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)
    return c, plan, results
