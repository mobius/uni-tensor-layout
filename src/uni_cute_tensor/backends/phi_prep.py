"""Phi preprocess kernels (scale) for hetero pipelines."""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from uni_cute_tensor.backends.phi_dgemm import (
    DEFAULT_CONTAINER,
    DEFAULT_LICENSE,
    MIC_HOST,
    MIC_LIBS_HOST,
    _deploy_mic_libs,
    _scp_base,
    _ssh_base,
    try_icc_license,
)

_SRC = Path(__file__).resolve().parents[1] / "kernels" / "phi" / "prep_scale.c"
_BUILD = Path(__file__).resolve().parents[3] / "build" / "phi"
_BIN = _BUILD / "prep_scale.mic"


@dataclass
class PhiPrepResult:
    gflops: float
    elapsed_sec: float
    max_abs_err: float
    status: str
    stdout: str
    compiler: str


def compile_phi_prep_scale(*, force: bool = False, container: str = DEFAULT_CONTAINER) -> Path:
    if not _SRC.is_file():
        raise FileNotFoundError(_SRC)
    _BUILD.mkdir(parents=True, exist_ok=True)
    if _BIN.is_file() and not force and _BIN.stat().st_mtime >= _SRC.stat().st_mtime:
        return _BIN
    if not try_icc_license(container).get("ok"):
        raise RuntimeError("ICC required for prep_scale.mic")
    # license inject via try_icc
    subprocess.run(
        ["podman", "cp", str(_SRC), f"{container}:/tmp/prep_scale.c"],
        check=True,
        capture_output=True,
        text=True,
    )
    script = r"""
set -e
source /opt/intel/bin/compilervars.sh intel64
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
icc -std=c99 -mmic -O3 -openmp -o /tmp/prep_scale.mic /tmp/prep_scale.c
"""
    r = subprocess.run(
        ["podman", "exec", container, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"prep_scale compile failed:\n{r.stderr}")
    subprocess.run(
        ["podman", "cp", f"{container}:/tmp/prep_scale.mic", str(_BIN)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _BIN


def run_phi_prep_scale(
    a: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    threads: int = 244,
    timeout: float = 120.0,
) -> tuple[np.ndarray, PhiPrepResult]:
    """C = alpha * A + beta on Phi; verify vs numpy."""
    if not Path("/dev/mic0").exists():
        raise RuntimeError("no mic0")
    binary = compile_phi_prep_scale()
    a = np.ascontiguousarray(a, dtype=np.float64)
    m, n = a.shape
    work = Path(tempfile.mkdtemp(prefix="cct_phi_prep_"))
    try:
        inp = work / "in.bin"
        outp = work / "out.bin"
        with inp.open("wb") as f:
            f.write(struct.pack("ii", m, n))
            f.write(struct.pack("dd", float(alpha), float(beta)))
            f.write(a.tobytes())

        scp, ssh = _scp_base(), _ssh_base()
        remote_lib = "/tmp/uni_cute_mic_libs"
        remote_bin = "/tmp/uni_cute_prep_scale.mic"
        remote_in = "/tmp/uni_cute_prep_in.bin"
        remote_out = "/tmp/uni_cute_prep_out.bin"
        subprocess.run(scp + [str(binary), f"{MIC_HOST}:{remote_bin}"], check=True, capture_output=True)
        subprocess.run(scp + [str(inp), f"{MIC_HOST}:{remote_in}"], check=True, capture_output=True)
        _deploy_mic_libs(ssh, scp, remote_lib, need_mkl=False)

        cmd = (
            f"export LD_LIBRARY_PATH={remote_lib}:$LD_LIBRARY_PATH; "
            f"export OMP_NUM_THREADS={threads}; "
            f"export KMP_AFFINITY=balanced,granularity=fine; "
            f"{remote_bin} {remote_in} {remote_out}"
        )
        t0 = time.perf_counter()
        r = subprocess.run(ssh + [MIC_HOST, cmd], capture_output=True, text=True, timeout=timeout)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)
        subprocess.run(scp + [f"{MIC_HOST}:{remote_out}", str(outp)], check=True, capture_output=True)

        with outp.open("rb") as f:
            mm, nn = struct.unpack("ii", f.read(8))
            c = np.frombuffer(f.read(mm * nn * 8), dtype=np.float64).copy().reshape(mm, nn)
        ref = alpha * a + beta
        err = float(np.max(np.abs(c - ref)))
        import re

        gflops = 0.0
        for line in (r.stdout or "").splitlines():
            mobj = re.search(r"([\d.]+)\s*GFLOPS", line)
            if mobj:
                gflops = float(mobj.group(1))
        return c, PhiPrepResult(
            gflops=gflops,
            elapsed_sec=wall,
            max_abs_err=err,
            status="pass" if err < 1e-9 else "fail",
            stdout=r.stdout or "",
            compiler="icc-mmic",
        )
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
