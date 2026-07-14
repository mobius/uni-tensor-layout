"""Real Xeon Phi DGEMM backend via k1om-gcc (or ICC if Comp-CL available).

I/O model: host stages binaries/data with scp to mic0, runs over ssh, scp
results back. micnativeloadex does not expose host filesystem paths to the
card process for fopen().
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_KERNEL_SRC = Path(__file__).resolve().parents[1] / "kernels" / "phi" / "dgemm_rect.c"
_BUILD_DIR = Path(__file__).resolve().parents[3] / "build" / "phi"
_KERNEL_BIN = _BUILD_DIR / "dgemm_rect.mic"

DEFAULT_CONTAINER = os.environ.get("PHI_PODMAN_CONTAINER", "centos7-phi-dev")
DEFAULT_LICENSE = Path(
    os.environ.get(
        "INTEL_LICENSE_FILE",
        str(Path.home() / "parallel_studio.lic"),
    ).split(":")[0]
)
MIC_HOST = os.environ.get("PHI_SSH_HOST", "mic0")


@dataclass
class PhiDgemmResult:
    m: int
    n: int
    k: int
    gflops: float
    elapsed_sec: float
    checksum: float
    max_abs_err: float
    status: str
    compiler: str
    stdout: str
    stderr: str
    binary: str


def phi_device_present() -> bool:
    return Path("/dev/mic0").exists()


def _podman_available() -> bool:
    return shutil.which("podman") is not None


def _ssh_base() -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
    ]


def _scp_base() -> list[str]:
    return [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
    ]


def _ensure_license_in_container(container: str, license_path: Path) -> None:
    """Copy host license into container (never log file contents)."""
    if not license_path.is_file():
        return
    subprocess.run(
        ["podman", "exec", container, "mkdir", "-p", "/opt/intel/licenses"],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = license_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    tmp = Path(tempfile.mkstemp(prefix="intel_lic_", suffix=".lic")[1])
    try:
        tmp.write_bytes(raw)
        tmp.chmod(0o644)
        subprocess.run(
            [
                "podman",
                "cp",
                str(tmp),
                f"{container}:/opt/intel/licenses/parallel_studio.lic",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        tmp.unlink(missing_ok=True)


def try_icc_license(container: str = DEFAULT_CONTAINER) -> dict:
    """Probe whether ICC can check out a license. Returns status dict (no secrets)."""
    if not _podman_available():
        return {"ok": False, "reason": "podman missing"}
    _ensure_license_in_container(container, DEFAULT_LICENSE)
    script = r"""
source /opt/intel/bin/compilervars.sh intel64 >/dev/null 2>&1
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
echo 'int main(){return 0;}' > /tmp/licprobe.c
icc -std=c99 -mmic -O0 -o /tmp/licprobe.mic /tmp/licprobe.c >/tmp/licprobe.err 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo ICC_LICENSE_OK
else
  echo ICC_LICENSE_FAIL
  sed -E 's/[0-9A-Fa-f]{10,}/<hex>/g' /tmp/licprobe.err | head -20
fi
"""
    r = subprocess.run(
        ["podman", "exec", container, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (r.stdout or "") + (r.stderr or "")
    ok = "ICC_LICENSE_OK" in out
    return {
        "ok": ok,
        "feature_requested": "Comp-CL",
        "license_path_used": str(DEFAULT_LICENSE) if DEFAULT_LICENSE.is_file() else "missing",
        "log_excerpt": "\n".join(out.splitlines()[:12]),
    }


def compile_phi_dgemm(
    *,
    force: bool = False,
    prefer_icc: bool = True,
    container: str = DEFAULT_CONTAINER,
) -> tuple[Path, str]:
    """Compile dgemm_rect.mic. Returns (binary_path, compiler_tag)."""
    if not _KERNEL_SRC.is_file():
        raise FileNotFoundError(_KERNEL_SRC)
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _KERNEL_BIN.is_file() and not force:
        if _KERNEL_BIN.stat().st_mtime >= _KERNEL_SRC.stat().st_mtime:
            tag = "cached"
            meta = _BUILD_DIR / "compiler.txt"
            if meta.is_file():
                tag = meta.read_text(encoding="utf-8").strip() or tag
            return _KERNEL_BIN, tag

    if not _podman_available():
        raise RuntimeError("podman required to compile Phi kernels")

    subprocess.run(
        ["podman", "cp", str(_KERNEL_SRC), f"{container}:/tmp/dgemm_rect.c"],
        check=True,
        capture_output=True,
        text=True,
    )

    compiler = "k1om-gcc"
    if prefer_icc:
        probe = try_icc_license(container)
        if probe.get("ok"):
            compiler = "icc-mmic"

    if compiler == "icc-mmic":
        # OpenMP + IMCI path; -restrict matches intel_phi peak_dgemm flags
        script = r"""
set -e
source /opt/intel/bin/compilervars.sh intel64
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
icc -std=c99 -mmic -O3 -openmp -restrict -o /tmp/dgemm_rect.mic /tmp/dgemm_rect.c
"""
    else:
        script = r"""
set -e
export PATH=/opt/mpss/3.8.6/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux:/opt/mpss/3.8.6/sysroots/x86_64-mpsssdk-linux/usr/bin:$PATH
SYSROOT=/opt/mpss/3.8.6/sysroots/k1om-mpss-linux
k1om-mpss-linux-gcc --sysroot=$SYSROOT -O3 -pthread -o /tmp/dgemm_rect.mic /tmp/dgemm_rect.c
"""

    r = subprocess.run(
        ["podman", "exec", container, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Phi compile failed ({compiler}):\n{r.stderr}\n{r.stdout}")

    subprocess.run(
        ["podman", "cp", f"{container}:/tmp/dgemm_rect.mic", str(_KERNEL_BIN)],
        check=True,
        capture_output=True,
        text=True,
    )
    (_BUILD_DIR / "compiler.txt").write_text(compiler + "\n", encoding="utf-8")
    return _KERNEL_BIN, compiler


def _write_input(path: Path, a: np.ndarray, b: np.ndarray) -> None:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    with path.open("wb") as f:
        f.write(struct.pack("iii", m, n, k))
        f.write(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(b, dtype=np.float64).tobytes())


def _read_output(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        m, n = struct.unpack("ii", f.read(8))
        data = f.read(m * n * 8)
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


def run_phi_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    work_dir: Optional[Path] = None,
    threads: int = 120,
    timeout: float = 300.0,
    force_recompile: bool = False,
) -> tuple[np.ndarray, PhiDgemmResult]:
    """Execute C = A @ B on mic0; verify against host numpy."""
    if not phi_device_present():
        raise RuntimeError("no /dev/mic0")

    binary, compiler = compile_phi_dgemm(force=force_recompile)
    m, k = a.shape
    n = b.shape[1]

    own = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="cct_phi_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    in_path = work_dir / "in.bin"
    out_path = work_dir / "out.bin"
    _write_input(in_path, a, b)

    remote_bin = "/tmp/uni_cute_dgemm_rect.mic"
    remote_in = "/tmp/uni_cute_phi_in.bin"
    remote_out = "/tmp/uni_cute_phi_out.bin"
    remote_libdir = "/tmp/uni_cute_mic_libs"

    scp = _scp_base()
    ssh = _ssh_base()

    # deploy binary + input
    r1 = subprocess.run(
        scp + [str(binary), f"{MIC_HOST}:{remote_bin}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r1.returncode != 0:
        raise RuntimeError(f"scp binary failed: {r1.stderr}")
    r2 = subprocess.run(
        scp + [str(in_path), f"{MIC_HOST}:{remote_in}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r2.returncode != 0:
        raise RuntimeError(f"scp input failed: {r2.stderr}")

    # OpenMP runtime for ICC -openmp MIC binaries (ssh path has no SINK_LD_*)
    mic_libs = Path.home() / "Work" / "intel_phi" / "icc_mic_libs"
    iomp = mic_libs / "libiomp5.so"
    if iomp.is_file():
        subprocess.run(
            ssh + [MIC_HOST, f"mkdir -p {remote_libdir}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            scp + [str(iomp), f"{MIC_HOST}:{remote_libdir}/libiomp5.so"],
            capture_output=True,
            text=True,
            timeout=60,
        )

    # OMP_NUM_THREADS for ICC OpenMP build; PHI_DGEMM_THREADS also read by kernel
    remote_cmd = (
        f"export LD_LIBRARY_PATH={remote_libdir}:$LD_LIBRARY_PATH; "
        f"export OMP_NUM_THREADS={int(threads)}; "
        f"export PHI_DGEMM_THREADS={int(threads)}; "
        f"export KMP_AFFINITY=balanced,granularity=fine; "
        f"{remote_bin} {remote_in} {remote_out}"
    )
    t0 = time.perf_counter()
    r3 = subprocess.run(
        ssh + [MIC_HOST, remote_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    text = (r3.stdout or "") + "\n" + (r3.stderr or "")

    if r3.returncode != 0:
        if own:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"Phi ssh run failed:\n{text}")

    r4 = subprocess.run(
        scp + [f"{MIC_HOST}:{remote_out}", str(out_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r4.returncode != 0 or not out_path.is_file():
        if own:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"scp output failed: {r4.stderr}")

    c = _read_output(out_path)
    ref = np.asarray(a @ b, dtype=np.float64)
    err = float(np.max(np.abs(c - ref)))
    result = PhiDgemmResult(
        m=m,
        n=n,
        k=k,
        gflops=_parse_gflops(text),
        elapsed_sec=_parse_elapsed(text) or wall,
        checksum=_parse_checksum(text),
        max_abs_err=err,
        status="pass" if err < 1e-6 else "fail",
        compiler=compiler,
        stdout=r3.stdout,
        stderr=r3.stderr,
        binary=str(binary),
    )
    if own:
        shutil.rmtree(work_dir, ignore_errors=True)
    return c, result
