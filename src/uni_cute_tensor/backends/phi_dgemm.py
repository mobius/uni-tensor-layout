"""Real Xeon Phi DGEMM: MKL (preferred), IMCI OpenMP, or k1om-gcc fallback.

I/O: scp binary/data to mic0, ssh run, scp results back.
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

_KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels" / "phi"
_SRC_IMCI = _KERNEL_DIR / "dgemm_rect.c"
_SRC_MKL = _KERNEL_DIR / "dgemm_mkl.c"
_BUILD_DIR = Path(__file__).resolve().parents[3] / "build" / "phi"
_BIN_IMCI = _BUILD_DIR / "dgemm_rect.mic"
_BIN_MKL = _BUILD_DIR / "dgemm_mkl.mic"

DEFAULT_CONTAINER = os.environ.get("PHI_PODMAN_CONTAINER", "centos7-phi-dev")
DEFAULT_LICENSE = Path(
    os.environ.get(
        "INTEL_LICENSE_FILE",
        str(Path.home() / "parallel_studio.lic"),
    ).split(":")[0]
)
MIC_HOST = os.environ.get("PHI_SSH_HOST", "mic0")
MIC_LIBS_HOST = Path(
    os.environ.get(
        "PHI_MIC_LIBS",
        str(Path.home() / "Work" / "intel_phi" / "icc_mic_libs"),
    )
)
MKL_ROOT_DEFAULT = (
    "/opt/intel/compilers_and_libraries_2016.0.109/linux/mkl"
)


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
    if not _podman_available():
        return {"ok": False, "reason": "podman missing"}
    _ensure_license_in_container(container, DEFAULT_LICENSE)
    script = r"""
source /opt/intel/bin/compilervars.sh intel64 >/dev/null 2>&1
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
echo 'int main(){return 0;}' > /tmp/licprobe.c
icc -std=c99 -mmic -O0 -o /tmp/licprobe.mic /tmp/licprobe.c >/tmp/licprobe.err 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo ICC_LICENSE_OK; else echo ICC_LICENSE_FAIL
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
    return {
        "ok": "ICC_LICENSE_OK" in out,
        "feature_requested": "Comp-CL",
        "license_path_used": str(DEFAULT_LICENSE) if DEFAULT_LICENSE.is_file() else "missing",
        "log_excerpt": "\n".join(out.splitlines()[:12]),
    }


def _uptodate(bin_path: Path, src: Path) -> bool:
    return bin_path.is_file() and bin_path.stat().st_mtime >= src.stat().st_mtime


def compile_phi_dgemm_imci(
    *,
    force: bool = False,
    prefer_icc: bool = True,
    container: str = DEFAULT_CONTAINER,
) -> tuple[Path, str]:
    """Compile IMCI OpenMP dgemm_rect.mic."""
    if not _SRC_IMCI.is_file():
        raise FileNotFoundError(_SRC_IMCI)
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _uptodate(_BIN_IMCI, _SRC_IMCI) and not force:
        tag = "cached"
        meta = _BUILD_DIR / "compiler_imci.txt"
        if meta.is_file():
            tag = meta.read_text(encoding="utf-8").strip() or tag
        return _BIN_IMCI, tag

    if not _podman_available():
        raise RuntimeError("podman required to compile Phi kernels")

    subprocess.run(
        ["podman", "cp", str(_SRC_IMCI), f"{container}:/tmp/dgemm_rect.c"],
        check=True,
        capture_output=True,
        text=True,
    )

    compiler = "k1om-gcc"
    if prefer_icc and try_icc_license(container).get("ok"):
        compiler = "icc-mmic"

    if compiler == "icc-mmic":
        script = r"""
set -e
source /opt/intel/bin/compilervars.sh intel64
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
icc -std=c99 -mmic -O3 -openmp -restrict -o /tmp/dgemm_rect.mic /tmp/dgemm_rect.c
"""
    else:
        script = r"""
set -e
export PATH=/opt/mpss/3.8.6/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux:$PATH
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
        raise RuntimeError(f"Phi IMCI compile failed ({compiler}):\n{r.stderr}\n{r.stdout}")

    subprocess.run(
        ["podman", "cp", f"{container}:/tmp/dgemm_rect.mic", str(_BIN_IMCI)],
        check=True,
        capture_output=True,
        text=True,
    )
    (_BUILD_DIR / "compiler_imci.txt").write_text(compiler + "\n", encoding="utf-8")
    return _BIN_IMCI, compiler


def compile_phi_dgemm_mkl(
    *,
    force: bool = False,
    container: str = DEFAULT_CONTAINER,
) -> tuple[Path, str]:
    """Compile MKL-native dgemm_mkl.mic (requires ICC + MKL mic libs)."""
    if not _SRC_MKL.is_file():
        raise FileNotFoundError(_SRC_MKL)
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _uptodate(_BIN_MKL, _SRC_MKL) and not force:
        return _BIN_MKL, "icc-mkl-cached"

    if not try_icc_license(container).get("ok"):
        raise RuntimeError("ICC license required for MKL MIC build")

    subprocess.run(
        ["podman", "cp", str(_SRC_MKL), f"{container}:/tmp/dgemm_mkl.c"],
        check=True,
        capture_output=True,
        text=True,
    )
    mkl = os.environ.get("MKLROOT", MKL_ROOT_DEFAULT)
    script = f"""
set -e
source /opt/intel/bin/compilervars.sh intel64
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
MKL="{mkl}"
if [ ! -f "$MKL/include/mkl.h" ]; then
  MKL=/opt/intel/compilers_and_libraries_2016.0.109/linux/mkl
fi
icc -std=c99 -mmic -O3 -openmp -I"$MKL/include" -o /tmp/dgemm_mkl.mic /tmp/dgemm_mkl.c \\
  -L"$MKL/lib/mic" -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
"""
    r = subprocess.run(
        ["podman", "exec", container, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Phi MKL compile failed:\n{r.stderr}\n{r.stdout}")

    subprocess.run(
        ["podman", "cp", f"{container}:/tmp/dgemm_mkl.mic", str(_BIN_MKL)],
        check=True,
        capture_output=True,
        text=True,
    )
    (_BUILD_DIR / "compiler_mkl.txt").write_text("icc-mkl\n", encoding="utf-8")
    return _BIN_MKL, "icc-mkl"


def compile_phi_dgemm(
    *,
    force: bool = False,
    prefer_icc: bool = True,
    backend: str = "auto",
    container: str = DEFAULT_CONTAINER,
) -> tuple[Path, str]:
    """Compile preferred Phi DGEMM binary.

    backend: auto | mkl | imci
      auto prefers MKL when ICC+license available, else IMCI/k1om.
    """
    backend = backend.lower()
    if backend == "auto":
        if prefer_icc and try_icc_license(container).get("ok"):
            try:
                return compile_phi_dgemm_mkl(force=force, container=container)
            except Exception:
                return compile_phi_dgemm_imci(
                    force=force, prefer_icc=True, container=container
                )
        return compile_phi_dgemm_imci(
            force=force, prefer_icc=prefer_icc, container=container
        )
    if backend == "mkl":
        return compile_phi_dgemm_mkl(force=force, container=container)
    if backend in ("imci", "rect"):
        return compile_phi_dgemm_imci(
            force=force, prefer_icc=prefer_icc, container=container
        )
    raise ValueError(f"unknown backend: {backend}")


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


def _deploy_mic_libs(ssh: list[str], scp: list[str], remote_libdir: str, need_mkl: bool) -> None:
    subprocess.run(
        ssh + [MIC_HOST, f"mkdir -p {remote_libdir}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    names = ["libiomp5.so", "libintlc.so.5", "libimf.so", "libsvml.so"]
    if need_mkl:
        names += [
            "libmkl_intel_lp64.so",
            "libmkl_intel_thread.so",
            "libmkl_core.so",
            "libmkl_sequential.so",
        ]
    for name in names:
        src = MIC_LIBS_HOST / name
        if not src.is_file():
            continue
        # skip re-upload if present with same size (cheap check)
        remote = f"{remote_libdir}/{name}"
        chk = subprocess.run(
            ssh + [MIC_HOST, f"test -s {remote} && echo yes || echo no"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "yes" in (chk.stdout or ""):
            continue
        subprocess.run(
            scp + [str(src), f"{MIC_HOST}:{remote}"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )


def run_phi_dgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    work_dir: Optional[Path] = None,
    threads: int = 244,
    timeout: float = 300.0,
    force_recompile: bool = False,
    backend: str = "auto",
) -> tuple[np.ndarray, PhiDgemmResult]:
    """Execute C = A @ B on mic0; verify against host numpy."""
    if not phi_device_present():
        raise RuntimeError("no /dev/mic0")

    binary, compiler = compile_phi_dgemm(force=force_recompile, backend=backend)
    need_mkl = "mkl" in compiler or binary.name.startswith("dgemm_mkl")
    m, k = a.shape
    n = b.shape[1]

    own = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="cct_phi_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    in_path = work_dir / "in.bin"
    out_path = work_dir / "out.bin"
    _write_input(in_path, a, b)

    remote_bin = f"/tmp/uni_cute_{binary.name}"
    remote_in = "/tmp/uni_cute_phi_in.bin"
    remote_out = "/tmp/uni_cute_phi_out.bin"
    remote_libdir = "/tmp/uni_cute_mic_libs"

    scp = _scp_base()
    ssh = _ssh_base()

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
        timeout=180,
    )
    if r2.returncode != 0:
        raise RuntimeError(f"scp input failed: {r2.stderr}")

    _deploy_mic_libs(ssh, scp, remote_libdir, need_mkl=need_mkl)

    remote_cmd = (
        f"export LD_LIBRARY_PATH={remote_libdir}:$LD_LIBRARY_PATH; "
        f"export OMP_NUM_THREADS={int(threads)}; "
        f"export MKL_NUM_THREADS={int(threads)}; "
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
        timeout=180,
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
