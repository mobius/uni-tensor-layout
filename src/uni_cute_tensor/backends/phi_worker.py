"""Persistent Phi worker with low-overhead control (ssh remote shell, minimal scp).

Control plane uses a single ssh invocation to write job.cmd+job.go and poll status
(no scp of control files). Data plane still scp's matrices (or ssh cat).
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from uni_cute_tensor.backends.phi_dgemm import (
    DEFAULT_CONTAINER,
    MIC_HOST,
    _deploy_mic_libs,
    _scp_base,
    _ssh_base,
    try_icc_license,
)
from uni_cute_tensor.backends.phi_prep import PhiPrepResult

_SRC = Path(__file__).resolve().parents[1] / "kernels" / "phi" / "phi_worker.c"
_BUILD = Path(__file__).resolve().parents[3] / "build" / "phi"
_BIN = _BUILD / "phi_worker.mic"


def compile_phi_worker(*, force: bool = False, container: str = DEFAULT_CONTAINER) -> Path:
    if not _SRC.is_file():
        raise FileNotFoundError(_SRC)
    _BUILD.mkdir(parents=True, exist_ok=True)
    if _BIN.is_file() and not force and _BIN.stat().st_mtime >= _SRC.stat().st_mtime:
        return _BIN
    if not try_icc_license(container).get("ok"):
        raise RuntimeError("ICC required for phi_worker")
    subprocess.run(
        ["podman", "cp", str(_SRC), f"{container}:/tmp/phi_worker.c"],
        check=True,
        capture_output=True,
        text=True,
    )
    script = r"""
set -e
source /opt/intel/bin/compilervars.sh intel64
export INTEL_LICENSE_FILE=/opt/intel/licenses/parallel_studio.lic
icc -std=c99 -mmic -O3 -openmp -o /tmp/phi_worker.mic /tmp/phi_worker.c
"""
    r = subprocess.run(
        ["podman", "exec", container, "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"phi_worker compile failed:\n{r.stderr}")
    subprocess.run(
        ["podman", "cp", f"{container}:/tmp/phi_worker.mic", str(_BIN)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _BIN


class PhiWorker:
    """Long-lived phi_worker.mic; control via remote shell (not scp for cmd/go)."""

    def __init__(self, *, remote_dir: str = "/tmp/uni_cute_phi_worker"):
        self.remote_dir = remote_dir
        self._proc: Optional[subprocess.Popen] = None
        self._ssh = _ssh_base()
        self._scp = _scp_base()

    def _remote(self, script: str, *, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._ssh + [MIC_HOST, script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def start(self, *, threads: int = 244) -> None:
        binary = compile_phi_worker()
        remote_lib = "/tmp/uni_cute_mic_libs"
        remote_bin = f"{self.remote_dir}/phi_worker.mic"
        self._remote(f"mkdir -p {self.remote_dir}")
        subprocess.run(
            self._scp + [str(binary), f"{MIC_HOST}:{remote_bin}"],
            check=True,
            capture_output=True,
            text=True,
        )
        _deploy_mic_libs(self._ssh, self._scp, remote_lib, need_mkl=False)
        self._remote(
            f"rm -f {self.remote_dir}/job.go {self.remote_dir}/job.cmd "
            f"{self.remote_dir}/job.status {self.remote_dir}/job.log"
        )
        cmd = (
            f"export LD_LIBRARY_PATH={remote_lib}:$LD_LIBRARY_PATH; "
            f"export OMP_NUM_THREADS={threads}; "
            f"export KMP_AFFINITY=balanced,granularity=fine; "
            f"{remote_bin} {self.remote_dir}"
        )
        self._proc = subprocess.Popen(
            self._ssh + [MIC_HOST, cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.4)
        if self._proc.poll() is not None:
            err = (self._proc.stderr.read() if self._proc.stderr else b"").decode()
            raise RuntimeError(f"phi worker died: {err}")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._submit("QUIT\n", timeout=15.0)
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def __enter__(self) -> "PhiWorker":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def _submit(self, cmd: str, *, timeout: float = 120.0) -> str:
        """Write job via one remote shell (printf) — no scp for control files."""
        # escape for remote single quotes
        safe = cmd.replace("'", "'\"'\"'")
        r = self._remote(
            f"rm -f {self.remote_dir}/job.status {self.remote_dir}/job.log; "
            f"printf %s '{safe}' > {self.remote_dir}/job.cmd; "
            f": > {self.remote_dir}/job.go",
            timeout=30,
        )
        if r.returncode != 0:
            raise RuntimeError(f"phi control write failed: {r.stderr}")
        t0 = time.time()
        while time.time() - t0 < timeout:
            chk = self._remote(
                f"if [ -f {self.remote_dir}/job.status ]; then "
                f"cat {self.remote_dir}/job.status; "
                f"cat {self.remote_dir}/job.log 2>/dev/null; "
                f"else echo WAIT; fi",
                timeout=30,
            )
            out = chk.stdout or ""
            if out.startswith("DONE") or out.startswith("FAIL"):
                return out
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("phi worker process died")
            time.sleep(0.005)
        raise TimeoutError("phi worker job timeout")

    def _put(self, local: Path, remote: str) -> None:
        """Data plane: scp (fallback ssh cat if scp fails)."""
        r = subprocess.run(
            self._scp + [str(local), f"{MIC_HOST}:{remote}"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0:
            with local.open("rb") as f:
                data = f.read()
            p = subprocess.run(
                self._ssh + [MIC_HOST, f"cat > {remote}"],
                input=data,
                capture_output=True,
                timeout=180,
            )
            if p.returncode != 0:
                raise RuntimeError(f"phi put failed: {r.stderr} / {p.stderr!r}")

    def _get(self, remote: str, local: Path) -> None:
        r = subprocess.run(
            self._scp + [f"{MIC_HOST}:{remote}", str(local)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0:
            p = subprocess.run(
                self._ssh + [MIC_HOST, f"cat {remote}"],
                capture_output=True,
                timeout=180,
            )
            if p.returncode != 0:
                raise RuntimeError(f"phi get failed: {r.stderr}")
            local.write_bytes(p.stdout)

    def scale(
        self,
        a: np.ndarray,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        timeout: float = 120.0,
    ) -> tuple[np.ndarray, PhiPrepResult]:
        a = np.ascontiguousarray(a, dtype=np.float64)
        m, n = a.shape
        work = Path(tempfile.mkdtemp(prefix="cct_phi_w_"))
        try:
            inp = work / "in.bin"
            outp = work / "out.bin"
            with inp.open("wb") as f:
                f.write(struct.pack("ii", m, n))
                f.write(struct.pack("dd", float(alpha), float(beta)))
                f.write(a.tobytes())
            rin = f"{self.remote_dir}/in.bin"
            rout = f"{self.remote_dir}/out.bin"
            self._put(inp, rin)
            t0 = time.perf_counter()
            log = self._submit(f"SCALE {rin} {rout}\n", timeout=timeout)
            wall = time.perf_counter() - t0
            if log.startswith("FAIL"):
                raise RuntimeError(log)
            self._get(rout, outp)
            with outp.open("rb") as f:
                mm, nn = struct.unpack("ii", f.read(8))
                c = np.frombuffer(f.read(mm * nn * 8), dtype=np.float64).copy().reshape(mm, nn)
            ref = alpha * a + beta
            err = float(np.max(np.abs(c - ref)))
            gflops = 0.0
            mobj = re.search(r"([\d.]+)\s*GFLOPS", log)
            if mobj:
                gflops = float(mobj.group(1))
            return c, PhiPrepResult(
                gflops=gflops,
                elapsed_sec=wall,
                max_abs_err=err,
                status="pass" if err < 1e-9 else "fail",
                stdout=log,
                compiler="phi-worker-sshctl",
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
