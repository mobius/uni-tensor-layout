"""Xeon Phi smoke tests.

Full ICC recompilation requires a valid Intel license inside the podman
container (often unavailable). We therefore support:

1. Running a prebuilt peak binary (path via PHI_PEAK_MIC or uni tree).
2. Optional compile path when ICC license works.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PhiSmokeResult:
    status: str
    gflops: float
    theory_pct: float
    elapsed_sec: float
    stdout: str
    stderr: str
    binary: str


def _default_peak_candidates() -> list[Path]:
    env = os.environ.get("PHI_PEAK_MIC")
    out: list[Path] = []
    if env:
        out.append(Path(env))
    uni = os.environ.get("UNI_ROOT")
    if uni:
        out.append(Path(uni) / "src" / "kernels" / "phi" / "peak_fp64.mic")
    home_uni = Path.home() / "Work" / "uni" / "src" / "kernels" / "phi" / "peak_fp64.mic"
    out.append(home_uni)
    return out


def find_phi_peak_binary() -> Optional[Path]:
    for p in _default_peak_candidates():
        if p.is_file():
            return p.resolve()
    return None


def phi_device_present() -> bool:
    return Path("/dev/mic0").exists()


def run_phi_peak_smoke(*, timeout: float = 120.0) -> PhiSmokeResult:
    """Run peak FP64 binary on mic0 via micnativeloadex."""
    if not phi_device_present():
        return PhiSmokeResult(
            status="skip",
            gflops=0.0,
            theory_pct=0.0,
            elapsed_sec=0.0,
            stdout="",
            stderr="no /dev/mic0",
            binary="",
        )
    binary = find_phi_peak_binary()
    if binary is None:
        return PhiSmokeResult(
            status="skip",
            gflops=0.0,
            theory_pct=0.0,
            elapsed_sec=0.0,
            stdout="",
            stderr="no peak_fp64.mic found (set PHI_PEAK_MIC or UNI_ROOT)",
            binary="",
        )

    env = os.environ.copy()
    mic_libs = Path.home() / "Work" / "intel_phi" / "icc_mic_libs"
    if mic_libs.is_dir():
        prev = env.get("SINK_LD_LIBRARY_PATH", "")
        env["SINK_LD_LIBRARY_PATH"] = (
            str(mic_libs) if not prev else f"{mic_libs}:{prev}"
        )

    cmd = ["micnativeloadex", str(binary), "-d", "0", "-t", "60"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    text = (r.stdout or "") + "\n" + (r.stderr or "")
    gflops = 0.0
    m = re.search(r"FP64 GFLOPS:\s*([\d.]+)", text)
    if m:
        gflops = float(m.group(1))
    theory = 0.0
    # Prefer trailing "47.5%" form over "1.208 TFLOPS" inside parentheses.
    m2 = re.search(r"% of theory[^:]*:\s*([\d.]+)\s*%", text)
    if not m2:
        m2 = re.search(r"([\d.]+)\s*%\s*of theory", text, re.I)
    if m2:
        theory = float(m2.group(1))
    elapsed = 0.0
    m3 = re.search(r"Elapsed:\s*([\d.]+)\s*sec", text)
    if m3:
        elapsed = float(m3.group(1))

    # Expect meaningful FP64 throughput on 7120P
    status = "pass" if gflops > 100.0 else "fail"
    if r.returncode != 0 and gflops <= 0:
        status = "fail"

    return PhiSmokeResult(
        status=status,
        gflops=gflops,
        theory_pct=theory,
        elapsed_sec=elapsed,
        stdout=r.stdout,
        stderr=r.stderr,
        binary=str(binary),
    )
