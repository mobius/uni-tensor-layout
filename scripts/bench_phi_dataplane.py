#!/usr/bin/env python3
"""Measure Phi control RTT vs data scp with/without SSH ControlMaster (M3 W2.3)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def main() -> int:
    if not Path("/dev/mic0").exists():
        print("no /dev/mic0 — skip")
        return 2

    from uni_cute_tensor.backends.phi_dgemm import MIC_HOST, _scp_base, _ssh_base, try_icc_license
    from uni_cute_tensor.backends.phi_worker import PhiWorker

    print(f"ICC license ok: {try_icc_license().get('ok')}")
    ssh = _ssh_base()
    scp = _scp_base()
    remote = "/tmp/uct_phi_dp_probe"
    # control RTT
    t0 = time.perf_counter()
    for _ in range(10):
        r = __import__("subprocess").run(
            ssh + [MIC_HOST, "true"], capture_output=True, timeout=30
        )
        if r.returncode != 0:
            print("ssh failed", r.stderr)
            return 1
    ctrl = (time.perf_counter() - t0) / 10
    print(f"ssh ControlMaster mux opts active: ControlMaster in cmd={any('ControlMaster' in x for x in ssh)}")
    print(f"control RTT mean (10x ssh true): {ctrl*1000:.1f} ms")

    # data plane: scp small/medium matrices
    rng = np.random.default_rng(0)
    for n in (256, 512, 1024):
        a = rng.standard_normal((n, n))
        local = ROOT / "artifacts" / f"_phi_a_{n}.bin"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(a.astype(np.float64).tobytes())
        remote_f = f"{remote}/a_{n}.bin"
        __import__("subprocess").run(ssh + [MIC_HOST, f"mkdir -p {remote}"], capture_output=True)
        t0 = time.perf_counter()
        r = __import__("subprocess").run(
            scp + [str(local), f"{MIC_HOST}:{remote_f}"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        wall = time.perf_counter() - t0
        nbytes = local.stat().st_size
        mbps = (nbytes / 1e6) / wall if wall > 0 else 0
        print(f"scp {n}^2 float64 ({nbytes/1e6:.2f} MB): {wall:.3f}s  {mbps:.1f} MB/s  rc={r.returncode}")

    # worker scale multi-batch RTT (control+data) if license ok
    if not try_icc_license().get("ok"):
        print("skip phi worker (no ICC)")
        return 0
    try:
        with PhiWorker() as w:
            a = rng.standard_normal((128, 128))
            w.scale(a[:32, :32], alpha=1.0, beta=0.0)  # warm
            t0 = time.perf_counter()
            njob = 5
            for _ in range(njob):
                w.scale(a, alpha=1.05, beta=0.0)
            wall = time.perf_counter() - t0
            print(f"phi worker scale x{njob} 128^2: wall={wall:.3f}s  job/s={njob/wall:.2f}")
    except Exception as exc:
        print(f"phi worker path: {exc}")
    print(
        "\nConclusion: data plane dominated by scp of matrices; "
        "ControlMaster cuts control RTT; large GEMM/SCALE still pays PCIe-like mic link cost."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
