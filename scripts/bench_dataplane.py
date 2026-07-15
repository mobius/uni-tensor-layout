#!/usr/bin/env python3
"""Compare DataPlane backends: host / file / worker / aveo / aveo_pinned."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.runtime.dataplane import GemmRequest, create_dataplane
from uni_cute_tensor.runtime.timeline import timeline_scope


def main() -> int:
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )
    ve = ve_device_names(discover_devices())
    print(f"VE={ve}")
    rng = np.random.default_rng(0)
    n = 512
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    backends = ["host"]
    if ve:
        backends += ["file", "worker", "aveo", "aveo_pinned"]

    print(f"{'backend':<14} {'wall':>10} {'kgflops':>10} {'err':>12} {'status':>8}  timeline")
    for be in backends:
        try:
            if be == "aveo_pinned":
                plane = create_dataplane(
                    "aveo_pinned", devices=ve[:1], pin_m=n, pin_n=n, pin_k=n
                )
            elif be == "host":
                plane = create_dataplane("host")
            else:
                plane = create_dataplane(be, devices=ve)
            with timeline_scope(job_id=f"dp-{be}") as tl:
                with plane:
                    plane.gemm(GemmRequest(a=a[:64, :64], b=b[:64, :64]))
                    t0 = time.perf_counter()
                    r = plane.gemm(GemmRequest(a=a, b=b))
                    wall = time.perf_counter() - t0
            out = ROOT / "artifacts" / f"timeline_{be}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            tl.write_jsonl(out)
            phases = tl.summary()
            print(
                f"{be:<14} {wall:10.4f} {r.kernel_gflops:10.1f} "
                f"{r.max_abs_err:12.3e} {r.status:>8}  {phases} -> {out.name}"
            )
            if r.status != "pass":
                return 1
        except Exception as exc:
            print(f"{be:<14} FAIL {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
