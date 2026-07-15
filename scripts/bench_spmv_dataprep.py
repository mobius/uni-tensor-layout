#!/usr/bin/env python3
"""Run SpMV + dataprep + auto-placed dense GEMM; optional TaskGraph path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from uni_cute_tensor.apps.spmv_dataprep import (
    random_csr,
    run_spmv_dataprep_multibatch,
    run_with_timeline,
)
from uni_cute_tensor.bridge.task_graph_bridge import run_spmv_gemm_task_graph
from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
from uni_cute_tensor.power import PowerCap


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
    print(f"VE={ve} PowerCap limit={PowerCap().effective_limit:.0f}W backend={PowerCap().backend}")
    rng = np.random.default_rng(0)

    nrows, ncols, k_rhs, n_out = 1024, 768, 768, 768
    density = 0.02
    csr = random_csr(nrows, ncols, density=density, rng=rng)
    print(f"CSR {nrows}x{ncols} nnz={csr.nnz} density~{csr.nnz/(nrows*ncols):.4f}")
    print(f"dense GEMM Y({nrows}x{k_rhs}) @ B({k_rhs}x{n_out})")

    # single-batch correctness + timeline
    x = rng.standard_normal((ncols, k_rhs))
    b = rng.standard_normal((k_rhs, n_out))
    res, tl = run_with_timeline(
        csr,
        x,
        b,
        devices=ve,
        use_phi_prep=False,
        compare_host=True,
        job_id="spmv-dp",
    )
    tl_path = ROOT / "artifacts" / "timeline_spmv_dataprep.jsonl"
    tl_path.parent.mkdir(parents=True, exist_ok=True)
    tl.write_jsonl(tl_path)
    print(
        f"single status={res.status} wall={res.wall_sec:.4f}s "
        f"spmv={res.spmv_sec:.4f} gemm={res.gemm_sec:.4f} err={res.max_abs_err:.2e}"
    )
    print(f"  host_wall={res.host_wall_sec:.4f}s speedup={res.speedup_vs_host:.2f}x")
    print(f"  notes={res.notes}")
    if res.choice:
        print(
            f"  plan={res.choice.strategy} devs={res.choice.devices} "
            f"est={res.choice.est_total_sec:.4f}s"
        )
        res.plan.write_json(ROOT / "artifacts" / "plan_spmv_dataprep.json")
    print(f"  timeline phases={tl.summary()} -> {tl_path.name}")
    if res.status != "pass":
        return 1

    # multi-batch with shared pool + SpMV∥GEMM overlap (amortized)
    nbatch = 6
    xs = [rng.standard_normal((ncols, k_rhs)) for _ in range(nbatch)]
    bs = [rng.standard_normal((k_rhs, n_out)) for _ in range(nbatch)]
    mb = run_spmv_dataprep_multibatch(csr, xs, bs, devices=ve, overlap=True)
    print(
        f"multibatch status={mb.status} wall={mb.wall_sec:.4f}s host={mb.host_wall_sec:.4f}s "
        f"speedup={mb.speedup_vs_host:.2f}x thr={mb.throughput_batches_per_sec:.2f} b/s "
        f"err={mb.max_abs_err:.2e}"
    )
    print(f"  notes={mb.notes}")
    if mb.status != "pass":
        return 1

    # TaskGraph path (uni or local)
    from uni_cute_tensor.apps.spmv_dataprep import csr_spmv

    def y_builder():
        y = csr_spmv(csr, x)
        return y if y.ndim == 2 else y[:, None]

    g = run_spmv_gemm_task_graph(
        y_builder, b, ve or ["host"], use_phi_prep=False, prefer_uni=True
    )
    print(f"task_graph backend={g.backend} status={g.status} wall={g.wall_sec:.4f}s notes={g.notes}")
    for name, tr in g.results.items():
        print(f"  node {name}: {tr.status} {tr.payload}")
    return 0 if g.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
