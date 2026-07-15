# Terminal-facing examples

Plan: [`docs/plan/20260714_232700_terminal_examples_roadmap.md`](../docs/plan/20260714_232700_terminal_examples_roadmap.md)

These examples tell **application stories** (batch regression, sparse→dense fields),
not bare GEMM smoke tests. They share `_common.py` (env, devices, metrics, PowerCap).

## Quick run

```bash
source env/.venv/bin/activate
export PYTHONPATH=src
# optional VE:
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib

python examples/e1_batch_dense_regression.py
python examples/e2_sparse_then_dense.py

# CI / laptop without accelerators
python examples/e1_batch_dense_regression.py --host-only --m 128 --k 96 --n 64 --batches 2
python examples/e2_sparse_then_dense.py --host-only --nx 24 --ny 24 --nrhs 32 --n_out 48
```

Artifacts land under `artifacts/examples/<name>/` (`metrics.json`, `timeline.jsonl`, optional `plan.json`).

## Catalog

| ID | Script | Story | Key flags |
|----|--------|-------|-----------|
| **E1** | `e1_batch_dense_regression.py` | Column-normalize X, multi-batch multi-RHS \(C=X_{\mathrm{n}}B\) | `--batches --m --k --n --phi --host-only` |
| **E2** | `e2_sparse_then_dense.py` | 2D 5-point stencil SpMV → scale → dense GEMM | `--pattern stencil5\|random --nx --ny --host-only` |
| E3–E5 | *(planned X2)* | dataprep / DAG / sustained | see plan doc |

Legacy demos (path validation): `demo_atoms.py`, `demo_partition.py`, `demo_real_ve.py`, `demo_hetero_pipeline.py`.

## Metrics notes

- **`speedup_vs_host`**: pure Host numpy/OpenBLAS e2e. Mid-size jobs often **&lt; 1×** (PCIe vs DRAM BLAS) — still a valid pipeline demo.
- **E1 `speedup_vs_fixed_all_ve`**: auto plan vs forcing all VEs row-split (shows placement value when ≥2 VEs).
- Phi is **opt-in** (`--phi`); default is Host scale.

## Alignment with uni-framework

| uni | This repo example |
|-----|-------------------|
| hetero_dataprep / dense prep | E1 (and planned E3) |
| hetero_spmv | E2 |
| multi_task / throughput | planned E4 / E5 |
