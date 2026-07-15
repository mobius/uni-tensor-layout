# Terminal-facing examples

Plan: [`docs/plan/20260714_232700_terminal_examples_roadmap.md`](../docs/plan/20260714_232700_terminal_examples_roadmap.md)

Application stories (not bare GEMM smoke). Shared helpers: `_common.py`.

## Quick run

```bash
source env/.venv/bin/activate
export PYTHONPATH=src
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib

python examples/e1_batch_dense_regression.py
python examples/e2_sparse_then_dense.py
python examples/e3_dataprep_project.py
python examples/e4_job_dag.py
python examples/e5_sustained_jobs.py --jobs 12

# CI / no accelerators
python examples/e1_batch_dense_regression.py --host-only --m 128 --k 96 --n 64 --batches 2
python examples/e2_sparse_then_dense.py --host-only --nx 24 --ny 24
python examples/e3_dataprep_project.py --host-only --m 256 --k 128 --n-out 64
python examples/e4_job_dag.py --host-only --prefer-local --m 128 --k 96 --n 64
python examples/e5_sustained_jobs.py --host-only --jobs 8 --m 128
```

Artifacts: `artifacts/examples/<name>/` (`metrics.json`, optional `timeline.jsonl` / `plan.json`).

## Catalog

| ID | Script | Story | Key flags |
|----|--------|-------|-----------|
| **E1** | `e1_batch_dense_regression.py` | Column-normalize X, multi-batch multi-RHS \(C=X_n B\) | `--batches --m --k --n --phi` |
| **E2** | `e2_sparse_then_dense.py` | 2D 5-point stencil SpMV → scale → dense GEMM | `--pattern stencil5\|random --nx --ny` |
| **E3** | `e3_dataprep_project.py` | Dirty clean → standardize → project \(Y=XW\) | `--nan-frac --n-out --clip-sigma` |
| **E4** | `e4_job_dag.py` | prep → gemm_left ∥ gemm_right → reduce | `--prefer-local` forces local DAG |
| **E5** | `e5_sustained_jobs.py` | Resident session jobs/s (pin/pool) | `--mode aveo_pin\|pool --jobs\|--seconds` |

Legacy path demos: `demo_atoms.py`, `demo_partition.py`, `demo_real_ve.py`, `demo_hetero_pipeline.py`.

## Metrics notes

- **`speedup_vs_host`**: Host numpy/OpenBLAS e2e. Mid-size often **&lt; 1×** (PCIe vs DRAM) — expected.
- **E1 `speedup_vs_fixed_all_ve`**: auto plan vs forcing all VEs.
- **E5 `speedup_vs_cold_oneshot`**: resident thr vs cold `ve_exec` oneshot.
- **Phi** is opt-in (`--phi`).

## Alignment with uni-framework

| uni | This repo |
|-----|-----------|
| hetero_dataprep | E1, **E3** |
| hetero_spmv | E2 |
| multi_task | **E4** |
| throughput | **E5** |
