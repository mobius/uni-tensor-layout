# uni-tensor-layout (uni-cute-tensor)

CuTe-style **tensor layout algebra** and a thin heterogeneous runtime for the machine
described by [uni-framework](https://github.com/mobius/uni-framework):

| Component | Role |
|-----------|------|
| Host 2× Xeon Gold 6252 | Layout planning, AVX-512 atoms, OpenBLAS GEMM |
| Intel Xeon Phi 7120P | KNC / MKL / irregular prep (via uni) |
| 3× NEC VE 1.0 | Dense FP64 (NLC / AVEO) |

Built on [tensor-layouts](https://github.com/facebookresearch/tensor-layouts)
(pure Python CuTe algebra — **no GPU required**).

Repository: https://github.com/mobius/uni-tensor-layout

## Status

**v1.7.0** — Service hardening + paper baseline (T1–T3): ops scripts, queue wait, pin policy, external matrices, full sweep path.

- Service: [`docs/SERVICE.md`](docs/SERVICE.md) · `bash scripts/service_start.sh` / `service_stop.sh`
- Serve: `uct-serve` · submit: `python scripts/submit_loop.py --job jobs/service_dense_stream.json --n 20 --socket`
- Paper: `python scripts/paper_sweep.py --exp-id esc4000_baseline` → `artifacts/paper/<id>/` · plot: `paper_plot.py`
- External data: `python scripts/make_sample_matrices.py` · `jobs/dense_external.json` / `sparse_external.json`
- Phi: job `"phi": true` (default off; Host fallback)
- Docs: [`docs/INDEX.md`](docs/INDEX.md) · Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- **No GitHub CI**; local: `bash scripts/ci_all.sh` (or `ci_l0.sh` / `ci_device.sh`)

### When to use Host / VE / serve

| Situation | Prefer | Why |
|-----------|--------|-----|
| Single mid-size GEMM, one-shot | **Host OpenBLAS** | Often wins latency; no VE cold-start tax |
| Many batches, fixed A, shared session | **VE AVEO pin** (+ `uct-serve`) | Resident thr ≫ cold oneshot (often tens–hundreds×) |
| Daily multi-user / multi-job stream | **`uct-serve` + pin preload** | Session reuse; metrics include `queue_wait_sec` |
| Irregular prep then dense | **Host prep** (or `phi:true` opt-in) → pin GEMM | Phi only when available; Host fallback |
| Paper table / thr–batches | `paper_sweep` + fixed `--seed` | Reproducible thr / err / vs_oneshot (no power KPIs) |

Claim discipline: **do not** claim “VE always beats Host OpenBLAS”; claim **residency**, multi-batch thr, and auditable placement.

### Performance baseline (this machine class)

| Scope | Case | Result |
|-------|------|--------|
| host | OpenBLAS @2048³ | ~**500** GFLOPS |
| host | AVX-512 @512³ | ~**170** GFLOPS |
| phi | MKL @1024³ | ~**626** GFLOPS |
| ve | multi-VE pool @1024³ | wall **~0.034 s** |
| ve | AVEO pinned 16×512 vs session | **~258 vs ~206** batch/s (~**1.25×**) |
| ve | auto-place vs fixed 3-VE @512³ | **~0.2 s vs ~0.5 s** |
| dp | `aveo_pinned` @512³ | wall **~0.008 s** |
| app | SpMV→GEMM + uni TaskGraph | **pass** (err ~1e-12) |
| e5 / uct-run ve_win | resident pin vs cold oneshot | ~**70×** thr (256³×32 batch sample) |
| paper | full sweep `esc4000_baseline` | see `docs/impl/*_paper_baseline_esc4000.md` |

Regenerate: `python scripts/bench_summary.py` · paper: `python scripts/paper_sweep.py --exp-id esc4000_baseline --seed 0`.

Optional: set `INTEL_LICENSE_FILE=$HOME/parallel_studio.lic` for ICC/Phi (file is never committed).

## Install (one page)

```bash
# requires uv; isolated env
uv venv --python 3.13 env/.venv
source env/.venv/bin/activate
uv pip install -e ".[dev]"

# extras (optional markers; toolchains stay on the machine)
# uv pip install -e ".[aveo,phi,viz]"

# hardware probe (no serial numbers)
bash scripts/check_hw.sh
# or: uct-check-hw

# L0 / full local checks (no GitHub CI — runners have no Phi/VE)
bash scripts/ci_l0.sh
# bash scripts/ci_all.sh          # L0 + device smoke
# UCT_SKIP_DEVICE=1 bash scripts/ci_all.sh

# terminal-facing examples (E1–E5) — see examples/README.md
python examples/e1_batch_dense_regression.py
python examples/e2_sparse_then_dense.py
python examples/e3_dataprep_project.py
python examples/e4_job_dag.py
python examples/e5_sustained_jobs.py --jobs 12
# legacy path demos
python examples/demo_atoms.py
python examples/demo_partition.py
python examples/demo_real_ve.py

# unit tests (no device) / device tests
pytest -q -m "not device"
pytest -q -m device

# full real suite (Host + 3×VE + Phi)
python scripts/run_real_tests.py --m 512 --k 512 --n 512
```

Optional: set `UNI_ROOT` to a local [uni-framework](https://github.com/mobius/uni-framework)
checkout (defaults to `~/Work/uni` if present).

### Extras

| Extra | Meaning |
|-------|---------|
| `test` | pytest |
| `dev` | pytest + matplotlib + ruff |
| `viz` | matplotlib |
| `aveo` | marker for AVEO/VEO host stack (system `libveo` + ncc) |
| `phi` | marker for Phi/ICC path (license + mic tools) |
| `all` | dev stack |

## Public API (quick)

```python
import uni_cute_tensor as uct
import numpy as np

A, B = np.random.randn(128, 96), np.random.randn(96, 80)
C, r = uct.host_dgemm(A, B, backend="auto")

plan = uct.partition_matrix_rows(128, 80, ["ve1", "ve2"], k=96)
choice = uct.choose_best_placement(128, 80, 96, ["ve1", "ve2", "ve3"])
# on VE hardware:
# res = uct.execute_plan(A, B, choice.plan)

with uct.create_dataplane("host") as plane:
    out = plane.gemm(uct.GemmRequest(a=A, b=B))
```

Device backends: import from `uni_cute_tensor.backends.*` / `apps.*` (stable contracts in the API doc).

### Deprecations (1.x kept)

- `host_blocked_dgemm` / `backend="blocked"` — **teaching only**, not a perf baseline.
- Prefer OpenBLAS / NLC / MKL / AVEO pinned for performance.

## What works on this hardware class

| Feature | Support |
|---------|---------|
| tensor-layouts algebra | Yes (Python ≥3.10) |
| Custom Host / Phi / VE atoms | Yes |
| Multi-VE partition + plan runner | Yes |
| DataPlane + Timeline | Yes |
| SpMV dataprep + TaskGraph | Yes |
| NVIDIA / AMD / AMX real MMA | Educational only (no such HW) |

## Repo layout

```
src/uni_cute_tensor/
  atoms/ partition/ apps/ bridge/ backends/ runtime/
docs/                 # INDEX + research|plan|impl|architecture + SERVICE.md
scripts/ci_l0.sh ci_device.sh ci_all.sh
scripts/service_start.sh service_stop.sh
scripts/paper_sweep.py paper_plot.py make_sample_matrices.py
scripts/bench_*.py
jobs/                 # dense/sparse/service/external templates
```

## Security

Before push: `bash scripts/audit_sensitive.sh`.  
Do not commit device serials, keys, `.env`, or license files.

## License

MIT. Upstream `tensor-layouts` is MIT.
