# uni-tensor-layout (uni-cute-tensor)

CuTe-style **tensor layout algebra** for the heterogeneous machine described by
[uni-framework](https://github.com/mobius/uni-framework):

| Component | Role |
|-----------|------|
| Host 2× Xeon Gold 6252 | Layout planning, AVX-512 atoms, reference GEMM |
| Intel Xeon Phi 7120P | KNC atoms / irregular work (via uni) |
| 3× NEC VE 1.0 | Dense FP64 (NLC tile atoms) |

Built on [tensor-layouts](https://github.com/facebookresearch/tensor-layouts)
(pure Python CuTe algebra — **no GPU required**).

Repository: https://github.com/mobius/uni-tensor-layout

## Status

**v0.8.0** — Phase 2 **M2**: auto **PlacementPlan** (row/col/k + PowerCap) + **SpMV→dataprep→GEMM** app + uni **TaskGraph** bridge.

### Real hardware (this machine class)

| Test | Result (example) |
|------|------------------|
| auto-place vs fixed 3-VE @512³ | **~0.19–0.27s vs ~0.51s** (auto picks 1 VE) |
| SpMV+GEMM pipeline | e2e **pass** (err ~1e-12); uni TaskGraph **pass** |
| multi-VE pool wall @1024³ | **~0.034 s** |
| AVEO pinned 16×512 vs session | **~258 vs ~206 batch/s** (~**1.25×**) |
| DataPlane `aveo_pinned` @512³ | wall **~0.008 s** (pass) |
| Phi MKL @1024 | ~**626 GFLOPS** |
| Host OpenBLAS @2048 | ~**500 GFLOPS** |
| Summary + gate | `python scripts/bench_summary.py` → `docs/impl/*_perf_gate.md` |
| Auto place / SpMV | `python scripts/bench_auto_place.py` / `bench_spmv_dataprep.py` |

Optional: set `INTEL_LICENSE_FILE=$HOME/parallel_studio.lic` for ICC probe (file is never committed).

## Quick start (uv, isolated env)

```bash
# requires uv; does not touch system site-packages
uv venv --python 3.13 env/.venv
source env/.venv/bin/activate
uv pip install -e ".[dev]"

# hardware probe (no serial numbers printed)
bash scripts/check_hw.sh
# or:
python -c "from uni_cute_tensor.cli import check_hw_main; check_hw_main()"

# demos
python examples/demo_atoms.py
python examples/demo_partition.py
python examples/demo_real_ve.py

# unit + device tests
pytest -q

# full real suite (Host + 3×VE + Phi)
python scripts/run_real_tests.py --m 512 --k 512 --n 512
```

Optional: set `UNI_ROOT` to a local [uni-framework](https://github.com/mobius/uni-framework)
checkout for richer device discovery (defaults to `~/Work/uni` if present).

## What works on this hardware class

| Feature | Support |
|---------|---------|
| tensor-layouts algebra + viz | Yes (Host Python ≥3.10) |
| Custom Host / Phi / VE atoms | Yes (planning layouts) |
| Multi-VE row partition + task specs | Yes |
| Host-simulated sharded DGEMM correctness | Yes |
| NVIDIA / AMD / AMX / Xe real MMA | Educational only (no such HW) |

## Layout of the repo

```
src/uni_cute_tensor/
  atoms/          # HOST_AVX512, PHI_KNC, VE_NLC atoms
  partition/      # PlacementPlan + cost model + plan runner
  apps/           # hetero + SpMV/dataprep pipelines
  bridge/         # uni adapter + TaskGraph bridge
  backends/       # host / Phi / VE / AVEO GEMM
  runtime/        # DataPlane + Timeline (Phase 2)
docs/
  research|plan|impl|architecture/   # timestamped process docs
  glossary.md
scripts/check_hw.sh
scripts/audit_sensitive.sh
scripts/bench_summary.py
scripts/bench_dataplane.py
```

## Documentation

Process docs use `YYYYMMDD_HHMMSS_*.md` under `docs/{research,plan,impl,architecture}/`.
Technical terms: [`docs/glossary.md`](docs/glossary.md).

## Security

Before push: `bash scripts/audit_sensitive.sh`.  
Do not commit device serials, keys, `.env`, or license files.

## License

MIT. Upstream `tensor-layouts` is MIT.
