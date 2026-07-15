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

**v1.0.0** — public API frozen. Phase 2 complete (DataPlane, auto PlacementPlan, SpMV app, CI L0).

- API: [`docs/architecture/20260714_230700_api_v1.md`](docs/architecture/20260714_230700_api_v1.md)
- Docs map: [`docs/INDEX.md`](docs/INDEX.md)
- Glossary: [`docs/glossary.md`](docs/glossary.md)

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

Regenerate table: `python scripts/bench_summary.py` → `docs/impl/*_perf_gate.md`.

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

# L0 CI locally (same as GitHub Actions)
bash scripts/ci_l0.sh

# demos
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
docs/                 # INDEX + research|plan|impl|architecture
scripts/ci_l0.sh      # local L0
scripts/bench_*.py
.github/workflows/ci.yml
```

## Security

Before push: `bash scripts/audit_sensitive.sh`.  
Do not commit device serials, keys, `.env`, or license files.

## License

MIT. Upstream `tensor-layouts` is MIT.
