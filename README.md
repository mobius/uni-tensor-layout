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

v0.1.1 — layout atoms, multi-device partitioner, **real multi-VE NLC DGEMM**, Phi peak smoke, host tile GEMM.

### Real hardware (this machine class)

| Test | Result (example) |
|------|------------------|
| Multi-VE layout DGEMM 1536×1024×1024 | max_abs_err ~4e-13; ~1.4 TFLOPS/card kernel |
| Phi peak FP64 | ~570 GFLOPS via prebuilt `.mic` |
| Phi layout DGEMM | k1om-gcc + scp/ssh; correctness vs numpy (ICC needs Comp-CL) |
| Host blocked tile=8 | correct vs numpy |

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
  partition/      # multi-device PlacementPlan
  bridge/         # uni-framework adapter
  backends/       # host reference GEMM
docs/
  research|plan|impl|architecture/   # timestamped process docs
  glossary.md
scripts/check_hw.sh
scripts/audit_sensitive.sh
```

## Documentation

Process docs use `YYYYMMDD_HHMMSS_*.md` under `docs/{research,plan,impl,architecture}/`.
Technical terms: [`docs/glossary.md`](docs/glossary.md).

## Security

Before push: `bash scripts/audit_sensitive.sh`.  
Do not commit device serials, keys, `.env`, or license files.

## License

MIT. Upstream `tensor-layouts` is MIT.
