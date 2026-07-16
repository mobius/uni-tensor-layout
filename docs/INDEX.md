# Documentation index

> Updated for **v1.4.0**. Process docs keep timestamps; this file is the entry map.

## Start here

| Doc | Purpose |
|-----|---------|
| [README.md](../README.md) | Install, status, performance baseline |
| [../examples/README.md](../examples/README.md) | **Terminal examples E1–E5** |
| [glossary.md](glossary.md) | Terms |
| [architecture/20260714_230700_api_v1.md](architecture/20260714_230700_api_v1.md) | **Public API v1 freeze** |
| [architecture/20260714_013534_uni_cute_tensor_architecture.md](architecture/20260714_013534_uni_cute_tensor_architecture.md) | Original architecture draft |

## Plans

| Doc | Purpose |
|-----|---------|
| [plan/20260714_013534_implementation_plan.md](plan/20260714_013534_implementation_plan.md) | Phase 0–5 initial plan |
| [plan/20260714_220428_next_optimization_roadmap.md](plan/20260714_220428_next_optimization_roadmap.md) | N+1–N+3 (done) |
| [plan/20260714_223206_phase2_system_roadmap.md](plan/20260714_223206_phase2_system_roadmap.md) | Phase 2 M1–M3 → v1.0 |
| [plan/20260714_232700_terminal_examples_roadmap.md](plan/20260714_232700_terminal_examples_roadmap.md) | Terminal examples E1–E5 (done) |
| [plan/20260715_214200_phase3_product_runtime.md](plan/20260715_214200_phase3_product_runtime.md) | **Phase 3 complete** (v1.4) |

## Implementation notes (selected)

| Doc | Milestone |
|-----|-----------|
| [impl/20260714_223900_m1_dataplane_aveo_pinned.md](impl/20260714_223900_m1_dataplane_aveo_pinned.md) | M1 DataPlane / pin / timeline |
| [impl/20260714_223845_perf_gate.md](impl/20260714_223845_perf_gate.md) | Perf gate sample |
| [impl/20260714_225100_m2_layout_spmv_dataprep.md](impl/20260714_225100_m2_layout_spmv_dataprep.md) | M2 placement + SpMV |
| [impl/20260714_230800_m3_v1_freeze.md](impl/20260714_230800_m3_v1_freeze.md) | M3 v1.0 freeze |
| [impl/20260714_233200_examples_e1_e2.md](impl/20260714_233200_examples_e1_e2.md) | Terminal examples X1 (E1–E2) |
| [impl/20260715_010300_examples_e3_e5.md](impl/20260715_010300_examples_e3_e5.md) | Terminal examples X2 (E3–E5) |
| [impl/20260715_014700_phase_close_v1_1.md](impl/20260715_014700_phase_close_v1_1.md) | Phase close → v1.1.0 |
| [impl/20260715_215000_phase3_m1_dispatch.md](impl/20260715_215000_phase3_m1_dispatch.md) | Phase 3 M1 dispatch/calibration |
| [impl/20260715_214911_breakeven.md](impl/20260715_214911_breakeven.md) | Host vs VE break-even sample |
| [impl/20260715_220500_phase3_m2_job_runner.md](impl/20260715_220500_phase3_m2_job_runner.md) | Phase 3 M2 uct-run + sessions |
| [impl/20260715_220000_phase3_m3_hetero_power.md](impl/20260715_220000_phase3_m3_hetero_power.md) | Phase 3 M3 AVEO/Phi/power |
| [architecture/20260715_221500_aveo_async_limits.md](architecture/20260715_221500_aveo_async_limits.md) | AVEO async limits |
| [architecture/20260715_221500_phi_dataplane_limits.md](architecture/20260715_221500_phi_dataplane_limits.md) | Phi data plane limits |

## Research

| Doc | Purpose |
|-----|---------|
| [research/20260714_013534_hw_env_feasibility.md](research/20260714_013534_hw_env_feasibility.md) | HW feasibility |
| [research/20260714_013534_tensor_layouts_and_uni_survey.md](research/20260714_013534_tensor_layouts_and_uni_survey.md) | tensor-layouts / uni survey |
| [research/20260714_080104_intel_license_doc_review.md](research/20260714_080104_intel_license_doc_review.md) | ICC license notes (no secrets) |

## Local checks / device jobs

GitHub Actions is **not** used: public runners have no Phi/VE (and L0-only CI adds little
for this hardware-bound repo). Run on the ESC4000 (or any host) instead:

```bash
# L0 (no accelerator required)
bash scripts/ci_l0.sh

# Device (this machine class)
export PYTHONPATH=src
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib
pytest -q -m device
python scripts/bench_summary.py
```
