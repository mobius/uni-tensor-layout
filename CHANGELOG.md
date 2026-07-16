# Changelog

## 1.4.0 — 2026-07-15

### Added (Phase 3 M3)

- Multi-source power sampling: RAPL + ipmitool + optional VE sensors (`power_sample.py`)
- AVEO overlap limits bench + architecture note (ordered VEO queue; pin/dual-buf wins)
- Phi SSH ControlMaster mux (`UCT_PHI_SSH_MUX`); `bench_phi_dataplane.py`
- Docs: `architecture/*_aveo_async_limits.md`, `*_phi_dataplane_limits.md`

## 1.3.0 — 2026-07-15

### Added (Phase 3 M2)

- Shared AVEO/pool sessions (`runtime/session.py`)
- `uct-run` job runner + `jobs/*.json` templates (dense_batch, sparse_dense, dataprep, ve_win)
- `scripts/timeline_report.py` for JSONL phase bars
- Steady-state pin thr vs cold oneshot in dense_batch metrics (~70× on sample)

## 1.2.0 — 2026-07-15

### Added (Phase 3 M1)

- `recommend_backend` / `uct-recommend` — host vs VE pin/pool dispatch policy
- `scripts/calibrate_cost_model.py` → `artifacts/calibration.json` (+ shipped default)
- `scripts/bench_breakeven.py` → break-even table under `docs/impl/*_breakeven.md`
- Auto-load calibration in `choose_best_placement` / recommend (`UCT_CALIBRATION`, `UCT_NO_CALIBRATION`)

## 1.1.0 — 2026-07-15

### Added

- Terminal examples **E1–E5** (`examples/`, shared `_common.py`)
- `apps/dataprep_clean.py` (dirty feature clean + projection matrix)
- `stencil5_csr` structured sparse operator
- Generic `bridge.task_graph_bridge.run_task_graph`
- Phase close note: `docs/impl/20260715_014700_phase_close_v1_1.md`

### Notes

- Public API remains the v1.0 freeze (`docs/architecture/20260714_230700_api_v1.md`)
- Examples are host-only capable for L0 CI

## 1.0.0 — 2026-07-14

- Phase 2 complete: DataPlane, AVEO pin, Timeline, auto PlacementPlan, SpMV app
- API freeze, CI L0, package extras, performance baseline in README
