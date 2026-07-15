# Changelog

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
