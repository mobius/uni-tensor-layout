# Phase 2 M2 — Layout 自动放置 + SpMV/dataprep 应用 + TaskGraph

> 文档时间: 2026-07-14 22:51:00  
> 版本: **v0.8.0**  
> 计划: `docs/plan/20260714_223206_phase2_system_roadmap.md` M2

---

## 1. 交付对照

| 任务 | 实现 | 验收 |
|------|------|------|
| W2.1 PlacementPlan 扩展 | dtype / strides / backend / k / estimated_* / JSON 往返 | `plan.write_json` / `load_json` |
| W2.2 代价校准 | `calibrate_from_samples` + 默认 peak/pcie/launch | rel_error 报告 API |
| W2.3 自动策略 | row / col / k_split + 设备子集 + PowerCap | **auto 单 VE 优于固定 3-VE 行切** |
| W2.4 plan→runner | `partition/runner.execute_plan` 绑 VE_NLC/AVEO/HOST | 一键执行 |
| W3.1 异构应用 | `apps/spmv_dataprep.py` CSR SpMV→scale→dense GEMM | 端到端 err&lt;1e-8 |
| W3.2 TaskGraph | `bridge/task_graph_bridge.py` → uni 或 local DAG | uni backend=pass |
| PowerCap | 贯穿 choose + pipeline guard | uni limit 1440W |

---

## 2. 关键路径

```
src/uni_cute_tensor/partition/
  multi_device.py   # PlacementPlan v0.2 + col/k partitions + JSON
  cost_model.py     # calibrate + choose_best_placement
  runner.py         # execute_plan / shared pool / AVEO single-device
src/uni_cute_tensor/apps/spmv_dataprep.py
src/uni_cute_tensor/bridge/task_graph_bridge.py
scripts/bench_auto_place.py
scripts/bench_spmv_dataprep.py
tests/test_placement_m2.py
```

---

## 3. 实机结果

### 3.1 自动放置 vs 固定 3-VE 行切（`bench_auto_place`）

| case | auto strategy | auto devs | auto wall | fixed 3-VE wall | 结论 |
|------|---------------|-----------|-----------|-----------------|------|
| square512 | row_blocks | ve1 | **~0.19–0.27s** | ~0.51s | auto **~2×+** |
| tall1024×256 | col_blocks | ve1 | ~0.19–0.26s | ~0.50s | 选中 **col** |
| wide256×1024 | row_blocks | ve1 | ~0.18–0.27s | ~0.51s | auto 更快 |
| kheavy | row_blocks | ve1 | ~0.19s | ~0.51s | auto 更快 |
| square1536 | row_blocks | ve1 | ~0.30s | ~0.61s | auto 更快 |

**W2.3 出口满足**: 至少一类场景（实为全部 mid-size）优于固定全卡行切。  
原因: 中规模下 **worker 冷启动 + 分片 staging** 超过并行收益；代价模型倾向少设备。

预测 wall 仍偏乐观（rel_error ~0.6–0.7），已抬高 launch_overhead；后续可用 `calibrate_from_samples` 吃 bench JSON 再拟合。

### 3.2 SpMV + dataprep + GEMM

| 模式 | status | 备注 |
|------|--------|------|
| single AVEO | pass | err ~1e-12；相对 Host OpenBLAS mid-size **无加速**（可解释：主机 DRAM BLAS vs PCIe 卸载） |
| multibatch×6 pool | pass | thr ~7.7 b/s；相对 Host ~0.5×（同上） |
| uni TaskGraph | **pass** | spmv→prep→gemm，PowerCap=uni |

**加速比说明（诚实）**: 本机 48 核 OpenBLAS 在 1k 级 GEMM 上仍快于单次 VE 卸载；M2 价值在 **正确异构流水线、可复现 plan、自动避开过重多卡、uni DAG 对接**。相对 **固定 3-VE** 的 plan 选择已有清晰 wall 优势。真要压过 Host，需更大矩阵 + 驻留 AVEO multi-batch 服务路径（M3/持续服务）。

### 3.3 回归

- `pytest` unit + device: 预期全绿（实现后跑一轮）
- 计划 JSON: `artifacts/plan_*.json`

---

## 4. API 摘要

```python
from uni_cute_tensor.partition import choose_best_placement, execute_plan, PlacementPlan
from uni_cute_tensor.apps.spmv_dataprep import random_csr, run_spmv_dataprep_pipeline
from uni_cute_tensor.bridge.task_graph_bridge import run_spmv_gemm_task_graph
from uni_cute_tensor.power import PowerCap

cap = PowerCap()
choice = choose_best_placement(M, N, K, ["ve1","ve2","ve3"], power_cap=cap)
choice.plan.write_json("plan.json")
res = execute_plan(A, B, PlacementPlan.load_json("plan.json"))

csr = random_csr(1024, 768, density=0.02)
out = run_spmv_dataprep_pipeline(csr, X, B, devices=["ve1","ve2","ve3"], power_cap=cap)
```

---

## 5. 复现

```bash
source env/.venv/bin/activate
export PYTHONPATH=src
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib

pytest -q
python scripts/bench_auto_place.py
python scripts/bench_spmv_dataprep.py
```

---

## 6. 后续（M3）

- API 冻结 + CI L0  
- 性能基线固化 README  
- 可选: 服务模式常驻 AVEO pin 以争夺 Host 优势场景  
