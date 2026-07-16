# 项目方向：日常算力服务 + 论文实验（基于 v1.5 框架）

> 文档时间: 2026-07-16 01:11:00  
> 基线: **v1.5.0**（`uct-run` / `uct-serve` / recommend / DataPlane / E1–E5）  
> 用户约束（已确认）:  
> 1. **受众**: 日常算力服务 **+** 论文实验  
> 2. **指标**: **不包含功耗**（不做 joule/RAPL 主指标）  
> 3. **Phi**: **按需求进入**，默认可不启用，路径保留、可开关  
> 状态: **方向已定，待选迭代开工**

---

## 1. 一句话目标

> 在本机（Host + 可选 Phi + 3×VE）上提供 **可常驻的稠密/异构作业服务**，并同步产出 **可复现的实验工件**（plan / timeline / metrics），支撑组内日常投递与论文图表；**不比拼中规模单次 GEMM 压过 Host OpenBLAS**，而强调 **常驻 thr、流水线、计划可审计、Phi 可插拔**。

---

## 2. 产品形态（双受众共用一套栈）

```text
                    ┌─────────────────────────────────────┐
                    │  日常算力服务                          │
                    │  uct-serve（Unix socket）常驻          │
                    │  共享 AVEO pin / 可选 pool            │
                    │  客户端: uct-run --socket / 脚本       │
                    └──────────────────┬──────────────────┘
                                       │ 同一 job JSON 语义
                    ┌──────────────────▼──────────────────┐
                    │  论文实验层                            │
                    │  固定 seed / 扫参 / 输出 artifacts/    │
                    │  plan.json + timeline.jsonl + metrics │
                    │  break-even / thr 表可重跑             │
                    └─────────────────────────────────────┘
```

| 受众 | 日常怎么用 | 论文怎么用 |
|------|------------|------------|
| 服务 | `uct-serve` 开着，脚本投递 job | 同一 serve 跑 sweep，落 artifacts |
| 实验 | 少关心 plan | 强制写 plan + 对比表 + 复现脚本 |

**Phi 策略**

| 模式 | 行为 |
|------|------|
| 默认 | `use_phi=false`：prep/scale 在 **Host** |
| 按需 | job 字段 `"phi": true` 或 CLI `--phi`：有 mic+license 则 Phi，否则 **自动 fallback Host** 并记 note |
| 论文 | 同一 workload 跑 `phi=off` vs `phi=on` 两列 thr/wall（能跑则跑） |

---

## 3. 主指标（无功耗）

| 指标 | 用途 |
|------|------|
| **jobs/s 或 batches/s** | 服务吞吐（主 KPI） |
| **wall_sec / p50–p99**（可选） | 单 job 延迟 |
| **speedup vs cold oneshot** | 论证常驻价值（已有 ~几十× 量级） |
| **speedup vs host_dgemm** | 诚实对照（中规模可 &lt;1） |
| **max_abs_err** | 正确性门禁 |
| **PlacementPlan + recommend 决策** | 论文可审计 |

**明确不做主指标**: RAPL/瓦特/焦耳/PowerCap 曲线（PowerCap 仍可作安全默认，但不写论文主表）。

---

## 4. 推荐主线工作包（3 条，可并行）

### P1 — 日常服务 MVP（优先）

**目标**: 组内「开箱即投」的本机算力入口。

| 任务 | 说明 |
|------|------|
| P1.1 | `uct-serve` 默认配置文档：socket 路径、preload 尺寸、`--phi` 策略 |
| P1.2 | 服务型 job 模板：`service_dense_stream`（多 batch 投影/多 RHS）、`service_sparse_dense` |
| P1.3 | 客户端小脚本：`scripts/submit_loop.py` 连续投 N 个 job，汇总 thr |
| P1.4 | 健康：`--health` 展示 session 是否 pin、jobs_done |
| P1.5 | （可选）job 队列长度/串行等待时间写入 metrics |

**出口**: 文档「如何开服务 + 如何投递」；一键 `submit_loop` 出 thr 表。

### P2 — 论文实验包（与 P1 共用 job）

**目标**: 可复现、可画图、可写进 method/experiment。

| 任务 | 说明 |
|------|------|
| P2.1 | **实验矩阵脚本** `scripts/paper_sweep.py`：扫 N∈{…}、batches∈{…}、backend∈{host,pin,oneshot}、phi∈{0,1} |
| P2.2 | 统一输出 `artifacts/paper/<exp_id>/`：metrics.jsonl、summary.md、plan 样例 |
| P2.3 | 核心图表定义（无功耗）: (1) thr vs batches；(2) pin vs oneshot；(3) host vs pin；(4) 可选 phi on/off prep 时间 |
| P2.4 | 方法节素材：PlacementPlan JSON 示例 + recommend 决策逻辑简述 |
| P2.5 | 固定 seed + 环境探测摘要（CPU/VE 数/无序列号）写入 manifest |

**出口**: 一篇 short experiment note（`docs/impl/*_paper_baseline.md`）+ 可重跑 sweep。

### P3 — Phi 按需路径硬化（不强制）

**目标**: 需要时稳定；不需要时零成本。

| 任务 | 说明 |
|------|------|
| P3.1 | job schema 统一 `phi: bool`，所有 service/paper job 识别 |
| P3.2 | fallback 语义：无 mic / 无 license → host + `notes` 明确 |
| P3.3 | hetero 模板：`phi_prep_ve_gemm`（重叠可选），仅 paper/service 显式开启 |
| P3.4 | 短测：phi on 时 control+data 时间拆分（已有 dataplane bench 可挂） |

**出口**: 「Phi 开关说明」一节；默认路径完全不依赖 Phi。

---

## 5. 里程碑建议

### 迭代 S1（服务可用，~1 小迭代）

- P1.1–P1.4 + P3.1–P3.2  
- 文档：日常如何 `uct-serve` + 投递  
- 验收：连续 50 job host-only 与（有卡时）pin 路径不崩；health 正确  

### 迭代 S2（论文可复现）

- P2.1–P2.5  
- 出 thr 主表 + pin vs oneshot 主图  
- 验收：他人按 README 一条命令重跑 sweep  

### 迭代 S3（Phi 按需 + 流水线故事）

- P3.3–P3.4 + 一条 sparse→dense 或 dataprep 服务模板  
- 验收：phi=on/off 两列数据齐全或 off 时自动 skip 有记录  

版本建议：功能合入后标 **v1.6**（服务+实验包），API 仍 1.x 兼容。

---

## 6. 与现有代码的映射（少造轮子）

| 需要 | 已有 |
|------|------|
| 常驻 | `uct-serve` / `session.py` |
| 投递 | `uct-run --socket` |
| 决策 | `recommend_backend` |
| 稀+稠 | `jobs/sparse_dense.json`、`apps/spmv_dataprep` |
| dataprep | `jobs/dataprep.json` |
| Phi prep | `phi_prep` / `PhiWorker`，hetero_pipeline |
| 计划工件 | `PlacementPlan.write_json`、timeline |
| 扫参半成品 | `bench_breakeven`、`bench_aveo_pinned`、E5 |

**新建优先**: `submit_loop`、`paper_sweep`、服务/论文 README 页、job 字段 `phi` 统一——而不是新 kernel。

---

## 7. 论文可写的 claim（务实）

适合写：

1. 异构节点上 **驻留 offload 会话** 使多 batch 稠密 thr 相对 cold oneshot 提升一个数量级以上。  
2. **Layout/placement 计划可序列化**，同一 job 规范驱动 Host/VE（及可选 Phi）。  
3. **可插拔预处理设备**（Host 默认，Phi 按需）与稠密 VE 后端解耦。  
4. 中规模下 Host BLAS 仍强 → **调度策略**（recommend + 常驻）比单点 kernel 优化更关键。

不适合当主 claim：

- 「全面绝对性能碾压 Host OpenBLAS」  
- 「AVEO 实现 DMA 与 compute 真重叠」  

---

## 8. 非目标（本方向）

- 功耗/能效主表（用户明确不需要）  
- GitHub CI / 公网服务  
- GPU、新稠密库、手写超 NLC 内核  
- 默认强制 Phi  

---

## 9. 建议立即拍板的实施顺序

| 顺序 | 内容 | 理由 |
|------|------|------|
| **1** | S1 服务文档 + submit_loop + phi 开关 | 日常服务先可用 |
| **2** | S2 paper_sweep + 实验 note | 论文数据可复现 |
| **3** | S3 Phi 流水线故事 | 按需增强，不挡 1–2 |

回复 **go S1** / **go S1+S2** 即可按该顺序写代码。
