# Phase 3 规划 — 从「可演示运行时」到「可决策、可服务」

> 文档时间: 2026-07-15 21:42:00  
> 基线: **v1.1.0**（Phase 2 + 终端案例 E1–E5 已关闭）  
> 前序:  
> - `docs/plan/20260714_223206_phase2_system_roadmap.md`（**完成**）  
> - `docs/plan/20260714_232700_terminal_examples_roadmap.md`（**完成**）  
> - 收尾: `docs/impl/20260715_014700_phase_close_v1_1.md`  
> 状态: **规划中（未实施）**  
> 建议目标版本: **v1.2 → v1.3**（保持 1.x API 兼容）

---

## 1. 阶段定位

### 1.1 我们已经有什么（v1.1）

| 层 | 能力 |
|----|------|
| 代数 / 放置 | PlacementPlan JSON、row/col/k、auto + PowerCap |
| 数据面 | DataPlane、AVEO pin、worker pool、file |
| 观测 | Timeline JSONL、perf_gate、E5 thr |
| 应用 | SpMV/dataprep 库 + **E1–E5 终端案例** |
| 工程 | API v1 冻结、CI L0、audit、docs 索引 |

### 1.2 仍痛的点（驱动 Phase 3）

| ID | 现象 | 含义 |
|----|------|------|
| P1 | 中规模 **Host OpenBLAS 常赢** 单次 VE 卸载 | 缺「何时上加速器」的决策层与大作业/多 batch 场景 |
| P2 | 代价模型 **预测偏乐观**（rel_error 高） | auto place 方向对，但不可当精确 SLA |
| P3 | Phi 热路径仍 **scp 重** | 异构重叠收益上限 |
| P4 | AVEO **真 DMA∥kernel** 未系统验证 | dual-buf/pin 有 thr，异步重叠文档不足 |
| P5 | Examples 是 CLI，不是 **常驻作业入口** | E5 接近服务，但无统一 job runner / 配置 |
| P6 | 功率采样 **常 N/A** | 功耗叙事弱于算力叙事 |

### 1.3 一句话目标

> 让 `uni_cute_tensor` 能 **回答「这个 job 该不该上 VE/Phi」**，在 **该上的场景** 给出可复现 thr/功耗，并提供 **常驻会话作业入口**（仍默认 CLI，可选轻量服务）。

**不**以再抠 NLC/MKL 峰值 GFLOPS 为主目标。

---

## 2. 工作流（Workstreams）

### W1 — 决策与代价（P0）

| 任务 | 描述 | 验收 |
|------|------|------|
| W1.1 | **Dispatch policy**：`recommend_backend(m,n,k,batch,…) → host\|ve\|hetero` + 理由 | 单元测试 + example 打印 policy |
| W1.2 | 代价模型 **校准流水线**：`scripts/calibrate_cost_model.py` 吃 bench 输出，写 `artifacts/calibration.json` | 报告 mean/max rel_error；较 v1.1 默认有改进文档 |
| W1.3 | **Break-even 曲线**：Host vs AVEO-pin vs pool，扫 N 与 batch | 表写入 `docs/impl/*_breakeven.md`；README 链一节 |
| W1.4 | auto place 使用校准参数（可 env 关闭） | 默认加载 calibration 若存在 |

### W2 — 数据面与 Phi（P0/P1）

| 任务 | 描述 | 验收 |
|------|------|------|
| W2.1 | AVEO：文档化 + 尝试 **write∥call** 重叠（若 API 允许） | 可行则 bench；不可行则限制说明进 architecture |
| W2.2 | 共享 **常驻 Session 句柄**（跨 example/E5/runner） | 同一进程多 job 不重复 open/close |
| W2.3 | Phi：批量 scp / 长连接 worker **数据+控制** 统一；评估 mic 路径 | SCALE/prep RTT 相对 v1.1 降 ≥20% 或证明饱和 |
| W2.4 | DataPlane 补 `phi` 后端（可选） | create_dataplane("phi") 或明确不做 |

### W3 — 作业入口与案例深化（P0/P1）

| 任务 | 描述 | 验收 |
|------|------|------|
| W3.1 | **`uct-run` / `python -m uni_cute_tensor.run`**：读 job YAML/JSON → plan → 执行 → metrics | 3 个内置 job 模板（dense_batch / sparse_dense / dataprep） |
| W3.2 | E1–E5 可被 job runner 调用（不重复实现） | 模板 → 同一 metrics schema |
| W3.3 | 大矩阵 / 多 batch 场景：展示 **VE 相对 Host 有 thr 或占用优势** 的至少 1 例 | 文档写清尺寸与条件 |
| W3.4 | （可选）Unix socket / 本地 HTTP 投递，默认关 | 仅当 W3.1 稳定后 |

### W4 — 观测与功耗（P1）

| 任务 | 描述 | 验收 |
|------|------|------|
| W4.1 | 功率：RAPL + VE sysfs + ipmitool 分层；统一 `PowerSample` | N/A 时明确 degrade 原因 |
| W4.2 | Timeline → 简单 **阶段占比 markdown/ASCII** | `scripts/timeline_report.py` |
| W4.3 | perf_gate 纳入 policy + break-even 摘要行 | 发布可 diff |

### W5 — 工程（P1/P2）

| 任务 | 描述 | 验收 |
|------|------|------|
| W5.1 | CI：L0 保持；可选 `workflow_dispatch` device 文档 | 不强制 GH 有卡 |
| W5.2 | API 文档增量：`recommend_*` / job schema（仍 1.x 兼容） | architecture 新时间戳文 |
| W5.3 | 可选 `uv.lock` 入库策略二选一（锁 or gitignore） | 团队约定写 README |
| W5.4 | 敏感信息与 license 策略不变 | audit 绿 |

---

## 3. 里程碑

### M1 — 决策层（→ 约 v1.2.0）

1. W1.1–W1.3 break-even + recommend_backend  
2. W1.2 校准脚本 + 默认模型更新  
3. E5 / bench 接入校准参数  
4. 文档：何时用 Host / VE  

**出口**: 用户跑一个命令能看到「推荐后端 + 依据」；至少一张 break-even 表。

### M2 — 常驻作业入口（→ 约 v1.2.x / 1.3.0）

1. W2.2 共享 session  
2. W3.1–W3.2 job runner + 3 模板  
3. W3.3 一个「Host 不占优」的可复现场景  
4. W4.2 timeline report  

**出口**: `uct-run jobs/dense_batch.yaml` 一键；metrics schema 统一。

### M3 — 异构深化（→ 约 v1.3.0）

1. W2.1 AVEO 重叠结论  
2. W2.3 Phi 数据面改进或上限证明  
3. W4.1 功率多源  
4. （可选）W3.4 本地投递  

**出口**: 异构路径有「限制说明 + 最佳实践」；功耗有数或明确 N/A。

---

## 4. 成功指标（Phase 3）

| 指标 | 目标 |
|------|------|
| 决策 | recommend 与实测一致率有报告（或诚实的不确定区间） |
| 校准 | 标定后 mean rel_error 较 v1.1 默认 **下降**（具体数字以实测为准，目标 &lt; 40%） |
| Break-even | 文档给出 N、batch 阈值的 Host vs VE 切换建议 |
| 常驻 | pin/pool 路径 thr 相对 oneshot 保持/扩大（对齐 E5 量级） |
| 作业入口 | ≥3 模板 + host-only CI 冒烟 |
| 兼容 | 不破坏 v1.0 公共 `__all__` 语义 |
| 正确性 | L0 + device 不回退 |

---

## 5. 非目标（Phase 3 不做）

- GPU/CUDA  
- 手写超 MKL/NLC 的 ISA 内核  
- 多机 MPI 集群  
- 重写 uni-framework  
- 强行宣传「全面相对 Host 加速」  
- 完整 Web UI  

---

## 6. 风险

| 风险 | 缓解 |
|------|------|
| 本机 Host 太强，VE 几乎无 wall 优势 | 指标改 thr/占用/功耗；找大 N 与多 batch |
| AVEO 无法 DMA∥compute | 文档关闭该优化项 |
| Phi 硬件/license 不稳 | 默认 host prep；Phi 为 optional path |
| Job runner 范围膨胀 | 只包现有 apps/examples，不新算法 |

---

## 7. 默认假设（可改）

| 项 | 默认 |
|----|------|
| 首迭代 | **M1 决策 + break-even + 校准** |
| 服务形态 | **CLI job runner 优先**，HTTP 可选且后置 |
| 版本 | 1.2 = 决策/校准；1.3 = runner + 异构深化 |
| API | 只 **增加** 符号，不删 v1.0 冻结集 |

---

## 8. 与 uni 的关系

| uni | Phase 3 对齐 |
|-----|----------------|
| PowerCap / TaskGraph | 继续桥接；job DAG 模板可映射 uni 语义 |
| hetero_spmv / dataprep | E2/E3 已叙事对齐；深化在 **决策与常驻**，不重复造 C 内核 |
| throughput 示例 | E5 + job runner 服务模式 |

**差异化**: layout plan 可序列化 + **dispatch policy** + 本仓库 DataPlane，而不是第二套 uni apps。

---

## 9. 建议实施顺序（你确认后）

```text
1) M1: calibrate + break-even + recommend_backend
2) M2: session reuse + uct-run job templates  
3) M3: AVEO/Phi 结论 + power sampling
```

回复 **go** / **go M1** 开始实施；若只要计划入库，提交本文件即可。

---

## 10. 一页纸：Phase 2 vs Phase 3

| | Phase 2（已完成） | Phase 3（本计划） |
|--|-------------------|-------------------|
| 关键词 | 能跑、能演示、API 稳 | 该不该跑、怎么常驻、可校准 |
| 出口 | v1.0 / v1.1 + E1–E5 | v1.2–1.3 + policy + job runner |
| 主指标 | 正确性、pin thr、放置不踩坑 | break-even、recommend 命中、jobs/s、功耗 |
