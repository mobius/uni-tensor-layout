# Phase 4 规划 — 硬化、可信决策、真占用场景

> 文档时间: 2026-07-15 22:56:00  
> 基线: **v1.4.0**（Phase 2 + 终端案例 + Phase 3 全完成；**无 GitHub CI**）  
> 前序:  
> - Phase 2 → v1.0/v1.1  
> - Phase 3 → v1.2–v1.4（`docs/plan/20260715_214200_phase3_product_runtime.md`）  
> 状态: **M2 已完成（v1.5.0 uct-serve）**；M1/M3 部分待实施  
> 建议版本: **v1.5 → v1.6**（保持 1.x 公共 API 兼容）

---

## 1. 阶段定位

### 1.1 已具备（到 v1.4）

| 能力 | 状态 |
|------|------|
| 布局 / 放置 / DataPlane / pin / 案例 E1–E5 | 有 |
| `recommend_backend` + break-even + 校准入口 | 有，**精度仍粗** |
| `uct-run` + 共享 session | 有 |
| AVEO/Phi **限制文档** | 有（不指望假重叠） |
| 功耗 RAPL 采样 | 有；ipmitool/VE 常 degrade |
| 工程 | **仅本机** `ci_l0.sh` + `pytest -m device` |

### 1.2 仍痛 / 未做

| ID | 问题 |
|----|------|
| Q1 | 代价/recommend **rel_error 高**，难当 SLA |
| Q2 | 中规模 **Host 仍常赢 thr**；VE 价值需「释放 Host / 多 batch 常驻 / 功耗」叙事 |
| Q3 | `uct-run` 是单次进程；无 **长驻 worker 收 job**（Phase 3 W3.4 跳过） |
| Q4 | 与 **uni TaskGraph** 深度编排仍偏桥接 demo |
| Q5 | 本地 device 回归未 **一键打包**（脚本分散） |
| Q6 | VE 功率传感器未标定；ipmitool 常空 |

### 1.3 一句话目标

> 把 v1.4 从「功能齐全的异构运行时」收成 **可日常使用的本机工具链**：决策更可信、常驻更省事、回归一键、叙事与硬件现实一致。

**不**开新算法栈、不接 GPU、不上 GitHub CI。

---

## 2. 工作流

### W1 — 决策可信度（P0）

| 任务 | 描述 | 验收 |
|------|------|------|
| W1.1 | 校准按 **模式** 分表：host / ve_pin / ve_pool / oneshot | 拟合 mean rel_error 有报告；目标较 v1.4 默认改善 |
| W1.2 | recommend 输出 **置信区间或 band**（fast/slow 估计） | CLI 可见；避免单点墙钟误导 |
| W1.3 | break-even 固化进 README「何时用 VE」一小节 | 与 `bench_breakeven` 表链接 |
| W1.4 | `uct-run` 默认尊重 recommend；`--force-ve` / `--force-host` 显式 | 行为测到 |

### W2 — 常驻使用体验（P0/P1）

| 任务 | 描述 | 验收 |
|------|------|------|
| W2.1 | **`uct-serve`**（可选）：Unix socket 或 stdin JSONL 收 job，复用 session | 本机多 job 不冷启动；**默认不监听公网** |
| W2.2 | session 健康检查 / 超时回收 | 文档 + 简单测试 |
| W2.3 | job 模板补 1–2 个「常驻有理」场景（小 N 多 batch、流水线） | metrics 含 vs_oneshot |

### W3 — 本机质量工程（P0）

| 任务 | 描述 | 验收 |
|------|------|------|
| W3.1 | **`scripts/ci_device.sh`**：device pytest + 短 bench smoke（可 skip） | 一键；失败码清晰 |
| W3.2 | **`scripts/ci_all.sh`** = audit + ci_l0 +（可选）ci_device | README 唯一入口 |
| W3.3 | 明确 **永不** 启用 GitHub Actions（无卡） | INDEX/README 保持 |
| W3.4 | 敏感 audit 进 ci_all | 已有则串联 |

### W4 — 观测与功耗（P1）

| 任务 | 描述 | 验收 |
|------|------|------|
| W4.1 | 标定本机 `UCT_VE_POWER_SENSORS`（若可测）写入 docs/env 示例（无秘密） | 或文档写「不可标定」 |
| W4.2 | RAPL 在 sustained / uct-run 可选 `--sample-power` | metrics 带 mean_w 或 N/A |
| W4.3 | timeline_report 接入 uct-run 输出目录 | 一键生成 phase 条 |

### W5 — uni 与应用（P1/P2）

| 任务 | 描述 | 验收 |
|------|------|------|
| W5.1 | 一个 **双作业 DAG** 模板：Phi prep job + VE gemm job 经 TaskGraph | 有/无 uni 均可 |
| W5.2 | 文档：与 uni apps 对照表更新到 v1.4 能力 | examples/README 或 architecture |
| W5.3 | （可选）读外部 MatrixMarket/CSR 路径，不入库大数据 | job 字段 `matrix_path` |

---

## 3. 里程碑

### M1 — 可信决策 + 本机 CI 入口（→ ~v1.5.0）

1. W1.1–W1.4 校准/recommend 改进 + uct-run 策略  
2. W3.1–W3.3 `ci_device.sh` / `ci_all.sh`  
3. README「何时 Host / VE」  

**出口**: 本机一条命令做 L0+device smoke；recommend 有 band 与更新 break-even。

### M2 — 常驻 serve（→ v1.5.0） — **DONE**

1. ~~W2.1–W2.2 Unix socket serve + health/ping/shutdown~~  
2. ~~W2.3 复用现有 jobs 模板~~  
3. W4.2–W4.3 功耗挂 serve metrics — 部分（health 含 session；功耗可选后续）  

**出口**: `uct-serve` + `uct-run --socket`。  
**实现**: `docs/impl/20260715_230500_phase4_m2_uct_serve.md`

### M3 — uni 编排与外部数据（→ ~v1.6）

1. W5.1–W5.2  
2. W4.1 VE 功率标定结论  
3. W5.3 可选外部矩阵  

**出口**: 异构 DAG 可复现；功耗叙事完整或明确 N/A。

---

## 4. 成功指标

| 指标 | 目标 |
|------|------|
| 决策 | pin/pool/host 分模式校准；文档不承诺单点墙钟 |
| 工程 | `ci_all.sh` 本机绿；**无 GH Actions** |
| 常驻 | serve 路径 thr 不低于现 uct-run pin 量级 |
| 正确性 | L0 + device 不回退 |
| API | 只增不删 v1.0 `__all__` 语义 |

---

## 5. 非目标

| 不做 | 原因 |
|------|------|
| GitHub / 云 CI 强上 device | 无卡；已否决 |
| GPU/CUDA | 无硬件 |
| 公网 HTTP 服务默认开 | 安全与范围 |
| 手写超 MKL/NLC 内核 | ROI 低 |
| 多机 MPI | 非本机定位 |
| 「全面比 Host 快」营销 | 与实测冲突 |

---

## 6. 默认假设

| 项 | 默认 |
|----|------|
| 首迭代 | **M1：校准硬化 + ci_all/ci_device**（低风险、立刻有用） |
| 服务 | Unix socket **优先于** HTTP |
| 版本 | 1.5 = 决策+本机 CI；1.6 = serve + uni |
| 实施环境 | 仅 ESC4000 本机 |

---

## 7. 与现状对照（一页纸）

| | v1.4 现在 | Phase 4 之后 |
|--|-----------|----------------|
| 决策 | 有 recommend，偏乐观 | 分模式校准 + band |
| 作业 | `uct-run` 单次 | 可选 `uct-serve` 长驻 |
| 回归 | ci_l0 + 手工 device | `ci_all.sh` 一键 |
| CI 托管 | 无 GH（正确） | 保持无 GH |
| 叙事 | 限制文档齐全 | 「何时上 VE」日常可用 |

---

## 8. 建议你怎么回

| 回复 | 含义 |
|------|------|
| **go** / **go M1** | 先做校准硬化 + 本机 ci_all/ci_device |
| **go M2** | 直接上 uct-serve（可与 M1 并行需说明） |
| 调整优先级 | 例如「只要 serve 不要校准」 |

本文件入库后等你确认再写代码。
