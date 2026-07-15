# 新阶段规划：Phase 2 — 从「算子堆栈」到「异构运行时」

> 文档时间: 2026-07-14 22:32:06  
> 基线: **v0.6.0** (`3270f73`)  
> 前序计划: `docs/plan/20260714_220428_next_optimization_roadmap.md`（N+1～N+3 **已完成**）  
> 状态: **M1 已完成（v0.7.0）**；M2/M3 待实施

---

## 1. 阶段定位

### 1.1 我们已经有什么（v0.6 收口）

| 层 | 能力 |
|----|------|
| 代数 | tensor-layouts + 自定义 Host/Phi/VE atoms + 文档/glossary |
| Host | OpenBLAS / AVX-512 / auto 后端 |
| Phi | MKL DGEMM、IMCI、peak smoke、SCALE prep、常驻 worker（ssh 控制） |
| VE | NLC 文件路径、share_b、ve_exec worker pool、AVEO session/async/batch |
| 异构 | Phi→VE 流水线、multi-batch 重叠、PowerCap（uni） |
| 工程 | uv 隔离环境、device 测试、audit、时间戳 docs |

**结论**: 「单算子正确性 + 调度原型 + 数据面几种实现」已闭环。  
**下一阶段不再以再涨 10% DGEMM GFLOPS 为主目标**，而以 **系统化、产品化、可复用应用** 为主。

### 1.2 Phase 2 一句话目标

> 把 `uni_cute_tensor` 做成 **layout 驱动的异构运行时薄层**：稳定数据面、可组合任务图、可观测吞吐/功耗，并至少落地 1～2 个接近真实的异构应用（对接 uni-framework 调度语义）。

目标版本号建议：**v1.0.0**（API 相对稳定 + 文档齐全 + 性能表可复现）。

---

## 2. 当前剩余瓶颈（驱动本阶段）

| ID | 瓶颈 | 现象 | Phase 2 对应 |
|----|------|------|----------------|
| B1 | Phi 数据面仍 scp | hetero 重叠收益有限 | W1 数据面 / Phi 零拷贝或批量 |
| B2 | AVEO 队列多为串行 | dual-buf 提吞吐，真 DMA∥kernel 未证 | W1 AVEO 深度 |
| B3 | Layout 计划偏静态 | 仅有 row/col 粗代价 | W2 自动放置 + 反馈 |
| B4 | 应用偏 demo | scale+gemm 为主 | W3 真实 workload |
| B5 | 可观测性弱 | ipmitool 样本 0；无统一 dashboard | W4 观测 |
| B6 | 与 uni 调度耦合松 | PowerCap 已接，TaskGraph 未深接 | W3/W5 |
| B7 | 工程化 | 无 CI、README 版本叙事需统一 | W5 |

---

## 3. 工作流（Workstreams）

### W1 — 数据面 2.0（P0）

**目标**: 减少 Host↔加速器热路径上的文件/SSH 往返。

| 任务 | 描述 | 验收 |
|------|------|------|
| W1.1 | AVEO 驻留缓冲：session 内 pin 大缓冲，多次 GEMM 只更新脏区 | mid-size 多 batch wall ≥ dual-buf 再 +20% 或证明 PCIe 上限 |
| W1.2 | AVEO 异步流水：尝试 `async_write(i+1)` 与 `call(i)` 在实现允许范围内重叠；记录 timeline | 文档化是否可行；若不可行写清限制 |
| W1.3 | Phi 数据面：评估 mic 共享目录 / `micnativeloadex -a` / 长连接批量传数 | SCALE 控制+数据 round-trip 相对 v0.6 降 ≥30% |
| W1.4 | 统一 DataPlane 抽象：`transfer_h2d / launch / d2h` 接口，后端 file\|worker\|aveo | 切换后端不改应用代码 |

### W2 — Layout 运行时（P0/P1）

**目标**: CuTe 代数真正驱动放置与执行计划。

| 任务 | 描述 | 验收 |
|------|------|------|
| W2.1 | `PlacementPlan` 扩展：显式 stride、dtype、device affinity、estimated cost | 序列化 JSON 可复现 |
| W2.2 | 代价模型校准：用 `bench_*` 实测拟合 pcie_gbps / peak_gflops | 预测 wall 与实测相对误差有报告 |
| W2.3 | 自动策略：row/col/K-split + 设备子集选择（在 PowerCap 下） | 至少 1 个场景优于固定行切 |
| W2.4 | 与 atom 绑定：VE_NLC / PHI_MKL / HOST_OBLAS 作为 plan 的 backend 字段 | plan → runner 一键执行 |

### W3 — 应用与 uni 对齐（P0/P1）

**目标**: 从 demo 升到可对标 uni 示例的异构应用。

| 任务 | 描述 | 验收 |
|------|------|------|
| W3.1 | **异构 SpMV 或 dataprep 风格流水线**（不规则 Phi + 稠密 VE） | 正确性 + 相对纯 Host 的加速比 |
| W3.2 | 可选接入 uni `TaskGraph`（依赖边、并行、PowerCap） | 同一 DAG 可在 uni 语义下跑 |
| W3.3 | Multi-batch 服务模式：常驻 VE pool + Phi worker，HTTP/CLI 投递可选（默认 CLI） | 持续 jobs/s 报告 |

### W4 — 可观测性与性能工程（P1）

| 任务 | 描述 | 验收 |
|------|------|------|
| W4.1 | Timeline 日志：H2D / kernel / D2H / wait 统一 schema（JSONL） | 单 job 可画出阶段占比 |
| W4.2 | 功率：ipmitool / RAPL / VE sysfs 多源采样；无传感器时明确 degrade | `bench_sustained` 有 mean/max W 或明确 N/A |
| W4.3 | 统一性能门禁：`bench_summary` 输出 markdown 表写入 `docs/impl` | 每次发布可 diff |

### W5 — 产品化与 API 稳定（P1/P2）

| 任务 | 描述 | 验收 |
|------|------|------|
| W5.1 | 公共 API 冻结草案：`host_dgemm` / `run_phi_*` / `multi_ve_*` / `run_hetero_*` / `PowerCap` | `docs/architecture/*_api_v1.md` |
| W5.2 | CI：无加速卡 L0（pytest 无 device）；本机可选 device job 文档化 | GitHub Actions 或本地 script |
| W5.3 | 打包：`pyproject` 可选 extras `aveo`、`phi`、`dev`；版本 **1.0.0** | 安装说明一页纸 |
| W5.4 | 敏感信息与环境：继续禁止 license/序列号入库；`check_hw` 脱敏 | audit 通过 |

---

## 4. 迭代切片（建议 3 个里程碑）

### M1 — 数据面与观测（约 1 迭代） — **DONE v0.7.0**

1. ~~W1.1 驻留缓冲 + W1.4 DataPlane 抽象雏形~~  
2. ~~W4.1 timeline JSONL~~  
3. ~~`bench_summary` → 自动写 `docs/impl/<ts>_perf_gate.md`~~  
4. ~~不破现有正确性测试~~  

**出口**: multi-batch AVEO 与 file-pool 有 apples-to-apples 对比表；API 草案 v0.1。  
**实现纪要**: `docs/impl/20260714_223900_m1_dataplane_aveo_pinned.md`；gate `docs/impl/20260714_223845_perf_gate.md`。

### M2 — Layout + 真应用（约 1 迭代）

1. W2.1–W2.3 自动放置  
2. W3.1 一个完整异构应用（优先 SpMV 或 dataprep 流水线，复用 uni 经验）  
3. W3.2 TaskGraph 对接（若 uni 路径可用）  
4. PowerCap 贯穿应用路径  

**出口**: 应用端到端正确 + 相对 Host 有可解释加速；layout plan 可复现。

### M3 — v1.0 冻结（约 1 迭代）

1. W5 API 冻结 + extras  
2. CI L0 + 文档索引刷新  
3. 性能基线表固化进 README  
4. 明确 deprecations（如仅教学用 blocked Python GEMM）  

**出口**: tag **v1.0.0**。

---

## 5. 成功指标（Phase 2）

| 指标 | 目标 |
|------|------|
| 正确性 | 现有 device 测试不回退；新应用 max_abs_err 达标（按算法） |
| 数据面 | 至少一种后端在「多 batch 小中规模」上优于 v0.6 dual-buf 或证明 PCIe 饱和 |
| 应用 | ≥1 个非纯 GEMM 异构应用可一键跑 |
| 功耗 | 满载路径默认经 PowerCap；文档写明预算 1440W |
| 工程 | L0 CI 绿；公开 API 有文档；无敏感信息泄漏 |
| 版本 | v1.0.0 发布说明完整 |

---

## 6. 非目标（Phase 2 不做）

- 手写 IMCI/AVX-512 超过 MKL/NLC/OpenBLAS  
- GPU/CUDA 路径  
- 多机集群调度  
- 重写 uni-framework  

---

## 7. 风险与依赖

| 风险 | 缓解 |
|------|------|
| AVEO 无法真正并行 DMA/compute | 以实测为准；保留 worker 文件路径 |
| Phi 无共享 FS / scp 慢 | 批量、压缩、或减少往返次数 |
| uni API 变动 | PowerCap/TaskGraph 适配层隔离 |
| 电源 | 默认不 3VE+Phi 峰值同开 |

---

## 8. 文档与目录约定（延续）

- 规划：`docs/plan/YYYYMMDD_HHMMSS_*.md`  
- 实现：`docs/impl/YYYYMMDD_HHMMSS_*.md`  
- 架构变更：`docs/architecture/` 新时间戳，不覆盖旧文  
- 术语：`docs/glossary.md` 追加  
- 提交前：`scripts/audit_sensitive.sh`  

---

## 9. 需要确认的默认假设（已按推荐写入，可改）

| 项 | 默认 |
|----|------|
| 下一实施入口 | **M1（数据面 + 观测）** |
| 首个真应用 | **异构 dataprep 或 SpMV 风格**（对齐 uni 经验） |
| v1.0 是否强制 uni TaskGraph | **可选**（有则接，无则自研薄 DAG） |
| PowerCap | **默认开启**（可 env 关闭） |

---

## 10. 下一步行动（实施时）

1. 开 M1：DataPlane 抽象 + AVEO 驻留缓冲 PoC  
2. 同步写 `docs/impl/<ts>_m1_*`  
3. 每完成里程碑更新本文件进度表（追加「进度」节，不删历史）  

**当前状态**: **M1 完成（v0.7.0）**。下一入口 **M2**（Layout 自动放置 + 真应用）。

---

## 11. 进度日志

| 日期 | 里程碑 | 摘要 |
|------|--------|------|
| 2026-07-14 | 规划 | Phase 2 路线图建立 |
| 2026-07-14 | **M1** | DataPlane + AVEO pin + Timeline + perf_gate；pinned@512 multi-batch ~**1.25×** session；device 15 pass |
