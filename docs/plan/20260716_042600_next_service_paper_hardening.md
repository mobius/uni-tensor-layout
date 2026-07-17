# 下阶段计划 — 服务硬化 + 论文闭环（v1.6 之后）

> 文档时间: 2026-07-16 04:26:00  
> 基线: **v1.6.0**（日常服务骨架 + paper_sweep + Phi 按需 + uct-serve）  
> 用户约束延续:  
> - 受众 = **日常算力服务 + 论文实验**  
> - 指标 = thr / 延迟 / 正确性 / 可复现（**不含功耗**）  
> - Phi = **按需**，默认关  
> - **无 GitHub CI**  
> 状态: **T1–T3 已实施 → v1.7.0**（Phi 长驻 worker / 分模式校准 band 仍可选遗留）  
> 版本: **v1.7.0**

---

## 1. 现状（已有什么）

| 层 | 能力 |
|----|------|
| 服务 | `uct-serve` / `uct-run --socket` / `submit_loop` / `docs/SERVICE.md` |
| 作业 | dense / sparse_dense / dataprep / service_* / phi_prep_ve_gemm |
| 论文 | `paper_sweep.py` → manifest + metrics.jsonl + summary.md |
| 决策 | `recommend_backend` + break-even 文档（精度仍粗） |
| 工程 | 仅本机 `ci_l0.sh`；device 测仍分散 |

### 相对「日常好用 + 论文能写」仍缺

| ID | 缺口 |
|----|------|
| G1 | 服务：**预热 pin 与 job 尺寸不一致**时行为弱文档；无简单队列等待时间；无一键 `systemd`/screen 样例 |
| G2 | 论文：**真机全量 sweep 基线表未固化进 docs/impl**；无出图脚本（matplotlib 可选） |
| G3 | 决策：校准分模式 / recommend band（Phase 4 M1）仍未做 |
| G4 | 数据：job 只合成矩阵，**不能读外部 CSR/bin**（论文真实 workload） |
| G5 | Phi：worker 多 batch 在 serve 路径未稳定编排；prep 时间拆分未进 paper 主表 |
| G6 | 工程：`ci_device.sh` / `ci_all.sh` 一键本机回归未做 |

---

## 2. 一句话目标

> 把 v1.6 从「能开服、能扫参」收成 **组内默认可依赖的算力入口**，并产出 **一版可引用的本机实验基线**（thr/oneshot/host 对照 + 复现命令）；Phi 保持按需。

---

## 3. 工作流

### W1 — 服务硬化（日常，P0）

| 任务 | 验收 |
|------|------|
| W1.1 `scripts/service_start.sh` / `service_stop.sh` | preload、socket、日志路径固定 |
| W1.2 serve metrics：排队等待时间（拿锁前） | `queue_wait_sec` 写入 result |
| W1.3 job 与 preload 尺寸策略 | 超 pin 容量自动 re-pin 或拒绝并明确错误 |
| W1.4 `submit_loop --socket` 与 service 模板默认对齐 | SERVICE.md 更新「推荐日常命令」 |
| W1.5 （可选）stdin JSONL 多 job 批投 | 少开 socket 客户端时的批量入口 |

### W2 — 论文闭环（P0）

| 任务 | 验收 |
|------|------|
| W2.1 本机跑 **全量** `paper_sweep`（非 quick） | `docs/impl/<ts>_paper_baseline_esc4000.md` 固化表 |
| W2.2 `scripts/paper_plot.py`（可选 viz extra） | thr–batches、pin vs oneshot 两张图进 artifacts |
| W2.3 方法附录：job schema + serve 架构 1 页 | 链到 architecture / SERVICE |
| W2.4 固定 seed 清单 + 复现 checklist | 「第三人 30 分钟重跑」 |

### W3 — 外部数据 + Phi 按需加深（P1）

| 任务 | 验收 |
|------|------|
| W3.1 job 支持 `matrix_a` / `matrix_b` / `csr_path`（本地路径） | 不入库大文件；文档给生成小样例 |
| W3.2 sparse_dense 可读外部 CSR（简单二进制或 npz） | 论文可用真实 stencil/图一次 |
| W3.3 phi 路径：prep_sec 进 paper_sweep 主列 | phi=on/off 表完整或 skip 有记录 |
| W3.4 （可选）serve 内 PhiWorker 长驻 | 仅 `phi:true` 多 job 时复用 |

### W4 — 本机质量与决策（P1）

| 任务 | 验收 |
|------|------|
| W4.1 `scripts/ci_device.sh` + `ci_all.sh` | audit + L0 + 可选 device smoke |
| W4.2 recommend 分模式校准收紧（pin/host） | 文档更新；不承诺 SLA |
| W4.3 README「何时 Host / VE / serve」一小节 | 与 baseline 表互链 |

---

## 4. 里程碑

### T1 — 服务可依赖（→ ~v1.7.0）

1. W1.1–W1.4  
2. W4.1 `ci_all` / `ci_device`  
3. SERVICE.md 运维级说明  

**出口**: 开服/停服/投递三命令稳定；本机一键回归。

### T2 — 论文基线（→ ~v1.7.x）

1. W2.1–W2.4 真机 sweep + 固化 impl 笔记 + 可选图  
2. W4.3 决策叙事  

**出口**: 可引用的 thr 基线表 + 复现脚本。

### T3 — 数据与 Phi（→ ~v1.8）

1. W3.1–W3.3 外部矩阵/CSR  
2. W3.4 可选 Phi worker 常驻  
3. W4.2 校准改进  

**出口**: 至少一个「非合成」workload job + phi 对比表。

---

## 5. 成功指标（无功耗）

| 指标 | 目标 |
|------|------|
| 服务 | serve 连续 ≥100 job（合成）零崩溃；health 正确 |
| thr | pin 路径 vs oneshot 仍有数量级优势（与 v1.5/1.6 一致） |
| 论文 | 一页 baseline 表可复现；err 全 pass |
| 工程 | `ci_all.sh` 本机绿；**仍无 GH Actions** |
| Phi | 默认路径零依赖；on 时有数据或明确 skip |

---

## 6. 非目标

- 功耗/能效主表  
- GitHub/云 CI  
- 公网 HTTP  
- GPU、多机 MPI  
- 「全面压过 Host OpenBLAS」主 claim  

---

## 7. 与既有计划的关系

| 文档 | 关系 |
|------|------|
| Phase 4 hardening | T1 吸收其 **ci_device/ci_all + 校准**（M1 遗留） |
| use_case S1–S3 | **已完成**（v1.6）；本阶段是 **硬化与闭环** |
| Phase 3 | 能力底座；不再扩 scope |

---

## 8. 建议默认开工顺序

```text
T1 服务运维脚本 + ci_all/ci_device
T2 真机 paper_sweep 固化 baseline
T3 外部数据 + Phi 表
```

回复 **go T1** / **go T1+T2** / **go**（按 T1→T2→T3）开始实施。
