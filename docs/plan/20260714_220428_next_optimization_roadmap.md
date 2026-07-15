# 下一步优化路线图（v0.3 之后）

> 文档时间: 2026-07-14 22:04:28  
> 基线版本: **0.3.0** (`b825dac`)  
> 状态: **N+1 已实施**（见 `docs/impl/20260715_021426_aveo_hetero_layout_host_auto.md`）  
> 更新: 2026-07-15 — AVEO / hetero overlap / layout cost / host auto

---

## 0. 当前基线（已完成，作为对照）

| 能力 | 水平 | 主要瓶颈 |
|------|------|----------|
| Phi DGEMM (MKL) | ~750 GFLOPS @ 1536 | 卡峰值与 PCIe/scp 进出 |
| Phi DGEMM (IMCI 手写) | ~100 GFLOPS | 算法/缓存，非调度 |
| multi-VE NLC kernel | ~1.5 TFLOPS/卡 | 接近 NLC 能力 |
| multi-VE wall | worker pool 后 4–10× | 仍有文件 I/O + 协议轮询 |
| Hetero Phi→VE | 正确性 OK | 冷启动、Phi scp、串行阶段 |
| Host | OpenBLAS 大矩阵胜 | 手写 AVX-512 仅中小规模 |

**判断**: 卡内稠密算子（NLC/MKL）已较“吃满”；下一阶段收益主要来自 **数据面（少拷贝/少启动）** 与 **应用级流水线**，而非再抠 10% 手写 FMA。

---

## 1. 优先级总览

| 优先级 | 主题 | 预期收益 | 难度 | 依赖 |
|--------|------|----------|------|------|
| **P0** | VE AVEO / 内存侧卸载，去掉（或大幅减少）文件 staging | wall 再 2–5×（中规模） | 高 | libveo, 稳定 API |
| **P0** | 异构流水线加深：重叠 Phi∥VE、多 batch 吞吐 | 端到端占用率 ↑ | 中 | worker pool 已有 |
| **P1** | Layout 驱动自动分片/选 tile（CuTe divide → 设备计划） | 可移植 + 少手工调参 | 中 | tensor-layouts |
| **P1** | 吞吐型基准 + 功耗封顶联调（接 uni PowerCap） | 可发表数字 / 安全 | 中 | uni power |
| **P2** | Host 生产路径显式走 OpenBLAS/MKL，手写仅作 atom 对照 | 大矩阵默认最优 | 低 | 已有 fair bench |
| **P2** | Phi 侧减少 scp（驻留 mic 进程 / 批量任务） | 小任务延迟 ↓ | 中 | 类似 VE worker |
| **P3** | VE-MPI 跨卡规约与 layout 映射 | 需通信的算法 | 高 | NEC MPI |
| **P3** | 文档/CI：无卡 L0 测试 + 本机 device 标记 | 工程化 | 低 | — |

---

## 2. P0 详细方案

### 2.1 VE：AVEO 数据面（替代文件为主路径）

**目标**: Host 分配缓冲 → `veo` 传到 VE → 调用 NLC dgemm → 回传 C；worker 内循环无 fopen 热路径。

**步骤**:

1. 调研/封装 `libveo`（`/opt/nec/ve/veos/lib64/libveo.so`）最小 PoC：alloc / write / call / read  
2. 将 `dgemm_worker` 改为 **符号导出**（`mk_veorun_static` / AVEO 风格）或保持 ve_exec 但共享内存映射  
3. 对比基准：`bench_ve_launch.py` 增加 `aveo` 列（wall、GB/s H2D/D2H、kernel GFLOPS）  
4. 失败回退：保留现有 split 文件 + worker 路径  

**验收**:

- 正确性 err &lt; 1e-12（FP64）  
- 1024³ multi-VE wall 相对 v0.3 pooled 再降 ≥30%（或证明 PCIe 已饱和）  
- 文档：`docs/impl/<ts>_aveo_dgemm.md`

**风险**: AVEO API 版本、权限、与 NLC OpenMP 线程共存。

### 2.2 异构流水线：重叠与多 batch

**目标**: 不只做「Phi scale → VE gemm」串行 demo，做成可测吞吐的流水线。

**阶段**:

| 阶段 | 内容 |
|------|------|
| A | 双缓冲：batch_i 在 VE 算时，batch_{i+1} 在 Phi 预处理 |
| B | 任务图：接 uni `TaskGraph` + 可选 PowerCap |
| C | 更有意义的 Phi 核：稀疏分块、不规则 gather、或归一化+白化（对齐 uni SpMV/dataprep） |

**验收**:

- `examples/demo_hetero_pipeline.py` 扩展为 multi-batch，报告 pipeline occupancy  
- 相对串行 end-to-end 吞吐 ≥1.3×（在合适 batch 大小下）  
- 文档：`docs/impl/<ts>_hetero_pipeline_overlap.md`

---

## 3. P1 详细方案

### 3.1 Layout 驱动自动放置

**现状**: `partition_matrix_rows` 均匀切行；atom 仅描述 shape_mnk。

**下一步**:

1. 用 `logical_divide` / `Tile` 从全局 `Layout` 生成 per-device view（含 stride）  
2. 按设备模型代价函数选切分维（行/列/K）：  
   `cost ≈ transfer_bytes / pcie_bw + flops / device_gflops`  
3. 输出可序列化 `PlacementPlan`（已有 `to_dict`）供调度  

**验收**: 至少 2 种切分策略自动选择，并在 1 个 bench 上优于固定行切（wall 或 transfer 指标）。

### 3.2 吞吐 + 功耗

- 新增 `scripts/bench_throughput_sustained.py`：固定时长多 job，报告 jobs/s、J/job（若可读传感器）  
- 可选接入 uni `PowerCap`：禁止 3VE+Phi 同时峰值  
- 记录：`docs/impl/<ts>_sustained_throughput.md`

---

## 4. P2 工程打磨

| 项 | 动作 |
|----|------|
| Host 默认后端 | `host_dgemm` 增加 `backend="openblas"\|"avx512"\|"auto"`，auto 大矩阵走 numpy |
| Phi 驻留 | 仿 `VeWorkerPool` 做 `PhiWorkerPool`（ssh 长连接 + 卡上 loop），减少 scp 二进制 |
| 基准统一 | 一张表：Host / Phi / 1VE / 3VE / hetero，写入 `docs/impl` + README 摘要 |
| README 状态行 | 与 0.3.0 能力对齐（当前 README Status 仍有旧版文案残留，P2 一并修） |

---

## 5. P3 探索项（不做强承诺）

1. **AVEO + multi-stream** 真重叠 H2D 与 compute  
2. **VE-MPI AllReduce** + layout 描述的分块梯度聚合  
3. **CuTe compose 链** 生成 swizzle/对齐约束检查（分析向，非必须提速）  
4. 容器化 CI：仅 L0 代数测试；device 测试本机 nightly  

---

## 6. 建议实施顺序（2–3 个迭代）

```text
迭代 N+1（数据面） ✅ 2026-07-15
  ├─ AVEO PoC（单 VE dgemm） ✅
  ├─ multi-VE AVEO + oneshot/pool/aveo 对比 ✅
  ├─ Hetero multi-batch 重叠（部分，Phi scp 仍限收益）✅
  ├─ Layout 代价模型 row/col ✅
  ├─ Host auto backend ✅
  └─ 文档 + bench ✅

迭代 N+2（应用面） ⏳ 下一步
  ├─ Hetero：Phi 常驻 / 更大 VE 算量以兑现重叠
  ├─ AVEO 异步 write/call/read 重叠 H2D∥compute
  └─ PowerCap 可选开关（接 uni）

迭代 N+3（产品化）
  ├─ 统一性能表 + README 刷新
  └─ Phi worker pool
```

---

## 7. 明确「不做什么」（避免范围膨胀）

- 不追求手写 IMCI/AVX-512 超过 MKL/OpenBLAS/NLC  
- 不重写 CUTLASS/CUDA 栈  
- 不把 license / 序列号写入仓库  
- 不默认满载 3VE+Phi（电源约束）  

---

## 8. 成功指标（汇总）

| 指标 | 目标（相对 v0.3） |
|------|-------------------|
| multi-VE 1024³ wall（含数据） | 再降 ≥30% 或证明 PCIe 上限 |
| hetero multi-batch 吞吐 | ≥1.3× 串行 |
| 正确性 | FP64 max_abs_err &lt; 1e-8（与现有一致） |
| 文档 | 每迭代 `docs/impl/<ts>_*.md` + 更新本 plan 进度表 |

---

## 9. 需要你拍板的选项

1. **下一迭代主攻**: AVEO 数据面（推荐） / Hetero 重叠 / Layout 自动放置  
2. **功耗**: 是否强制接 uni PowerCap  
3. **Host 默认**: auto→OpenBLAS 是否接受（手写仅 `--backend avx512`）  

确认后可按迭代 N+1 直接开工。
