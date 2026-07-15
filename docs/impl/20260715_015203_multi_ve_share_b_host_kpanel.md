# 实现记录 — multi-VE 共享 B + Host K-panel

> 文档时间: 2026-07-15 01:52:03

---

## 1. multi-VE：B 只写一次（split 模式）

### 变更

| 项 | 说明 |
|----|------|
| 内核 | `dgemm_rect.c` 支持 `a.bin b.bin out.bin` |
| `a.bin` | `int32 M,K` + `A[M*K]` |
| `b.bin` | `int32 K,N` + `B[K*N]`（三卡共用） |
| API | `multi_ve_layout_dgemm(..., share_b=True)` 默认开启 |
| 兼容 | 仍支持 combined：`input.bin out.bin` |

### 实测 wall（3×VE 并行）

| shape | combined wall | share_b wall | 说明 |
|-------|---------------|--------------|------|
| 768³ | 0.43 s / eff 2.1 GF | **0.21 s / eff 4.2 GF** | **~2×** wall 改善 |
| 1536×1024×1024 | 0.20 s | 0.21 s | I/O 已非主导 |
| 2048×1536×1536 | 0.20 s | 0.23 s | 卡内 kernel 主导 |

单卡 kernel 仍约 **1.5 TFLOPS** 级（NLC）。  
中等规模时减少重复写 B 明显；超大矩阵时 wall 主要受进程启动/FS 并发限制。

基准：`python scripts/bench_multi_ve.py`

## 2. Host：K-panel AVX-512

- `dgemm_avx512.c`：按 `HOST_DGEMM_BK`（默认 256）切 K，线程同步流过同一 B 面板  
- 小/中规模仍快于朴素 numpy 初始化路径；大矩阵仍由 OpenBLAS/MKL 类库领先  

`python scripts/bench_host_dgemm.py`

## 3. 回归

- pytest **22 passed**  
- real suite **failed=0**（VE `mode=split`，Phi MKL ~617 GFLOPS @ 1024）

## 4. 后续

- 驻留进程 / AVEO 减少 `ve_exec` 启动  
- Host 链接系统 OpenBLAS 做公平对照  
- 异构：Phi 小任务 + VE 主算流水线  
