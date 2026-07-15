# 实现记录 — N+2：AVEO 分阶段、Phi worker、PowerCap、统一性能表

> 文档时间: 2026-07-15 02:21:10

---

## 交付

| 项 | 路径 |
|----|------|
| AVEO async 分阶段 | `aveo_session_dgemm_async`：H2D(A‖B) → kernel → D2H |
| Phi 常驻 worker | `kernels/phi/phi_worker.c` + `backends/phi_worker.py` |
| PowerCap | `power.py`（本地模型 + 可选 import uni `PowerCap`） |
| Hetero | multi-batch 支持 `use_phi_worker` + power guard |
| 统一性能表 | `scripts/bench_summary.py` |
| AVEO 分阶段 bench | `scripts/bench_aveo_async.py` |

## AVEO 分阶段（单卡 session）

| N | mode | total s | h2d | kern | d2h | kernel GFLOPS |
|---|------|---------|-----|------|-----|---------------|
| 512 | async | 0.0027 | 0.0014 | 0.0002 | 0.0010 | ~1150 |
| 1024 | async | 0.0093 | 0.0049 | 0.0014 | 0.0029 | ~1560 |
| 1536 | async | 0.0193 | 0.0095 | 0.0042 | 0.0057 | ~1740 |

说明：sync `aveo-session` 的 `elapsed` 主要量的是 `call_sync`（不含完整 H2D/D2H），GFLOPS 会偏高；**async 路径的 total/h2d/kern/d2h 更适合做数据面分析**。大矩阵 H2D+D2H 常超过 kernel。

## Hetero multi-batch（4×512×384×384）

| phi_worker | overlap | wall | batch/s |
|------------|---------|------|---------|
| False | False | 12.39 s | 0.323 |
| False | True | 11.71 s | 0.342 |
| True | False | 16.11 s | 0.248（控制面 scp 额外开销） |
| True | True | **11.63 s** | **0.344** |

PowerCap：`backend=uni`，limit=1440W。

## 统一性能摘要（`bench_summary.py` 摘录）

| scope | case | metric |
|-------|------|--------|
| host | openblas@2048 | ~481 GFLOPS |
| host | avx512@512 | ~169 GFLOPS |
| phi | mkl@1024 | ~630 GFLOPS |
| ve | pool@1024 | wall **0.035 s** |
| ve | aveo@1024 | wall 0.074 s |
| ve | oneshot@1024 | wall 0.157 s |
| hetero | phi_worker+pool ×3 | ~0.34 batch/s |

## 测试

**29 pytest passed**

## 复现

```bash
export INTEL_LICENSE_FILE=$HOME/parallel_studio.lic
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=48
source env/.venv/bin/activate
python scripts/bench_aveo_async.py
python scripts/bench_hetero_overlap.py
python scripts/bench_summary.py
pytest -q
```
