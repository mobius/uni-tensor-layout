# 实现记录 — VE 常驻 worker、异构流水线、Host BLAS 公平对照

> 文档时间: 2026-07-15 01:59:54

---

## 1. VE 常驻 worker（降低 ve_exec 启动）

| 组件 | 路径 |
|------|------|
| 内核 | `kernels/ve/dgemm_worker.c` |
| 池 | `backends/ve_worker.py` → `VeWorkerPool` |
| 基准 | `scripts/bench_ve_launch.py` |

协议：`control_dir/job.cmd` + `job.go` → worker 写 `job.log` / `job.status`。

### wall 对比（3×VE，share_b，正确性全过）

| shape | oneshot (ve_exec) | **pooled (常驻)** | 加速 |
|-------|-------------------|-------------------|------|
| 512³ | 0.127 s | **0.012 s** | ~**10×** |
| 1024³ | 0.196 s | **0.037 s** | ~**5×** |
| 1536×1024×1024 | 0.179 s | **0.044 s** | ~**4×** |

结论：中小规模 wall 几乎被进程启动主导；常驻 worker 后有效吞吐显著上升。

## 2. 异构流水线 Phi → multi-VE

| 步骤 | 设备 | 算子 |
|------|------|------|
| 1 | Phi | `A' = αA + β`（IMCI OpenMP） |
| 2 | 3×VE | `C = A' @ B`（NLC，worker pool + share_b） |

- 代码：`apps/hetero_pipeline.py`、`examples/demo_hetero_pipeline.py`
- 实测：`status=pass`，`err~3e-13`，VE 分片 kernel 均值 ~1 TFLOPS 级

## 3. Host 公平对照（多轮中位）

numpy 后端：**OpenBLAS 0.3.33**（scipy-openblas64，Haswell 动态，最多 64 线程）。

| N | numpy 中位 GFLOPS | AVX-512 手写中位 | 比值 |
|---|-------------------|------------------|------|
| 256 | 189 | 139 | 0.73 |
| 512 | 260 | 177 | 0.68 |
| 1024 | 403 | 159 | 0.40 |
| 2048 | 485 | 129 | 0.27 |
| 3072 | 525 | 186 | 0.35 |

手写内核在教学/可控路径有价值；生产大矩阵仍应用 OpenBLAS/MKL。

基准：`python scripts/bench_host_blas_fair.py`

## 4. 测试

- pytest **24 passed**（含 worker pool、hetero pipeline）

## 5. 复现

```bash
export INTEL_LICENSE_FILE=$HOME/parallel_studio.lic
export OMP_NUM_THREADS=48
source env/.venv/bin/activate
python scripts/bench_ve_launch.py
python scripts/bench_host_blas_fair.py
python examples/demo_hetero_pipeline.py
pytest -q
```
