# 实现记录 — Phi DGEMM IMCI/OpenMP 优化

> 文档时间: 2026-07-14 08:46:54

---

## 优化内容

| 项 | 说明 |
|----|------|
| 编译 | `icc -std=c99 -mmic -O3 -openmp -restrict` |
| 并行 | OpenMP **按行** `schedule(static)`（修正早期按 BM tile 并行导致线程空闲） |
| 向量 | IMCI `_mm512_fmadd_pd`，J 维 4×8 展开（32 doubles） |
| 对齐 | N pad 到 8 的倍数，保证 `load_pd/store_pd`（KNC 无 loadu） |
| 运行时 | scp `libiomp5.so` + `LD_LIBRARY_PATH`；`KMP_AFFINITY=balanced` |
| 回退 | 无 ICC 时仍可用 k1om-gcc pthread 标量路径 |

参考：`intel_phi/tests/perf/phi_peak_dgemm.c`（原实测约 63 GFLOPS @ 2048）。

## 基准（本机 mic0，icc-mmic）

| N | GFLOPS | max_abs_err vs numpy |
|---|--------|----------------------|
| 128 | 10.3 | 0 |
| 256 | **78.8** | 0 |
| 512 | **95.3** | 1.8e-13 |
| 1024 | **94.8** | 4.3e-13 |
| 1536 | 92.3 | 8.0e-13 |
| **2048** | **99.7** | 8.8e-13 |
| 384 | 73.1 | 0 |
| 1000 | 87.1 | 3.8e-13 |

对比：优化前朴素 pthread 约 **0.7–3 GFLOPS**；intel_phi 简易向量化约 **63 GFLOPS @ 2048**。

## 全套实测

- pytest **20 passed**
- multi-VE NLC + Phi peak + Phi DGEMM 1024³：**failed=0**（Phi kernel ~71 GFLOPS @ 120 threads）

## 复现

```bash
export INTEL_LICENSE_FILE=$HOME/parallel_studio.lic
source env/.venv/bin/activate
python scripts/bench_phi_dgemm.py
python scripts/run_real_tests.py --phi-m 1024 --phi-k 1024 --phi-n 1024
```

## 后续可优化

- L2 分块（BK/BN）进一步抬升大矩阵  
- 调用 MKL DGEMM for MIC（若库可用）  
- Host AVX-512 分块同样升级  
