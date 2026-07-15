# 实现记录 — N+1：AVEO、Hetero 重叠、Layout 代价、Host auto

> 文档时间: 2026-07-15 02:14:26  
> 对应路线图: `docs/plan/20260714_220428_next_optimization_roadmap.md` 迭代 N+1

---

## 交付

| 能力 | 路径 |
|------|------|
| VE AVEO lib | `kernels/ve/dgemm_lib.c` → `/tmp/uni_cute_libve_dgemm.so` |
| Host VEO wrapper | `kernels/host/veo_dgemm_host.c` + `backends/ve_aveo.py` |
| multi-VE AVEO | `multi_ve_aveo_dgemm` / `AveoSessionPool` |
| Hetero multi-batch 重叠 | `apps/hetero_pipeline.run_hetero_multibatch` |
| Layout 代价选切分 | `partition/cost_model.py` |
| Host auto 后端 | `host_dgemm(backend="auto"\|"openblas"\|"avx512")` |
| 路径对比 bench | `scripts/bench_ve_paths.py` |
| 重叠 bench | `scripts/bench_hetero_overlap.py` |

环境要点：

```bash
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
# VEO node 编号与 ve_exec -N 一致：1,2,3（0 offline）
```

## 实测：multi-VE 三路径 wall

| shape | oneshot 文件 | **worker pool** | **AVEO session** |
|-------|--------------|-----------------|------------------|
| 512³ | 0.143 s | **0.014 s** | 0.018 s |
| 1024³ | 0.215 s | **0.032 s** | 0.036 s |

- 正确性：err ~1e-13  
- 卡内 kgflops：pool/AVEO 与 oneshot 同量级（~1.2–1.6 TFLOPS/卡 量级均值）  
- **结论**：常驻（pool 或 AVEO session）相对每次 `ve_exec` 仍有 **~6–10×** wall 优势；AVEO 与文件 worker 接近，数据面无文件但 H2D/D2H 仍在。

## Hetero multi-batch

| 模式 | wall (4×512×384×384) | batch/s |
|------|----------------------|---------|
| 串行 | 12.43 s | 0.322 |
| Phi∥VE 重叠 | **11.75 s** | **0.340**（~**+6%**） |

Phi scp 仍占主导，重叠收益有限；AVEO/更大 VE 算量时重叠会更明显。

## Host auto

- `max(M,N,K) < 768` → avx512 手写  
- 否则 → numpy/OpenBLAS  
- `test_host_dgemm_auto` 覆盖

## Layout 代价

- `choose_best_placement` 比较 `row_blocks` vs `col_blocks`  
- 基于 transfer/PCIe + flops/peak 的粗模型（非仿真器）

## 测试

- **pytest 27 passed**

## 复现

```bash
export INTEL_LICENSE_FILE=$HOME/parallel_studio.lic
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=48
source env/.venv/bin/activate
python scripts/bench_ve_paths.py
python scripts/bench_hetero_overlap.py
pytest -q
```
