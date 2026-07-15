# 实现记录 — N+3：AVEO 双缓冲批处理、Phi SSH 控制面、持续吞吐

> 文档时间: 2026-07-15 02:27:43

---

## 交付

| 项 | 说明 |
|----|------|
| AVEO batch dual-buf | `aveo_session_dgemm_batch`：双 slot 复用 VE 缓冲，多 batch 一次 session |
| Phi worker 控制面 | 远程 `printf` 写 job.cmd/go，**不再 scp 控制文件**（`phi-worker-sshctl`） |
| 持续吞吐 | `scripts/bench_sustained.py` + PowerCap + 可选功率采样 |
| 功率采样 | `power_sample.py`（ipmitool 有则采样，无则 n=0） |

## AVEO multi-batch（8 batches，单卡 ve1）

| N | loop-async wall | dual-buf wall | 加速 | batch/s (dual) |
|---|-----------------|---------------|------|----------------|
| 256 | 0.021 s | **0.0046 s** | ~**4.6×** | 1737 |
| 512 | 0.065 s | **0.018 s** | ~**3.5×** | 434 |
| 1024 | 0.183 s | **0.058 s** | ~**3.1×** | 137 |

正确性 err ≤ 4.5e-13。

## 持续吞吐（3×VE worker pool，512³，10s）

| 指标 | 值 |
|------|-----|
| jobs | 663 |
| jobs/s | **65.7** |
| effective_gflops（含调度/传输） | 17.6 |
| max_err | 2e-13 |
| PowerCap | uni，1440W |
| ipmitool 功率样本 | 0（本机无可用传感器输出） |

## Phi worker

- `compiler=phi-worker-sshctl`，SCALE 正确性 pass  
- 控制面 round-trip 仍受 ssh 延迟影响；数据面仍 scp/cat  

## 测试

**30 pytest passed**

## 复现

```bash
export INTEL_LICENSE_FILE=$HOME/parallel_studio.lic
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
source env/.venv/bin/activate
python scripts/bench_aveo_batch.py
python scripts/bench_sustained.py --seconds 10 --m 512 --k 512 --n 512
pytest -q
```
