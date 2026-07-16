# 日常算力服务（uct-serve）

> 面向本机组内投递；**不**暴露公网。指标：jobs/s、正确性；**不含功耗**。  
> Phi **默认关**；job 里 `"phi": true` 或需要时再开（失败自动 Host）。

## 快速开始

```bash
source env/.venv/bin/activate
export PYTHONPATH=src
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib

# 终端 1：常驻（无卡调试加 --host-only）
uct-serve --socket /tmp/uct-serve.sock
# 预热 VE pin（推荐服务场景）:
# uct-serve --socket /tmp/uct-serve.sock --preload 256 --ve-node 1

# 终端 2：投递
uct-run --socket jobs/service_dense_stream.json
uct-run --socket jobs/service_sparse_dense.json
uct-run --socket --health
uct-run --socket --ping

# 连续投递测 thr
python scripts/submit_loop.py --job jobs/service_dense_stream.json --n 20 --socket
```

## 服务型 job 模板

| 文件 | 用途 |
|------|------|
| `jobs/service_dense_stream.json` | 多 batch 稠密流（force_ve pin） |
| `jobs/service_sparse_dense.json` | stencil SpMV → dense |
| `jobs/phi_prep_ve_gemm.json` | Phi 按需 scale → VE gemm |
| `jobs/dense_batch.json` 等 | 通用模板 |

### Job 字段（常用）

| 字段 | 含义 |
|------|------|
| `type` | dense / sparse_dense / dataprep / phi_prep_ve_gemm / service_* |
| `phi` | `true` 时尝试 Phi scale，否则 Host |
| `force_ve` | 强制 AVEO pin（忽略 recommend=host） |
| `batches` | 多 batch 次数 |
| `compare_oneshot` | 是否测 cold oneshot 对照 |

## 健康检查

```bash
uct-run --socket --health
# sessions.aveo / jobs_done / jobs_fail
```

## 停止

```bash
uct-run --socket --shutdown
# 或 Ctrl+C 服务进程
```

## 安全

- 仅 **Unix socket**（默认 `/tmp/uct-serve.sock`，mode 0600）
- 不监听 TCP；不用于不可信网络
