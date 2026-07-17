# 日常算力服务（uct-serve）

> 面向本机组内投递；**不**暴露公网。指标：thr、延迟、正确性、可复现；**不含功耗**。  
> Phi **默认关**；job 里 `"phi": true` 时再开（失败自动 Host）。  
> 版本：v1.7+

## 推荐日常命令（三件套）

```bash
source env/.venv/bin/activate
export PYTHONPATH=src

# 1) 开服（预热 pin 256³，VE node 1）
bash scripts/service_start.sh
# 调试无卡: UCT_HOST_ONLY=1 bash scripts/service_start.sh

# 2) 投递
uct-run --socket jobs/service_dense_stream.json
uct-run --socket --health
python scripts/submit_loop.py --job jobs/service_dense_stream.json --n 20 --socket

# 3) 停服
bash scripts/service_stop.sh
```

环境变量（`service_start.sh`）：

| 变量 | 默认 | 含义 |
|------|------|------|
| `UCT_SOCKET` | `/tmp/uct-serve.sock` | Unix socket |
| `UCT_SERVE_LOG` | `artifacts/uct-serve.log` | 日志 |
| `UCT_SERVE_PID` | `artifacts/uct-serve.pid` | pidfile |
| `UCT_PRELOAD` | `256` | 预热 pin 边长 |
| `UCT_VE_NODE` | `1` | VE 节点 |
| `UCT_HOST_ONLY` | `0` | `1` 时不碰 VE |

## 手工启动（等价）

```bash
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib
export OMP_NUM_THREADS=48

uct-serve --socket /tmp/uct-serve.sock --preload 256 --ve-node 1
# uct-serve --socket /tmp/uct-serve.sock --host-only
```

## 服务型 job 模板

| 文件 | 用途 |
|------|------|
| `jobs/service_dense_stream.json` | 多 batch 稠密流（force_ve pin） |
| `jobs/service_sparse_dense.json` | stencil SpMV → dense |
| `jobs/phi_prep_ve_gemm.json` | Phi 按需 scale → VE gemm |
| `jobs/dense_external.json` | 外部 `.npy` GEMM（先 `make_sample_matrices`） |
| `jobs/sparse_external.json` | 外部 CSR `.npz` |
| `jobs/dense_batch.json` 等 | 通用模板 |

### Job 字段（常用）

| 字段 | 含义 |
|------|------|
| `type` | dense_batch / sparse_dense / dataprep / phi_prep_ve_gemm / service_* |
| `phi` | `true` 时尝试 Phi scale，否则 Host |
| `force_ve` | 强制 AVEO pin（忽略 recommend=host） |
| `batches` | 多 batch 次数 |
| `compare_oneshot` | 是否测 cold oneshot 对照 |
| `pin_mode` | `grow`（默认，超 preload 自动 re-pin）或 `strict`（超限报错，要求重启更大 preload） |
| `matrix_a` / `matrix_b` | 外部稠密矩阵路径（`.npy` / `.npz`） |
| `csr_path` | 外部 CSR（`.npz`: indptr/indices/data[/nrows/ncols]） |
| `matrix_x` / `matrix_w` | sparse_dense 的 RHS / 投影矩阵 |

### 外部矩阵

```bash
python scripts/make_sample_matrices.py   # → artifacts/sample_data/
uct-run jobs/dense_external.json --host-only
uct-run jobs/sparse_external.json --host-only
```

大文件**不要**入库；job 只引用本机路径。

## 指标（服务侧）

| 字段 | 含义 |
|------|------|
| `queue_wait_sec` | 拿到服务串行锁前的等待（顶层 result 与 metrics 均有） |
| `throughput_batches_per_sec` | 本 job 吞吐 |
| `max_abs_err` | 正确性 |
| `prep_sec` / `prep_note` | 可选 Phi prep |
| `pin_mode` | 实际 pin 策略 |

## 健康检查

```bash
uct-run --socket --health
# sessions.aveo / jobs_done / jobs_fail
uct-run --socket --ping
```

## pin 与 preload

- 默认 `pin_mode=grow`：job 尺寸 > preload 时自动扩大 pin（服务继续，但可能有一次 re-pin 成本）。
- 生产可在 job 中设 `"pin_mode": "strict"`：超过 preload 直接失败并提示重启更大 `--preload`，避免静默抖动。

## 本机回归

```bash
bash scripts/ci_l0.sh          # 无加速器
bash scripts/ci_device.sh      # VE/Phi 若在则测
bash scripts/ci_all.sh         # L0 + device；UCT_SKIP_DEVICE=1 跳过 device
```

**无 GitHub CI**（公网 runner 无本机设备）。

## 论文对照

```bash
python scripts/paper_sweep.py --quick --exp-id smoke
python scripts/paper_sweep.py --exp-id esc4000_baseline --seed 0
python scripts/paper_plot.py artifacts/paper/esc4000_baseline   # 需 matplotlib
```

基线表见 `docs/impl/*_paper_baseline_esc4000.md` 与 README「何时 Host / VE / serve」。

## 安全

- 仅 **Unix socket**（默认 `/tmp/uct-serve.sock`，mode 0600）
- 不监听 TCP；不用于不可信网络
- 不提交 license / 大矩阵 / 设备序列号
