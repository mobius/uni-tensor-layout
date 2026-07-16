# Phase 3 M3 — AVEO 限制、Phi 数据面、多源功耗

> 文档时间: 2026-07-15 22:00:00  
> 版本: **1.4.0**

---

## 1. AVEO 重叠（W2.1）

Bench: `scripts/bench_aveo_overlap_limits.py`  
架构: `docs/architecture/20260715_221500_aveo_async_limits.md`

| N | 最佳 thr 模式（本机 12 batch 摘） |
|---|----------------------------------|
| 256 | pinned ~600 b/s |
| 512 | async/session ~130 b/s 量级 |
| 1024 | dual_buf ~48 b/s 优于 session ~40 |

**结论**: 单 VEO context **有序队列** → 不指望 H2D∥kernel；收益来自 **pin / dual-buf / session**。

---

## 2. Phi 数据面（W2.3）

- 默认 **SSH ControlMaster**（`UCT_PHI_SSH_MUX=1`）  
- scp 实测约 **5–7 MB/s**（256–1024² float64）  
- worker scale ~6 job/s @128²（含 scp）  

架构: `docs/architecture/20260715_221500_phi_dataplane_limits.md`  
Bench: `scripts/bench_phi_dataplane.py`

**结论**: 控制面 RTT 可降；**数据面上限 = scp/mic 链路**。

---

## 3. 功耗（W4.1）

`PowerSampler` 多源：

| 源 | 本机 |
|----|------|
| RAPL | **可用**（~110–180 W 空闲/轻载采样） |
| ipmitool | 二进制在，sensor 解析可能无有效 Power 行 → None |
| VE sensors | 需 `UCT_VE_POWER_SENSORS` 显式配置，默认不猜 |

`probe_power_sources()` / `bench_sustained` 打印 sources + degrade。

---

## 4. 复现

```bash
python scripts/bench_aveo_overlap_limits.py
python scripts/bench_phi_dataplane.py   # 需 mic0
python scripts/bench_sustained.py --seconds 5
pytest -q tests/test_power_sample.py
```
