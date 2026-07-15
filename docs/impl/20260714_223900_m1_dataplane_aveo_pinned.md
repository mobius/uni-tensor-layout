# Phase 2 M1 — DataPlane + AVEO pinned + Timeline + perf gate

> 文档时间: 2026-07-14 22:39:00  
> 版本: **v0.7.0**  
> 计划: `docs/plan/20260714_223206_phase2_system_roadmap.md` M1

---

## 1. 交付内容

| 任务 | 实现 | 状态 |
|------|------|------|
| W1.4 DataPlane | `src/uni_cute_tensor/runtime/dataplane.py` — host \| file \| worker \| aveo \| aveo_pinned | 完成 |
| W1.1 AVEO 驻留缓冲 | `aveo_session_pin` / `aveo_session_dgemm_pinned` + `AveoSessionPool.pin_all` / `dgemm_pinned` | 完成 |
| W4.1 Timeline JSONL | `runtime/timeline.py` span/emit/write_jsonl；DataPlane gemm 挂 span | 完成 |
| W4.3 perf gate | `bench_summary.py --write-doc` → `docs/impl/<ts>_perf_gate.md` | 完成 |
| 测试 | `tests/test_runtime.py` + 既有 device 套件 | 完成 |

### 新增/变更路径

```
src/uni_cute_tensor/runtime/
  __init__.py
  dataplane.py      # create_dataplane / GemmRequest / GemmResult
  timeline.py       # Timeline + timeline_scope
src/uni_cute_tensor/kernels/host/veo_dgemm_host.c  # pin + dgemm_pinned
src/uni_cute_tensor/backends/ve_aveo.py            # ctypes + pool API
scripts/bench_dataplane.py
scripts/bench_aveo_pinned.py
scripts/bench_summary.py   # + pinned + dp rows + markdown gate
tests/test_runtime.py
```

---

## 2. 实机结果（本机 ESC4000 G4）

### 2.1 AVEO session vs pinned（16 batch 单 VE）

| N | session batch/s | pinned batch/s | 备注 |
|---|-----------------|----------------|------|
| 256 | ~447 | ~597 | 小矩阵 alloc 占比高，pinned 收益明显 |
| 512 | ~123 / **206** | ~130 / **258** | summary 冷路径后 ~**1.25×** |
| 1024 | ~41 | ~44 | 接近 PCIe/H2D 上限，收益收窄 |

结论: **W1.1 在 mid-size 多 batch 上相对 session 循环 ≥+20% 吞吐（512 约 +25%）**；大 N 以传输为主，再压 alloc 收益有限。

### 2.2 DataPlane 统一路径 @512³

| backend | wall (s) | kernel GF | status |
|---------|----------|-----------|--------|
| host | ~0.01 | ~270 | pass |
| file | ~0.23 | ~1210 | pass |
| worker | ~0.02 | ~1190 | pass |
| aveo | ~0.03 | ~940–1000 | pass |
| aveo_pinned | ~0.01 | ~1160 | pass |

应用侧只需 `create_dataplane(name)` + `plane.gemm(GemmRequest(...))`。

### 2.3 Timeline

`artifacts/timeline_{backend}.jsonl` 含 `phase`/`device`/`duration_sec`/`job_id`。  
`Timeline.summary()` 按 phase 聚合，可画阶段占比。

### 2.4 回归

- unit: 19 passed  
- device: 15 passed  
- `bench_summary` 全 pass；见 `docs/impl/20260714_223845_perf_gate.md`

---

## 3. API 草案 v0.1（运行时）

```python
from uni_cute_tensor.runtime import create_dataplane, GemmRequest, timeline_scope

with timeline_scope(job_id="job-1") as tl:
    with create_dataplane("aveo_pinned", devices=["ve1"], pin_m=1024, pin_n=1024, pin_k=1024) as plane:
        r = plane.gemm(GemmRequest(a=A, b=B))
tl.write_jsonl(Path("artifacts/timeline.jsonl"))
```

底层 pin（不经 DataPlane）:

```python
with AveoSessionPool([1]) as pool:
    pool.pin_all(M, N, K)
    C, res = pool.dgemm_pinned(1, A, B)
```

---

## 4. 限制与后续

- `aveo_pinned` 多设备路径仍走 row-shard 的非 pin multi-VE（pin 仅单设备 gemm 热路径）。  
- pin 容量必须 ≥ 每次 M,N,K；超容量返回错误。  
- 真 DMA∥kernel 重叠仍属 W1.2（M1 未做）。  
- Phi 数据面仍 scp（W1.3 → 后续）。  
- M2: PlacementPlan / 自动放置 / 真应用。

---

## 5. 复现

```bash
source env/.venv/bin/activate
export PYTHONPATH=src
export LD_LIBRARY_PATH=/opt/nec/ve/veos/lib64:$LD_LIBRARY_PATH
export VE_LD_LIBRARY_PATH=/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib

python scripts/bench_aveo_pinned.py
python scripts/bench_dataplane.py
python scripts/bench_summary.py          # writes docs/impl/*_perf_gate.md
pytest -q
pytest -q -m device
```
