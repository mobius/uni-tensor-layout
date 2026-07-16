# AVEO / VEO async & overlap limits

> 文档时间: 2026-07-15 22:15:00  
> 版本: v1.4.0 / Phase 3 M3  
> 相关: `scripts/bench_aveo_overlap_limits.py`，`kernels/host/veo_dgemm_host.c`

---

## 1. VEO programming model (this stack)

Host uses **libveo**:

| API | Role |
|-----|------|
| `veo_write_mem` / `veo_async_write_mem` | H2D |
| `veo_call_sync` / `veo_call_async` | launch VE symbol |
| `veo_read_mem` / `veo_async_read_mem` | D2H |
| `veo_thr_ctxt` + `veo_call_wait_result` | async command queue |

Our session holds **one** `veo_thr_ctxt` (`aveo_session.ctx`).

---

## 2. What “async” does **not** mean here

Unlike CUDA streams with concurrent copy engines:

1. Commands on **one VEO context are ordered**.  
2. Typical path: async H2D A, async H2D B → **wait both** → async call → **wait** → async D2H → **wait**.  
3. Therefore **kernel does not run under H2D** of the same batch on that context.  
4. Phase timers (`h2d`, `kernel`, `d2h`) usually **sum ≈ wall** (little true overlap).

**Implication**: Do not design algorithms assuming PCIe DMA free-runs under NLC DGEMM on a single context.

---

## 3. Where thr still improves

| Technique | Mechanism |
|-----------|-----------|
| Session reuse | Avoid `veo_proc_create` / load_library per job |
| Pinned buffers | Avoid alloc/free per GEMM |
| Dual-buffer batch | Two VE buffer sets; finish D2H of slot while scheduling next H2D/call on other slot **when queue allows**; mainly hides host-side stalls + alloc |
| Multi-VE | Parallel procs/contexts on different cards |

Measure: `python scripts/bench_aveo_overlap_limits.py`.

---

## 4. Best practice (this machine class)

1. Multi-job: **shared session + pin** (`runtime/session.py`, `uct-run`).  
2. Many equal-size batches: **`dgemm_batch` dual-buf** or pin loop.  
3. Single large GEMM: Host OpenBLAS may still win wall — use `uct-recommend`.  
4. Do not promise “CUDA-like overlap” in docs or SLAs.

---

## 5. Future (out of scope unless proven)

- Multiple VEO contexts / devices for copy-compute split  
- Vendor APIs beyond current libveo queue semantics  
