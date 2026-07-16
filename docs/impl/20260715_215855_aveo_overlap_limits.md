# AVEO overlap / phase limits

> Generated: 20260715_215855
> nbatch=12

## Results

| N | mode | wall_s | batch/s | notes |
|---|------|-------:|--------:|-------|
| 256 | session | 0.0246 | 488.07 |  |
| 256 | async_phases | 0.0220 | 545.82 | sum_ph=0.0056 wall=0.0220 |
| 256 | dual_buf_batch | 0.0214 | 559.63 |  |
| 256 | pinned | 0.0199 | 602.21 |  |
| 512 | session | 0.0948 | 126.53 |  |
| 512 | async_phases | 0.0899 | 133.53 | sum_ph=0.0234 wall=0.0899 |
| 512 | dual_buf_batch | 0.1010 | 118.84 |  |
| 512 | pinned | 0.0971 | 123.60 |  |
| 1024 | session | 0.2969 | 40.42 |  |
| 1024 | async_phases | 0.3129 | 38.35 | sum_ph=0.1067 wall=0.3129 |
| 1024 | dual_buf_batch | 0.2522 | 47.58 |  |
| 1024 | pinned | 0.2648 | 45.32 |  |

## Conclusion

- VEO **single thr context is ordered**: async H2D then wait, then call, then D2H.
- **True DMA∥kernel on one context is not observed** as wall ≪ sum(phases).
- Throughput wins come from **session reuse**, **pinned buffers**, **dual-buf batch** (hide alloc + pipeline D2H of i with setup of i+1 at host scheduling level).
- Prefer `aveo_pin` / `dgemm_batch` for multi-job; do not expect CUDA-style streams.

