# Phi data plane limits

> 文档时间: 2026-07-15 22:15:00  
> Phase 3 M3 / W2.3

---

## Model

| Plane | Mechanism |
|-------|-----------|
| Control | `ssh mic0` shell: write `job.cmd` / `job.go`, poll status (**no scp** for control in `PhiWorker`) |
| Data | **scp** (or `ssh cat`) of float64 binaries |

MIC cannot `fopen` Host paths from native processes in our setup → data must be copied.

---

## Improvements in v1.4

- **SSH ControlMaster** (`UCT_PHI_SSH_MUX=1`, default): reuses TCP/auth for ssh/scp (`ControlPersist=120`).  
- Cuts repeated control RTT; **does not remove** matrix copy cost.

Disable: `UCT_PHI_SSH_MUX=0`.

---

## Expected bottlenecks

1. **scp of A/B/C** dominates mid-size SCALE/DGEMM jobs.  
2. Worker helps **amortize process start**, not PCIe/mic link bandwidth.  
3. Relative v1.1: control path improved; data plane ceiling = link + scp.

Probe: `python scripts/bench_phi_dataplane.py` (requires `/dev/mic0`).

---

## Best practice

- Prefer **PhiWorker** for multi-batch prep (scale).  
- Large dense GEMM: VE or Host, not Phi, unless algorithm needs KNC.  
- Hetero: overlap Phi prep of batch i+1 with VE GEMM of batch i (`hetero_pipeline`).  
