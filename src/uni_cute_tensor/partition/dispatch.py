"""Dispatch policy: recommend host vs VE (vs optional hetero) for a GEMM-like job.

Phase 3 M1 — answers "should this job use accelerators?" using the calibrated
cost model (when available). Recommendations are advisory; mid-size Host OpenBLAS
often wins single-shot wall on this machine class.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from uni_cute_tensor.partition.cost_model import (
    estimate_gemm_placement,
    get_device_model,
    try_autoload_calibration,
)

BackendRec = Literal["host", "ve", "hetero"]


@dataclass
class BackendEstimate:
    backend: str
    est_wall_sec: float
    est_compute_sec: float = 0.0
    est_transfer_sec: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchRecommendation:
    recommended: BackendRec
    reason: str
    estimates: list[BackendEstimate]
    m: int
    n: int
    k: int
    batches: int
    ve_available: bool
    phi_available: bool
    calibration_loaded: bool
    confidence: str  # high | medium | low

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended,
            "reason": self.reason,
            "estimates": [e.to_dict() for e in self.estimates],
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "batches": self.batches,
            "ve_available": self.ve_available,
            "phi_available": self.phi_available,
            "calibration_loaded": self.calibration_loaded,
            "confidence": self.confidence,
        }


def _estimate_host(m: int, n: int, k: int, batches: int) -> BackendEstimate:
    md = get_device_model("host")
    flops = 2.0 * m * n * k
    compute = flops / (md.peak_gflops * 1e9) if md.peak_gflops > 0 else 0.0
    # multi-batch: almost full recompute, tiny fixed overhead
    wall = batches * (compute + md.launch_overhead_sec)
    return BackendEstimate(
        backend="host",
        est_wall_sec=wall,
        est_compute_sec=batches * compute,
        est_transfer_sec=0.0,
        notes=[f"peak={md.peak_gflops:.0f}GF launch={md.launch_overhead_sec:.4f}s"],
    )


def _estimate_ve(
    m: int,
    n: int,
    k: int,
    batches: int,
    devices: Sequence[str],
    *,
    mode: str = "pin",
) -> BackendEstimate:
    """mode: pin (amortize launch once) | pool | oneshot (launch per batch)."""
    if not devices:
        return BackendEstimate(
            backend="ve",
            est_wall_sec=float("inf"),
            notes=["no VE devices"],
        )
    # pin/oneshot are single-node paths; pool may use multi-VE row split
    use_devs = list(devices) if mode == "pool" else list(devices)[:1]
    ch = estimate_gemm_placement(
        m, n, k, use_devs, strategy="row_blocks", backend="VE_NLC"
    )
    md = get_device_model("ve")
    n_dev = max(len(use_devs), 1)
    # strip multi-device launch inflation for per-batch body after session is warm
    launch_pack = md.launch_overhead_sec * (0.7 + 0.5 * n_dev)
    per_batch_body = max(
        ch.est_compute_sec,
        ch.est_compute_sec + max(0.0, ch.est_transfer_sec - launch_pack) * 0.85,
    )

    if mode == "oneshot":
        wall = batches * ch.est_total_sec
        note_mode = "oneshot(full_launch_each)"
    elif mode == "pool":
        wall = launch_pack + batches * (per_batch_body + 0.25 * md.launch_overhead_sec)
        note_mode = "pool(shared_workers)"
    else:  # pin — best multi-batch case on one VE
        wall = md.launch_overhead_sec + batches * (
            per_batch_body + 0.1 * md.launch_overhead_sec
        )
        note_mode = "aveo_pin(amortized)"

    return BackendEstimate(
        backend=f"ve:{mode}",
        est_wall_sec=wall,
        est_compute_sec=batches * ch.est_compute_sec,
        est_transfer_sec=batches * ch.est_transfer_sec,
        notes=[
            note_mode,
            f"devs={use_devs}",
            f"peak={md.peak_gflops:.0f}GF pcie={md.pcie_gbps:.1f}GB/s",
            f"single_est={ch.est_total_sec:.4f}s",
        ],
    )


def _estimate_hetero(
    m: int,
    n: int,
    k: int,
    batches: int,
    ve_devices: Sequence[str],
) -> BackendEstimate:
    """Phi prep + VE gemm (serial estimate; overlap not assumed)."""
    phi = get_device_model("phi")
    # prep ~ scale on m*k elements (very rough: treat as 2*m*k flops)
    prep = (2.0 * m * k) / (phi.peak_gflops * 1e9) + phi.launch_overhead_sec
    ve = _estimate_ve(m, n, k, batches, ve_devices, mode="pin")
    # first batch pays prep+launch; rest prep may overlap — use 0.5*prep amortization
    wall = batches * prep * 0.5 + ve.est_wall_sec + 0.5 * prep
    return BackendEstimate(
        backend="hetero",
        est_wall_sec=wall,
        est_compute_sec=ve.est_compute_sec,
        est_transfer_sec=ve.est_transfer_sec + batches * prep * 0.2,
        notes=["phi_prep+ve_pin (partial overlap assumed)", *ve.notes],
    )


def recommend_backend(
    m: int,
    n: int,
    k: int,
    *,
    batches: int = 1,
    ve_devices: Optional[Sequence[str]] = None,
    phi_available: bool = False,
    prefer_mode: str = "pin",
    autoload_calibration: bool = True,
) -> DispatchRecommendation:
    """Recommend host | ve | hetero for dense multi-RHS style jobs C=A@B.

    Parameters
    ----------
    batches:
        Number of successive GEMMs with session reuse (pin/pool).
    ve_devices:
        e.g. ['ve1'] or multi; empty/None → probe later or treat as no VE.
    """
    cal_loaded = False
    if autoload_calibration:
        cal_loaded = bool(try_autoload_calibration())

    ve_devices = list(ve_devices or [])
    ve_ok = len(ve_devices) > 0

    estimates: list[BackendEstimate] = [_estimate_host(m, n, k, batches)]
    if ve_ok:
        estimates.append(
            _estimate_ve(m, n, k, batches, ve_devices, mode=prefer_mode)
        )
        estimates.append(_estimate_ve(m, n, k, batches, ve_devices, mode="oneshot"))
        if prefer_mode != "pool":
            estimates.append(_estimate_ve(m, n, k, batches, ve_devices, mode="pool"))
    if ve_ok and phi_available:
        estimates.append(_estimate_hetero(m, n, k, batches, ve_devices))

    # pick among host + preferred VE mode + optional hetero
    pool = {e.backend: e for e in estimates}
    pick_list = [pool["host"]]
    key = f"ve:{prefer_mode}"
    if key in pool:
        pick_list.append(pool[key])
    if "hetero" in pool:
        pick_list.append(pool["hetero"])

    best = min(pick_list, key=lambda e: e.est_wall_sec)
    host_e = pool["host"]

    if best.backend == "host":
        rec: BackendRec = "host"
        if ve_ok:
            ve_e = pool.get(key)
            ratio = (ve_e.est_wall_sec / host_e.est_wall_sec) if ve_e and host_e.est_wall_sec > 0 else 0
            reason = (
                f"Host estimated faster ({host_e.est_wall_sec:.4f}s vs "
                f"{ve_e.est_wall_sec:.4f}s VE/{prefer_mode}, ratio={ratio:.2f}); "
                f"typical for mid-size single-shot or few batches on strong OpenBLAS."
            )
        else:
            reason = "No VE devices available; Host only."
    elif best.backend == "hetero":
        rec = "hetero"
        reason = (
            f"Hetero (Phi prep + VE) estimated best wall={best.est_wall_sec:.4f}s "
            f"for batches={batches}."
        )
    else:
        rec = "ve"
        reason = (
            f"VE/{prefer_mode} estimated best wall={best.est_wall_sec:.4f}s "
            f"(host={host_e.est_wall_sec:.4f}s); favorable when batches>={batches} "
            f"amortize PCIe/launch or problem is transfer-tolerant."
        )

    # confidence: higher with calibration + larger separation
    if len(pick_list) >= 2:
        walls = sorted(e.est_wall_sec for e in pick_list)
        sep = (walls[1] - walls[0]) / walls[0] if walls[0] > 0 else 0
    else:
        sep = 1.0
    if cal_loaded and sep > 0.25:
        conf = "high"
    elif sep > 0.15 or cal_loaded:
        conf = "medium"
    else:
        conf = "low"

    # force host if VE infinite
    if rec == "ve" and not ve_ok:
        rec = "host"
        reason = "VE requested but unavailable."

    return DispatchRecommendation(
        recommended=rec,
        reason=reason,
        estimates=estimates,
        m=m,
        n=n,
        k=k,
        batches=batches,
        ve_available=ve_ok,
        phi_available=phi_available,
        calibration_loaded=cal_loaded,
        confidence=conf,
    )


def default_calibration_paths() -> list[Path]:
    """Search order for calibration JSON."""
    paths: list[Path] = []
    env = os.environ.get("UCT_CALIBRATION")
    if env:
        paths.append(Path(env))
    # package-adjacent artifacts
    root = Path(__file__).resolve().parents[3]
    paths.append(root / "artifacts" / "calibration.json")
    paths.append(Path.cwd() / "artifacts" / "calibration.json")
    return paths
