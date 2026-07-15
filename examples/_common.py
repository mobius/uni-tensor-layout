"""Shared helpers for terminal-facing examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def ensure_repo_path() -> Path:
    return ROOT


def setup_ve_env() -> None:
    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )


def add_common_args(p: argparse.ArgumentParser, *, default_out: str) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--host-only", action="store_true", help="Force host path (no Phi/VE)")
    p.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Comma list e.g. ve1,ve2,ve3 or 'auto'",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "aveo", "pool", "host"),
        default="auto",
        help="Dense backend hint (auto picks by hardware)",
    )
    p.add_argument("--phi", action="store_true", help="Enable Phi prep when available")
    p.add_argument("--no-power-cap", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / default_out)
    p.add_argument("--quiet", action="store_true")


def resolve_ve_devices(spec: str, *, host_only: bool) -> list[str]:
    if host_only:
        return []
    from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names
    from uni_cute_tensor.backends.ve_dgemm import ve_toolchain_available

    if not ve_toolchain_available():
        return []
    found = ve_device_names(discover_devices())
    if spec.strip().lower() in ("", "auto"):
        return list(found)
    want = [d.strip() for d in spec.split(",") if d.strip()]
    return [d for d in want if d in found] or list(found)


def make_power_cap(enabled: bool):
    from uni_cute_tensor.power import PowerCap

    if not enabled:
        # effectively unlimited local cap
        cap = PowerCap(psu_limit_w=1e9, safety_margin=1.0)
        cap._uni = None
        return cap
    return PowerCap()


def write_metrics(out_dir: Path, metrics: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_kv(title: str, rows: Sequence[tuple[str, Any]], *, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"=== {title} ===")
    for k, v in rows:
        print(f"  {k:<18} {v}")


def column_normalize(x: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_norm, mean, std) with column-wise z-score."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std, mean, std


def host_scale(a: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return np.ascontiguousarray(alpha * a + beta, dtype=np.float64)


def maybe_phi_scale(
    a: np.ndarray,
    *,
    use_phi: bool,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[np.ndarray, str]:
    if not use_phi:
        return host_scale(a, alpha, beta), "host_scale"
    try:
        from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale
        from pathlib import Path

        if not Path("/dev/mic0").exists():
            return host_scale(a, alpha, beta), "host_scale(no_mic)"
        out, prep = run_phi_prep_scale(a, alpha=alpha, beta=beta)
        if prep.status != "pass":
            return host_scale(a, alpha, beta), f"host_scale(phi_fail:{prep.status})"
        return out, f"phi_scale gflops={prep.gflops:.2f}"
    except Exception as exc:
        return host_scale(a, alpha, beta), f"host_scale(phi_skip:{exc})"


def dense_gemm_auto(
    a: np.ndarray,
    b: np.ndarray,
    devices: Sequence[str],
    *,
    power_cap=None,
    backend_hint: str = "auto",
) -> dict[str, Any]:
    """Run dense GEMM via host or auto placement on VE."""
    from uni_cute_tensor.backends.host_dgemm import host_dgemm
    from uni_cute_tensor.partition.cost_model import choose_best_placement
    from uni_cute_tensor.partition.multi_device import partition_matrix_rows
    from uni_cute_tensor.partition.runner import execute_plan
    from uni_cute_tensor.power import PowerCap

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    n = b.shape[1]
    cap = power_cap or PowerCap()
    t0 = time.perf_counter()

    if not devices or backend_hint == "host":
        c, r = host_dgemm(a, b, backend="auto")
        wall = time.perf_counter() - t0
        return {
            "c": c,
            "wall_sec": wall,
            "status": r.status,
            "max_abs_err": r.max_abs_err,
            "backend": "HOST",
            "strategy": "host",
            "devices": ["host"],
            "kernel_gflops": r.gflops,
            "plan": None,
            "notes": [f"host_dgemm atom={r.atom_name}"],
        }

    choice = choose_best_placement(
        m, n, k, list(devices), backend="VE_NLC", power_cap=cap
    )
    with cap.guard(choice.devices, {d: "dgemm" for d in choice.devices}):
        res = execute_plan(a, b, choice.plan, use_pool=True, check=True)
    wall = time.perf_counter() - t0
    return {
        "c": res.c,
        "wall_sec": wall,
        "status": res.status,
        "max_abs_err": res.max_abs_err,
        "backend": res.backend,
        "strategy": res.strategy,
        "devices": list(choice.devices),
        "kernel_gflops": res.kernel_gflops_mean,
        "plan": choice.plan,
        "est_total_sec": choice.est_total_sec,
        "notes": list(res.shard_notes),
    }


def fixed_row_baseline(
    a: np.ndarray,
    b: np.ndarray,
    devices: Sequence[str],
) -> dict[str, Any]:
    """Compare against fixed full-device row partition (when ≥2 VEs)."""
    from uni_cute_tensor.partition.multi_device import partition_matrix_rows
    from uni_cute_tensor.partition.runner import execute_plan

    if len(devices) < 2:
        return {"wall_sec": None, "status": "skip", "note": "need >=2 VE"}
    m, k = a.shape
    n = b.shape[1]
    plan = partition_matrix_rows(m, n, list(devices), k=k, backend="VE_NLC")
    t0 = time.perf_counter()
    res = execute_plan(a, b, plan, use_pool=True, check=True)
    wall = time.perf_counter() - t0
    return {
        "wall_sec": wall,
        "status": res.status,
        "max_abs_err": res.max_abs_err,
        "strategy": "row_blocks_fixed_all",
        "devices": list(devices),
    }
