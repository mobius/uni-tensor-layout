"""Placement cost model with calibration + auto device/strategy selection.

W2.2: calibrated pcie_gbps / peak_gflops / launch_overhead.
W2.3: row/col/k_split + device subset under PowerCap.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

from uni_cute_tensor.partition.multi_device import (
    BACKEND_ATOMS,
    DTYPE_BYTES,
    PlacementPlan,
    partition_matrix_cols,
    partition_matrix_k,
    partition_matrix_rows,
)
from uni_cute_tensor.partition.pcie_cost import DEFAULT_PCIE_GBPS, estimate_h2d_seconds

Strategy = Literal["row_blocks", "col_blocks", "k_split"]


@dataclass
class DeviceModel:
    name: str
    kind: str  # ve | phi | host
    peak_gflops: float
    pcie_gbps: float = DEFAULT_PCIE_GBPS
    launch_overhead_sec: float = 0.005
    online: bool = True


# Defaults from v0.7–0.8 gates: pool cold-start dominates mid-size multi-VE
DEFAULT_MODELS: dict[str, DeviceModel] = {
    "ve": DeviceModel(
        "ve", "ve", peak_gflops=1550.0, pcie_gbps=8.0, launch_overhead_sec=0.06
    ),
    "phi": DeviceModel(
        "phi", "phi", peak_gflops=620.0, pcie_gbps=6.0, launch_overhead_sec=0.08
    ),
    "host": DeviceModel(
        "host", "host", peak_gflops=480.0, pcie_gbps=100.0, launch_overhead_sec=0.0
    ),
}

# Per-device override table used after calibration
_CALIBRATION: dict[str, DeviceModel] = {}
_AUTOLOAD_TRIED: bool = False
_AUTOLOAD_OK: bool = False


@dataclass
class CalibrationSample:
    device_kind: str
    m: int
    n: int
    k: int
    wall_sec: float
    kernel_gflops: float
    transfer_bytes: int = 0
    note: str = ""


@dataclass
class CalibrationReport:
    samples: list[CalibrationSample] = field(default_factory=list)
    fitted: dict[str, dict[str, float]] = field(default_factory=dict)
    mean_rel_error: float = 0.0
    max_rel_error: float = 0.0
    n_eval: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "mean_rel_error": self.mean_rel_error,
            "max_rel_error": self.max_rel_error,
            "n_eval": self.n_eval,
            "samples": [asdict(s) for s in self.samples],
        }


def set_calibration(models: dict[str, DeviceModel]) -> None:
    global _AUTOLOAD_OK
    _CALIBRATION.clear()
    _CALIBRATION.update(models)
    _AUTOLOAD_OK = bool(models)


def calibration_active() -> bool:
    return bool(_CALIBRATION)


def try_autoload_calibration(
    path: Optional[Union[str, Path]] = None,
    *,
    force: bool = False,
) -> bool:
    """Load calibration JSON once (unless force). Returns True if loaded."""
    global _AUTOLOAD_TRIED, _AUTOLOAD_OK
    if os.environ.get("UCT_NO_CALIBRATION", "").strip() in ("1", "true", "yes"):
        return False
    if _AUTOLOAD_TRIED and not force and _CALIBRATION:
        return _AUTOLOAD_OK
    if _AUTOLOAD_TRIED and not force and not _CALIBRATION:
        return False
    _AUTOLOAD_TRIED = True
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env = os.environ.get("UCT_CALIBRATION")
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[3]
    candidates.append(root / "artifacts" / "calibration.json")
    candidates.append(Path.cwd() / "artifacts" / "calibration.json")
    # shipped default (this machine class; override via UCT_CALIBRATION)
    candidates.append(
        Path(__file__).resolve().parents[1] / "config" / "calibration_default.json"
    )
    for p in candidates:
        if p.is_file():
            try:
                load_calibration(p)
                _AUTOLOAD_OK = True
                return True
            except Exception:
                continue
    _AUTOLOAD_OK = False
    return False


def get_device_model(kind_or_name: str) -> DeviceModel:
    if kind_or_name in _CALIBRATION:
        return _CALIBRATION[kind_or_name]
    base = "".join(c for c in kind_or_name if not c.isdigit())
    if base in _CALIBRATION:
        return _CALIBRATION[base]
    if kind_or_name in DEFAULT_MODELS:
        return DEFAULT_MODELS[kind_or_name]
    if base in DEFAULT_MODELS:
        return DEFAULT_MODELS[base]
    return DEFAULT_MODELS["ve"]


def calibrate_from_samples(
    samples: Sequence[CalibrationSample],
) -> CalibrationReport:
    """Fit peak_gflops / launch_overhead (and optional pcie) from measured walls.

    - peak: median of kernel_gflops when provided, else from flops/wall for large jobs
    - launch: residual wall - compute for small jobs (positive part)
    """
    by_kind: dict[str, list[CalibrationSample]] = {}
    for s in samples:
        by_kind.setdefault(s.device_kind, []).append(s)

    fitted: dict[str, dict[str, float]] = {}
    models: dict[str, DeviceModel] = {}
    for kind, ss in by_kind.items():
        base = DEFAULT_MODELS.get(kind, DEFAULT_MODELS["ve"])
        peaks = sorted(s.kernel_gflops for s in ss if s.kernel_gflops > 0)
        if peaks:
            # prefer high quantile (sustained large-N rate), not median of small tiles
            peak = peaks[max(0, (3 * len(peaks)) // 4)]
        else:
            # derive from large jobs: peak ≈ flops / wall
            derived = []
            for s in ss:
                if s.wall_sec > 0 and s.m * s.n * s.k >= 512**3:
                    flops = 2.0 * s.m * s.n * s.k
                    derived.append(flops / s.wall_sec / 1e9)
            peak = (
                sorted(derived)[len(derived) // 2]
                if derived
                else base.peak_gflops
            )
        pcie = base.pcie_gbps
        # launch residual on smaller problems
        residuals = []
        for s in ss:
            if s.wall_sec <= 0:
                continue
            flops = 2.0 * s.m * s.n * s.k
            compute = flops / (peak * 1e9) if peak > 0 else 0.0
            # rough xfer for non-host
            xfer = 0.0
            if kind != "host" and s.transfer_bytes > 0:
                xfer = estimate_h2d_seconds(s.transfer_bytes, pcie)
            elif kind != "host":
                nbytes = 8 * (s.m * s.k + s.k * s.n + s.m * s.n)
                xfer = estimate_h2d_seconds(nbytes, pcie)
            resid = s.wall_sec - compute - (0.0 if kind == "host" else 0.5 * xfer)
            if resid > 0:
                residuals.append(resid)
        if residuals:
            residuals.sort()
            launch = residuals[len(residuals) // 2]
            # clamp
            launch = float(min(max(launch, 0.0), 0.5 if kind != "host" else 0.05))
        else:
            launch = base.launch_overhead_sec
        if kind == "host":
            launch = min(launch, 0.002)
            pcie = 100.0
        model = DeviceModel(
            name=kind,
            kind=kind,
            peak_gflops=float(peak),
            pcie_gbps=float(pcie),
            launch_overhead_sec=float(launch),
        )
        models[kind] = model
        fitted[kind] = {
            "peak_gflops": model.peak_gflops,
            "pcie_gbps": model.pcie_gbps,
            "launch_overhead_sec": model.launch_overhead_sec,
        }

    set_calibration(models)

    errs: list[float] = []
    for s in samples:
        if s.device_kind == "host":
            md = get_device_model("host")
            flops = 2.0 * s.m * s.n * s.k
            pred_w = flops / (md.peak_gflops * 1e9) + md.launch_overhead_sec
        else:
            pred = estimate_gemm_placement(
                s.m,
                s.n,
                s.k,
                [f"{s.device_kind}1" if s.device_kind == "ve" else f"{s.device_kind}0"],
                strategy="row_blocks",
            )
            pred_w = pred.est_total_sec
        if s.wall_sec > 0 and pred_w > 0:
            errs.append(abs(pred_w - s.wall_sec) / s.wall_sec)

    report = CalibrationReport(
        samples=list(samples),
        fitted=fitted,
        mean_rel_error=sum(errs) / len(errs) if errs else 0.0,
        max_rel_error=max(errs) if errs else 0.0,
        n_eval=len(errs),
    )
    return report


def load_calibration(path: Union[str, Path]) -> CalibrationReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    models = {}
    for kind, vals in data.get("fitted", {}).items():
        models[kind] = DeviceModel(
            name=kind,
            kind=kind,
            peak_gflops=float(vals["peak_gflops"]),
            pcie_gbps=float(vals.get("pcie_gbps", DEFAULT_PCIE_GBPS)),
            launch_overhead_sec=float(vals.get("launch_overhead_sec", 0.005)),
        )
    set_calibration(models)
    return CalibrationReport(
        fitted=data.get("fitted", {}),
        mean_rel_error=float(data.get("mean_rel_error", 0.0)),
        max_rel_error=float(data.get("max_rel_error", 0.0)),
        n_eval=int(data.get("n_eval", 0)),
    )


def save_calibration(report: CalibrationReport, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass
class PlacementChoice:
    strategy: Strategy
    plan: PlacementPlan
    est_transfer_sec: float
    est_compute_sec: float
    est_total_sec: float
    transfer_bytes: int
    devices: list[str] = field(default_factory=list)
    power_watts: float = 0.0

    def attach_costs_to_plan(self) -> PlacementPlan:
        p = self.plan
        p.estimated_transfer_sec = self.est_transfer_sec
        p.estimated_compute_sec = self.est_compute_sec
        p.estimated_total_sec = self.est_total_sec
        p.transfer_bytes = self.transfer_bytes
        p.meta = {
            **(p.meta or {}),
            "devices": self.devices,
            "power_watts": self.power_watts,
            "strategy": self.strategy,
        }
        return p


def _kind_of(device: str) -> str:
    base = "".join(c for c in device if not c.isdigit())
    if base.startswith("ve"):
        return "ve"
    if base.startswith("phi"):
        return "phi"
    if base.startswith("host"):
        return "host"
    return "ve"


def _build_plan(
    m: int,
    n: int,
    k: int,
    devices: Sequence[str],
    strategy: Strategy,
    *,
    backend: str,
    dtype: str,
) -> PlacementPlan:
    atom = BACKEND_ATOMS.get(backend, "VE_NLC_DGEMM_64x64x64_F64")
    if strategy == "row_blocks":
        return partition_matrix_rows(
            m, n, devices, atom_name=atom, backend=backend, dtype=dtype, k=k
        )
    if strategy == "col_blocks":
        return partition_matrix_cols(
            m, n, devices, atom_name=atom, backend=backend, dtype=dtype, k=k
        )
    if strategy == "k_split":
        return partition_matrix_k(
            m, n, k, devices, atom_name=atom, backend=backend, dtype=dtype
        )
    raise ValueError(f"unknown strategy {strategy}")


def estimate_gemm_placement(
    m: int,
    n: int,
    k: int,
    devices: Sequence[str],
    *,
    strategy: Strategy = "row_blocks",
    dtype: str = "float64",
    backend: str = "VE_NLC",
    peak_override: Optional[float] = None,
    pcie_override: Optional[float] = None,
) -> PlacementChoice:
    """Estimate cost for a placement strategy (GEMM C=A@B)."""
    if not devices:
        raise ValueError("devices must be non-empty")
    dtype_bytes = DTYPE_BYTES.get(dtype, 8)
    plan = _build_plan(m, n, k, devices, strategy, backend=backend, dtype=dtype)

    # Per-kind models (use slowest peak among devices for conservative compute)
    models = [get_device_model(_kind_of(d)) for d in devices]
    peak = peak_override if peak_override is not None else min(md.peak_gflops for md in models)
    pcie = pcie_override if pcie_override is not None else min(md.pcie_gbps for md in models)
    n_dev = max(len(devices), 1)
    # pool cold-start scales with #workers; file staging per shard adds constant
    per_dev_launch = max(md.launch_overhead_sec for md in models)
    launch = per_dev_launch * (0.7 + 0.5 * n_dev)

    a_bytes = m * k * dtype_bytes
    b_bytes = k * n * dtype_bytes
    c_bytes = m * n * dtype_bytes
    # parallel efficiency drops slightly with more devices (sync + imbalance)
    eff = max(0.75, 1.0 - 0.06 * (n_dev - 1))

    if strategy == "row_blocks":
        # share_b: full B once + all A shards + all C
        xfer = a_bytes + b_bytes + c_bytes
        flops = 2.0 * m * n * k
        compute = flops / (n_dev * peak * 1e9 * eff)
    elif strategy == "col_blocks":
        xfer = a_bytes + b_bytes + c_bytes
        flops = 2.0 * m * n * k
        compute = flops / (n_dev * peak * 1e9 * eff)
    else:  # k_split — C reduce traffic grows with n_dev
        xfer = a_bytes + b_bytes + c_bytes * n_dev
        flops = 2.0 * m * n * k
        compute = flops / (n_dev * peak * 1e9 * eff)
        compute += (m * n * dtype_bytes * n_dev) / (pcie * (1024**3) * 0.5)

    transfer = estimate_h2d_seconds(xfer, pcie) + launch
    total = max(transfer, compute) + 0.35 * min(transfer, compute)

    # Prefer row when shapes equal (share_b path is well-tuned); k_split rarer
    if strategy == "k_split":
        total *= 1.08
    if strategy == "col_blocks" and m >= n * 1.5:
        total *= 0.97  # tall-skinny: col panels can help B locality model
    if strategy == "row_blocks" and n >= m * 1.5:
        total *= 0.97

    return PlacementChoice(
        strategy=strategy,
        plan=plan,
        est_transfer_sec=transfer,
        est_compute_sec=compute,
        est_total_sec=total,
        transfer_bytes=xfer,
        devices=list(devices),
    )


def estimate_power_for_devices(devices: Sequence[str], op: str = "dgemm") -> float:
    from uni_cute_tensor.power import estimate_power

    return sum(estimate_power(d, op) for d in devices)


def choose_best_placement(
    m: int,
    n: int,
    k: int,
    devices: Sequence[str],
    *,
    strategies: Optional[Sequence[Strategy]] = None,
    backend: str = "VE_NLC",
    dtype: str = "float64",
    power_cap: Any = None,
    max_devices: Optional[int] = None,
    min_devices: int = 1,
    attach: bool = True,
    autoload_calibration: bool = True,
) -> PlacementChoice:
    """Pick strategy + device subset minimizing estimated total time under PowerCap."""
    if autoload_calibration:
        try_autoload_calibration()
    devices = list(devices)
    if not devices:
        raise ValueError("devices must be non-empty")
    strategies = list(strategies or ("row_blocks", "col_blocks", "k_split"))
    max_d = max_devices or len(devices)
    max_d = min(max_d, len(devices))
    min_d = max(1, min(min_devices, max_d))

    candidates: list[PlacementChoice] = []
    for n_dev in range(min_d, max_d + 1):
        # contiguous prefix subsets (VE numbering order) + all combos for small n
        subsets: list[tuple[str, ...]]
        if len(devices) <= 4:
            subsets = list(itertools.combinations(devices, n_dev))
        else:
            subsets = [tuple(devices[:n_dev])]

        for subset in subsets:
            if power_cap is not None:
                ops = {d: "dgemm" for d in subset}
                try:
                    if not power_cap.can_launch(list(subset), ops):
                        continue
                except Exception:
                    pass
            watts = estimate_power_for_devices(subset)
            for strat in strategies:
                if strat == "k_split" and k < n_dev * 32:
                    continue  # too thin K panels
                try:
                    ch = estimate_gemm_placement(
                        m, n, k, subset, strategy=strat, backend=backend, dtype=dtype
                    )
                except ValueError:
                    continue
                ch.power_watts = watts
                candidates.append(ch)

    if not candidates:
        # fallback: single first device row
        ch = estimate_gemm_placement(
            m, n, k, devices[:1], strategy="row_blocks", backend=backend, dtype=dtype
        )
        ch.power_watts = estimate_power_for_devices(devices[:1])
        if attach:
            ch.attach_costs_to_plan()
        return ch

    best = min(candidates, key=lambda c: (c.est_total_sec, c.power_watts, -len(c.devices)))
    if attach:
        best.attach_costs_to_plan()
    return best


def prediction_error_report(
    measured_wall: float,
    choice: PlacementChoice,
) -> dict[str, float]:
    pred = choice.est_total_sec
    abs_err = abs(pred - measured_wall)
    rel = abs_err / measured_wall if measured_wall > 0 else float("inf")
    return {
        "measured_wall": measured_wall,
        "predicted_wall": pred,
        "abs_error": abs_err,
        "rel_error": rel,
    }
