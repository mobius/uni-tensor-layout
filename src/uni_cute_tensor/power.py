"""Power budget gate — thin local model + optional uni PowerCap import."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_POWER: dict[str, dict[str, float]] = {
    "phi": {"dgemm": 280.0, "scale": 150.0, "idle": 50.0, "fma_peak": 280.0},
    "ve": {"dgemm": 280.0, "scale": 150.0, "idle": 50.0, "fma_peak": 280.0},
    "host": {"dgemm": 200.0, "scale": 100.0, "idle": 50.0, "fma_peak": 300.0},
}


def _kind(device: str) -> str:
    base = "".join(c for c in device if not c.isdigit())
    return base if base in _POWER else "host"


def estimate_power(device: str, op: str = "dgemm") -> float:
    model = _POWER[_kind(device)]
    return model.get(op, model["idle"])


def try_import_uni_power():
    """Return uni PowerCap class if UNI_ROOT/scheduler available."""
    roots = []
    env = os.environ.get("UNI_ROOT")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / "Work" / "uni")
    for root in roots:
        sched = root / "src"
        if (sched / "scheduler" / "power.py").is_file():
            if str(sched) not in sys.path:
                sys.path.insert(0, str(sched))
            try:
                from scheduler.power import PowerCap as UniPowerCap  # type: ignore

                return UniPowerCap
            except Exception:
                continue
    return None


@dataclass
class PowerCap:
    """Local power budget (same defaults as uni: 1600W * 0.9)."""

    psu_limit_w: float = 1600.0
    safety_margin: float = 0.90
    _reserved: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.effective_limit = self.psu_limit_w * self.safety_margin
        Uni = try_import_uni_power()
        self._uni = None
        if Uni is not None:
            try:
                self._uni = Uni(
                    psu_limit_w=self.psu_limit_w, safety_margin=self.safety_margin
                )
            except Exception:
                self._uni = None

    @property
    def backend(self) -> str:
        return "uni" if self._uni is not None else "local"

    def reserved_watts(self) -> float:
        if self._uni is not None:
            # best-effort
            try:
                return sum(
                    getattr(r, "watts", 0.0)
                    for r in getattr(self._uni, "_reservations", {}).values()
                )
            except Exception:
                pass
        return sum(self._reserved.values())

    def can_launch(self, devices: list[str], ops: Optional[dict[str, str]] = None) -> bool:
        ops = ops or {}
        if self._uni is not None:
            try:
                # uni API: can_launch(devices, ops=...) or op_weights
                return bool(self._uni.can_launch(devices, ops=ops))
            except TypeError:
                try:
                    return bool(self._uni.can_launch(devices, ops))
                except Exception:
                    pass
            except Exception:
                pass
        add = sum(estimate_power(d, ops.get(d, "dgemm")) for d in devices)
        return self.reserved_watts() + add <= self.effective_limit

    def reserve(self, devices: list[str], ops: Optional[dict[str, str]] = None) -> None:
        ops = ops or {}
        if self._uni is not None:
            try:
                self._uni.reserve(devices, ops=ops)
                return
            except TypeError:
                try:
                    self._uni.reserve(devices, ops)
                    return
                except Exception:
                    pass
            except Exception:
                pass
        for d in devices:
            self._reserved[d] = estimate_power(d, ops.get(d, "dgemm"))

    def release(self, devices: list[str]) -> None:
        if self._uni is not None:
            try:
                self._uni.release(devices)
                return
            except Exception:
                pass
        for d in devices:
            self._reserved.pop(d, None)

    def guard(self, devices: list[str], ops: Optional[dict[str, str]] = None):
        """Context manager style helper."""
        cap = self

        class _Guard:
            def __enter__(self_inner):
                if not cap.can_launch(devices, ops):
                    raise RuntimeError(
                        f"PowerCap blocked launch {devices} "
                        f"(reserved={cap.reserved_watts():.0f}W "
                        f"limit={cap.effective_limit:.0f}W backend={cap.backend})"
                    )
                cap.reserve(devices, ops)
                return cap

            def __exit__(self_inner, *args):
                cap.release(devices)

        return _Guard()
