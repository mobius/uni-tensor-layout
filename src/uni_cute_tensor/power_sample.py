"""Multi-source power sampling: RAPL, VE sensors, ipmitool (Phase 3 M3 / W4.1).

Degrades cleanly when a source is missing — never raises from sample loop.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _read_rapl_energy_uj() -> Optional[float]:
    """Sum package energy_uj across intel-rapl domains (package-* only)."""
    root = Path("/sys/class/powercap")
    if not root.is_dir():
        return None
    total = 0.0
    found = False
    for d in sorted(root.glob("intel-rapl:*")):
        # skip nested dram under package if we only want package: match name
        name_p = d / "name"
        energy_p = d / "energy_uj"
        if not energy_p.is_file():
            continue
        name = name_p.read_text(encoding="utf-8").strip() if name_p.is_file() else d.name
        # include package-N and top-level; include dram as optional via env
        include_dram = os.environ.get("UCT_RAPL_DRAM", "").strip() in ("1", "true", "yes")
        if name.startswith("dram") and not include_dram:
            continue
        try:
            total += float(energy_p.read_text(encoding="utf-8").strip())
            found = True
        except (OSError, ValueError):
            continue
    return total if found else None


def read_rapl_watts(prev: Optional[tuple[float, float]]) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """Return (watts, new_state). prev is (energy_uj, monotonic_time)."""
    e = _read_rapl_energy_uj()
    now = time.monotonic()
    if e is None:
        return None, None
    if prev is None:
        return None, (e, now)
    de = e - prev[0]
    dt = now - prev[1]
    if dt <= 0:
        return None, (e, now)
    # energy_uj is microjoules → watts = (J) / s = (uj * 1e-6) / dt
    if de < 0:
        # counter wrap
        return None, (e, now)
    w = (de * 1e-6) / dt
    return w, (e, now)


def read_ve_sensor_watts() -> Optional[float]:
    """Best-effort sum of VE card power from sysfs sensors.

    NEC VE sensor indices vary by firmware. Override with
    UCT_VE_POWER_SENSORS=ve0:15,ve0:16 (sensor ids) or leave default heuristic:
    prefer sensors named via optional power_mw files; else None if unclear.
    """
    env = os.environ.get("UCT_VE_POWER_SENSORS", "").strip()
    ve_root = Path("/sys/class/ve")
    if not ve_root.is_dir():
        return None
    total = 0.0
    found = False
    if env:
        for part in env.split(","):
            part = part.strip()
            if not part:
                continue
            # format ve0:12 or ve0/sensor_12
            if ":" in part:
                ve, sid = part.split(":", 1)
                path = ve_root / ve / f"sensor_{sid}"
            else:
                path = Path(part)
            try:
                raw = float(path.read_text(encoding="utf-8").strip())
                # values often in uW or mW — if large, treat as uW
                w = raw / 1e6 if raw > 1e5 else raw / 1e3 if raw > 1e3 else raw
                total += w
                found = True
            except (OSError, ValueError):
                continue
        return total if found else None

    # Heuristic: some stacks expose sensor with power in mW around 1e5–5e5
    # We do NOT guess blindly (wrong index misleads). Return None without env.
    return None


def read_ipmitool_watts() -> Optional[float]:
    """Try ipmitool PSU / Power sensors sum."""
    if not shutil.which("ipmitool"):
        return None
    try:
        r = subprocess.run(
            ["ipmitool", "sensor"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode != 0:
            return None
        total = 0.0
        found = False
        for line in r.stdout.splitlines():
            low = line.lower()
            if "power" not in low and "watt" not in low:
                continue
            if "status" in low and "ok" not in low:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            # skip non-numeric
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", parts[1])
            if not m:
                continue
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            # filter absurd
            if val <= 0 or val > 20000:
                continue
            total += val
            found = True
        return total if found else None
    except Exception:
        return None


def read_power_watts() -> Optional[float]:
    """Backward-compatible: prefer ipmitool, else None (RAPL needs delta)."""
    return read_ipmitool_watts()


@dataclass
class PowerSample:
    t: float
    watts: Optional[float]
    source: str  # rapl | ve | ipmitool | none
    components: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class PowerSampler:
    """Background multi-source sampler.

    Sources attempted each tick: RAPL (diff), VE sensors (if configured), ipmitool.
    ``watts`` is the best available aggregate for that sample.
    """

    interval_sec: float = 0.5
    samples: list[PowerSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thr: Optional[threading.Thread] = None
    _rapl_state: Optional[tuple[float, float]] = field(default=None, repr=False)
    sources_used: list[str] = field(default_factory=list)

    def start(self) -> None:
        self.samples.clear()
        self.sources_used.clear()
        self._stop.clear()
        self._rapl_state = None

        def _loop():
            t0 = time.time()
            # prime RAPL
            _, self._rapl_state = read_rapl_watts(None)
            time.sleep(min(0.2, self.interval_sec))
            while not self._stop.is_set():
                comps: dict[str, Optional[float]] = {}
                rapl_w, self._rapl_state = read_rapl_watts(self._rapl_state)
                comps["rapl"] = rapl_w
                ve_w = read_ve_sensor_watts()
                comps["ve"] = ve_w
                ipmi_w = read_ipmitool_watts()
                comps["ipmitool"] = ipmi_w

                # prefer chassis ipmitool total, else rapl package sum, else ve
                if ipmi_w is not None:
                    w, src = ipmi_w, "ipmitool"
                elif rapl_w is not None:
                    w, src = rapl_w, "rapl"
                elif ve_w is not None:
                    w, src = ve_w, "ve"
                else:
                    w, src = None, "none"

                if src != "none" and src not in self.sources_used:
                    self.sources_used.append(src)
                self.samples.append(
                    PowerSample(t=time.time() - t0, watts=w, source=src, components=comps)
                )
                time.sleep(self.interval_sec)

        self._thr = threading.Thread(target=_loop, daemon=True)
        self._thr.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=5)
        vals = [s.watts for s in self.samples if s.watts is not None]
        by_src: dict[str, list[float]] = {}
        for s in self.samples:
            if s.watts is None:
                continue
            by_src.setdefault(s.source, []).append(s.watts)

        def _stats(xs: list[float]) -> dict[str, Any]:
            return {
                "n": len(xs),
                "mean_w": sum(xs) / len(xs) if xs else None,
                "max_w": max(xs) if xs else None,
                "min_w": min(xs) if xs else None,
            }

        out: dict[str, Any] = {
            "n": len(vals),
            "mean_w": sum(vals) / len(vals) if vals else None,
            "max_w": max(vals) if vals else None,
            "min_w": min(vals) if vals else None,
            "sources_used": list(self.sources_used),
            "by_source": {k: _stats(v) for k, v in by_src.items()},
            "degrade": None if vals else "no power source (RAPL/ipmitool/VE env)",
            "samples": [
                {"t": s.t, "w": s.watts, "src": s.source} for s in self.samples[:80]
            ],
        }
        return out


def probe_power_sources() -> dict[str, Any]:
    """One-shot capability report for docs/CLI."""
    _, st = read_rapl_watts(None)
    time.sleep(0.15)
    rapl_w, _ = read_rapl_watts(st)
    return {
        "rapl": rapl_w is not None or _read_rapl_energy_uj() is not None,
        "rapl_sample_w": rapl_w,
        "ve_sensors": read_ve_sensor_watts() is not None,
        "ve_note": "set UCT_VE_POWER_SENSORS=ve0:N,... for VE wattage",
        "ipmitool": shutil.which("ipmitool") is not None,
        "ipmitool_sample_w": read_ipmitool_watts(),
    }
