"""Best-effort power sampling for sustained benches."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def read_power_watts() -> Optional[float]:
    """Try ipmitool PSU power sum; None if unavailable."""
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
            if "Power" in line and "Watts" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    try:
                        total += float(parts[1].strip())
                        found = True
                    except ValueError:
                        pass
        return total if found else None
    except Exception:
        return None


@dataclass
class PowerSampler:
    interval_sec: float = 0.5
    samples: list[tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thr: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()

        def _loop():
            t0 = time.time()
            while not self._stop.is_set():
                w = read_power_watts()
                if w is not None:
                    self.samples.append((time.time() - t0, w))
                time.sleep(self.interval_sec)

        self._thr = threading.Thread(target=_loop, daemon=True)
        self._thr.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=5)
        if not self.samples:
            return {"n": 0, "mean_w": None, "max_w": None, "min_w": None}
        vals = [w for _, w in self.samples]
        return {
            "n": len(vals),
            "mean_w": sum(vals) / len(vals),
            "max_w": max(vals),
            "min_w": min(vals),
            "samples": self.samples[:50],  # cap for logs
        }
