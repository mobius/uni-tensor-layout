"""Adapter between PlacementPlan and uni-framework device discovery.

Does not import uni at module import time. Discovery works standalone via
sysfs/lspci; optional UNI_ROOT enables richer scheduler integration later.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from cpu_cute_tensor.partition.multi_device import PlacementPlan


@dataclass(frozen=True)
class DeviceSummary:
    name: str
    kind: str  # host | phi | ve
    online: bool
    ve_id: Optional[int] = None
    numa_node: int = -1
    source: str = "local"  # local | uni


def resolve_uni_root() -> Optional[Path]:
    """Return UNI_ROOT if set and valid, else well-known local path if present."""
    env = os.environ.get("UNI_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    # Common local checkout (no secrets). Not required for standalone mode.
    candidates.append(Path.home() / "Work" / "uni")
    for c in candidates:
        if (c / "src" / "scheduler").is_dir():
            return c.resolve()
    return None


def _discover_phi_local() -> Optional[DeviceSummary]:
    if not Path("/dev/mic0").exists():
        return None
    online = True
    try:
        r = subprocess.run(
            ["micctrl", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        online = "online" in (r.stdout or "").lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return DeviceSummary(name="phi0", kind="phi", online=online, source="local")


def _discover_ve_local() -> list[DeviceSummary]:
    devices: list[DeviceSummary] = []
    ve_root = Path("/sys/class/ve")
    if not ve_root.is_dir():
        # fallback: /dev/veN
        for i in range(8):
            if Path(f"/dev/ve{i}").exists():
                devices.append(
                    DeviceSummary(
                        name=f"ve{i + 1}",
                        kind="ve",
                        online=True,
                        ve_id=i + 1,
                        source="local",
                    )
                )
        return devices

    for entry in sorted(ve_root.iterdir()):
        name = entry.name
        if not name.startswith("ve") or not name[2:].isdigit():
            continue
        sysfs_id = int(name[2:])
        ve_n = sysfs_id + 1
        devices.append(
            DeviceSummary(
                name=f"ve{ve_n}",
                kind="ve",
                online=True,
                ve_id=ve_n,
                source="local",
            )
        )
    return devices


def _discover_via_uni(uni_root: Path) -> list[DeviceSummary]:
    import sys

    sched = str(uni_root / "src")
    if sched not in sys.path:
        sys.path.insert(0, sched)
    from scheduler.devices import discover_all  # type: ignore

    out: list[DeviceSummary] = []
    for d in discover_all():
        out.append(
            DeviceSummary(
                name=d.name,
                kind=d.kind,
                online=bool(d.online),
                ve_id=getattr(d, "ve_id", None),
                numa_node=getattr(d, "numa_node", -1),
                source="uni",
            )
        )
    return out


def discover_devices(*, prefer_uni: bool = True) -> list[DeviceSummary]:
    """Discover Phi + VE. Prefer uni if UNI_ROOT available; always include host."""
    found: list[DeviceSummary] = [
        DeviceSummary(name="host", kind="host", online=True, source="local"),
    ]
    uni = resolve_uni_root() if prefer_uni else None
    if uni is not None:
        try:
            found.extend(_discover_via_uni(uni))
            return found
        except Exception:
            pass
    phi = _discover_phi_local()
    if phi:
        found.append(phi)
    found.extend(_discover_ve_local())
    return found


def plan_to_task_specs(plan: PlacementPlan, op: str = "dgemm") -> list[dict[str, Any]]:
    """Convert PlacementPlan shards into serializable task specs for a DAG."""
    specs: list[dict[str, Any]] = []
    for i, shard in enumerate(plan.shards):
        specs.append(
            {
                "name": f"{op}_{shard.device}_{i}",
                "device": shard.device,
                "op": op,
                "row_start": shard.row_start,
                "row_end": shard.row_end,
                "col_start": shard.col_start,
                "col_end": shard.col_end,
                "layout": str(shard.layout),
                "atom_name": shard.atom_name,
                "depends_on": ["prepare"] if i == 0 else [f"{op}_{plan.shards[i - 1].device}_{i - 1}"],
            }
        )
    # Independent shards: clear serial depends for parallel VE work
    for s in specs:
        s["depends_on"] = ["prepare"]
    return specs


def ve_device_names(devices: Sequence[DeviceSummary], *, online_only: bool = True) -> list[str]:
    names = []
    for d in devices:
        if d.kind != "ve":
            continue
        if online_only and not d.online:
            continue
        names.append(d.name)
    return names
