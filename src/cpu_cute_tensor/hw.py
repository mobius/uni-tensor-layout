"""Hardware capability probe (no secrets: no serial numbers in output)."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HwReport:
    hostname: str
    machine: str
    cpu_model: str
    cpu_flags_key: list[str]
    has_nvidia: bool
    has_amx: bool
    phi_online: bool
    ve_nodes: list[str]
    ncc_version: str
    python_version: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def layout_algebra_ok(self) -> bool:
        return True  # pure Python path

    @property
    def hetero_ok(self) -> bool:
        return self.phi_online or len(self.ve_nodes) > 0


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (r.stdout or r.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def probe() -> HwReport:
    flags = _cpu_flags()
    key = sorted(
        f
        for f in flags
        if f.startswith("avx") or f.startswith("amx") or f in {"fma", "vnni"}
    )
    phi_online = False
    if Path("/dev/mic0").exists():
        out = _run(["micctrl", "-s"])
        phi_online = "online" in out.lower() or True  # device node is strong signal

    ve_nodes: list[str] = []
    for p in sorted(Path("/dev").glob("ve[0-9]*")):
        if p.name[2:].isdigit():
            ve_nodes.append(p.name)

    ncc = _run(["ncc", "--version"]).splitlines()
    ncc_version = ncc[0] if ncc else "missing"

    has_nvidia = bool(_run(["nvidia-smi", "-L"]))
    notes: list[str] = []
    if not has_nvidia:
        notes.append("No NVIDIA GPU: atoms_nv is educational only")
    if "amx_tile" not in flags and not any(f.startswith("amx") for f in flags):
        notes.append("No AMX: atoms_amx educational only")
    if not ve_nodes:
        notes.append("No /dev/ve*: VE backends unavailable")
    if not Path("/dev/mic0").exists():
        notes.append("No /dev/mic0: Phi backends unavailable")

    return HwReport(
        hostname=platform.node(),
        machine=platform.machine(),
        cpu_model=_cpu_model(),
        cpu_flags_key=key,
        has_nvidia=has_nvidia,
        has_amx=any(f.startswith("amx") for f in flags),
        phi_online=phi_online and Path("/dev/mic0").exists(),
        ve_nodes=ve_nodes,
        ncc_version=ncc_version,
        python_version=platform.python_version(),
        notes=notes,
    )
