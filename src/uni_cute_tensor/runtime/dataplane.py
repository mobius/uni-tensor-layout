"""Unified DataPlane: run GEMM on Host / file-worker / AVEO without app changes."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from uni_cute_tensor.runtime.timeline import get_timeline

BackendName = Literal["host", "file", "worker", "aveo", "aveo_pinned"]


@dataclass
class GemmRequest:
    a: np.ndarray
    b: np.ndarray
    device: str = "host"  # host | ve1 | ve2 | ve3 | phi0
    name: str = "gemm"


@dataclass
class GemmResult:
    c: np.ndarray
    backend: str
    device: str
    wall_sec: float
    kernel_gflops: float
    max_abs_err: float
    status: str
    mode: str = ""


class DataPlane(ABC):
    name: str

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def gemm(self, req: GemmRequest) -> GemmResult: ...

    def __enter__(self) -> "DataPlane":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


class HostPlane(DataPlane):
    name = "host"

    def __init__(self, *, backend: str = "auto"):
        self.backend = backend

    def open(self) -> None:
        return

    def close(self) -> None:
        return

    def gemm(self, req: GemmRequest) -> GemmResult:
        from uni_cute_tensor.backends.host_dgemm import host_dgemm

        tl = get_timeline()
        t0 = time.perf_counter()
        with tl.span("host_dgemm", "kernel", device="host") if tl else _nullspan():
            c, r = host_dgemm(req.a, req.b, backend=self.backend)
        wall = time.perf_counter() - t0
        return GemmResult(
            c=c,
            backend=self.name,
            device="host",
            wall_sec=wall,
            kernel_gflops=r.gflops,
            max_abs_err=r.max_abs_err,
            status=r.status,
            mode=r.atom_name,
        )


class _nullspan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FilePlane(DataPlane):
    """One-shot ve_exec file staging (share_b style single device or multi via multi_ve)."""

    name = "file"

    def __init__(self, devices: Optional[Sequence[str]] = None):
        self.devices = list(devices) if devices else []

    def open(self) -> None:
        from uni_cute_tensor.backends.ve_dgemm import compile_ve_dgemm

        compile_ve_dgemm()

    def close(self) -> None:
        return

    def gemm(self, req: GemmRequest) -> GemmResult:
        from uni_cute_tensor.backends.ve_dgemm import multi_ve_layout_dgemm, run_ve_dgemm_shard
        from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names

        tl = get_timeline()
        devices = self.devices or ve_device_names(discover_devices())
        t0 = time.perf_counter()
        if len(devices) <= 1:
            ve_id = 1
            if devices and devices[0].startswith("ve"):
                ve_id = int(devices[0].replace("ve", ""))
            with tl.span("file_ve_gemm", "kernel", device=f"ve{ve_id}") if tl else _nullspan():
                c, r = run_ve_dgemm_shard(req.a, req.b, ve_id=ve_id)
            wall = time.perf_counter() - t0
            return GemmResult(
                c=c,
                backend=self.name,
                device=f"ve{ve_id}",
                wall_sec=wall,
                kernel_gflops=r.gflops,
                max_abs_err=r.max_abs_err,
                status=r.status,
                mode=r.mode,
            )
        with tl.span("file_multi_ve", "kernel", device=",".join(devices)) if tl else _nullspan():
            c, plan, results = multi_ve_layout_dgemm(
                req.a, req.b, devices, share_b=True
            )
        wall = time.perf_counter() - t0
        err = float(np.max(np.abs(c - req.a @ req.b)))
        kg = sum(r.gflops for r in results) / max(len(results), 1)
        return GemmResult(
            c=c,
            backend=self.name,
            device=",".join(devices),
            wall_sec=wall,
            kernel_gflops=kg,
            max_abs_err=err,
            status="pass" if err < 1e-8 else "fail",
            mode="multi-share_b",
        )


class WorkerPlane(DataPlane):
    name = "worker"

    def __init__(self, devices: Optional[Sequence[str]] = None):
        self.devices = list(devices) if devices else []
        self._pool = None

    def open(self) -> None:
        from uni_cute_tensor.backends.ve_worker import VeWorkerPool
        from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names

        devices = self.devices or ve_device_names(discover_devices())
        self.devices = list(devices)
        ve_ids = [int(d.replace("ve", "")) for d in self.devices]
        self._pool = VeWorkerPool(ve_ids)
        self._pool.start()

    def close(self) -> None:
        if self._pool:
            self._pool.stop()
            self._pool = None

    def gemm(self, req: GemmRequest) -> GemmResult:
        from uni_cute_tensor.backends.ve_worker import multi_ve_layout_dgemm_pooled

        assert self._pool is not None
        tl = get_timeline()
        t0 = time.perf_counter()
        with tl.span("worker_multi_ve", "kernel", device=",".join(self.devices)) if tl else _nullspan():
            c, plan, results = multi_ve_layout_dgemm_pooled(
                req.a, req.b, self.devices, self._pool, share_b=True
            )
        wall = time.perf_counter() - t0
        err = float(np.max(np.abs(c - req.a @ req.b)))
        kg = sum(r.gflops for r in results) / max(len(results), 1)
        return GemmResult(
            c=c,
            backend=self.name,
            device=",".join(self.devices),
            wall_sec=wall,
            kernel_gflops=kg,
            max_abs_err=err,
            status="pass" if err < 1e-8 else "fail",
            mode="pooled",
        )


class AveoPlane(DataPlane):
    """AVEO multi-VE; optional pin for resident buffers (single-node gemm_pinned)."""

    name = "aveo"

    def __init__(
        self,
        devices: Optional[Sequence[str]] = None,
        *,
        pin: bool = False,
        pin_m: int = 2048,
        pin_n: int = 2048,
        pin_k: int = 2048,
    ):
        self.devices = list(devices) if devices else []
        self.pin = pin
        self.pin_m, self.pin_n, self.pin_k = pin_m, pin_n, pin_k
        self._pool = None

    def open(self) -> None:
        from uni_cute_tensor.backends.ve_aveo import AveoSessionPool
        from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names

        devices = self.devices or ve_device_names(discover_devices())
        self.devices = list(devices)
        nodes = [int(d.replace("ve", "")) for d in self.devices]
        self._pool = AveoSessionPool(nodes)
        self._pool.start()
        if self.pin:
            self._pool.pin_all(self.pin_m, self.pin_n, self.pin_k)

    def close(self) -> None:
        if self._pool:
            self._pool.stop()
            self._pool = None

    def gemm(self, req: GemmRequest) -> GemmResult:
        from uni_cute_tensor.backends.ve_aveo import multi_ve_aveo_dgemm

        assert self._pool is not None
        tl = get_timeline()
        t0 = time.perf_counter()
        # single device + pin path
        if self.pin and len(self.devices) == 1:
            ve_id = int(self.devices[0].replace("ve", ""))
            with tl.span("aveo_pinned", "kernel", device=f"ve{ve_id}") if tl else _nullspan():
                c, r = self._pool.dgemm_pinned(ve_id, req.a, req.b)
            wall = time.perf_counter() - t0
            return GemmResult(
                c=c,
                backend="aveo_pinned",
                device=f"ve{ve_id}",
                wall_sec=wall,
                kernel_gflops=r.gflops,
                max_abs_err=r.max_abs_err,
                status=r.status,
                mode=r.mode,
            )
        with tl.span("aveo_multi", "kernel", device=",".join(self.devices)) if tl else _nullspan():
            c, plan, results = multi_ve_aveo_dgemm(
                req.a, req.b, self.devices, pool=self._pool
            )
        wall = time.perf_counter() - t0
        err = float(np.max(np.abs(c - req.a @ req.b)))
        kg = sum(r.gflops for r in results) / max(len(results), 1)
        return GemmResult(
            c=c,
            backend=self.name,
            device=",".join(self.devices),
            wall_sec=wall,
            kernel_gflops=kg,
            max_abs_err=err,
            status="pass" if err < 1e-8 else "fail",
            mode="multi-aveo",
        )


def create_dataplane(
    backend: BackendName,
    *,
    devices: Optional[Sequence[str]] = None,
    **kwargs,
) -> DataPlane:
    if backend == "host":
        return HostPlane(backend=kwargs.get("host_backend", "auto"))
    if backend == "file":
        return FilePlane(devices)
    if backend == "worker":
        return WorkerPlane(devices)
    if backend in ("aveo", "aveo_pinned"):
        pin = backend == "aveo_pinned" or bool(kwargs.get("pin", False))
        return AveoPlane(
            devices,
            pin=pin,
            pin_m=int(kwargs.get("pin_m", 2048)),
            pin_n=int(kwargs.get("pin_n", 2048)),
            pin_k=int(kwargs.get("pin_k", 2048)),
        )
    raise ValueError(f"unknown dataplane backend: {backend}")
