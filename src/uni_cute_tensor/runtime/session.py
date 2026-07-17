"""Process-wide shared VE sessions (AVEO pin / worker pool).

Phase 3 M2: avoid open/close per job inside one process (E5 / uct-run / execute_plan).
"""

from __future__ import annotations

import atexit
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

_lock = threading.RLock()


@dataclass
class _AveoEntry:
    nodes: tuple[int, ...]
    pool: Any
    pin_m: int = 0
    pin_n: int = 0
    pin_k: int = 0


@dataclass
class _PoolEntry:
    ve_ids: tuple[int, ...]
    pool: Any


_aveo: Optional[_AveoEntry] = None
_worker: Optional[_PoolEntry] = None
_closed = False


def _ensure_env() -> None:
    import os

    os.environ.setdefault(
        "VE_LD_LIBRARY_PATH",
        "/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        f"/opt/nec/ve/veos/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    )


def get_aveo_pool(
    nodes: Sequence[int],
    *,
    pin_m: int = 0,
    pin_n: int = 0,
    pin_k: int = 0,
    pin_mode: str = "grow",
):
    """Return shared AveoSessionPool; grow or reject pin capacity.

    pin_mode:
      - grow (default): re-pin to max(current, requested)
      - strict: raise if request exceeds existing pin (service preload guard)
    """
    global _aveo
    _ensure_env()
    nodes_t = tuple(sorted(int(n) for n in nodes))
    if not nodes_t:
        raise ValueError("nodes must be non-empty")
    mode = (pin_mode or "grow").lower()
    with _lock:
        if _aveo is not None and _aveo.nodes == nodes_t:
            if pin_m > 0 and (
                pin_m > _aveo.pin_m or pin_n > _aveo.pin_n or pin_k > _aveo.pin_k
            ):
                if mode == "strict" and _aveo.pin_m > 0:
                    raise RuntimeError(
                        f"pin capacity exceeded: need ({pin_m},{pin_n},{pin_k}) "
                        f"have ({_aveo.pin_m},{_aveo.pin_n},{_aveo.pin_k}); "
                        f"restart uct-serve with larger --preload"
                    )
                pm = max(pin_m, _aveo.pin_m)
                pn = max(pin_n, _aveo.pin_n)
                pk = max(pin_k, _aveo.pin_k)
                _aveo.pool.pin_all(pm, pn, pk)
                _aveo.pin_m, _aveo.pin_n, _aveo.pin_k = pm, pn, pk
            return _aveo.pool
        # replace
        if _aveo is not None:
            try:
                _aveo.pool.stop()
            except Exception:
                pass
            _aveo = None
        from uni_cute_tensor.backends.ve_aveo import AveoSessionPool

        pool = AveoSessionPool(list(nodes_t))
        pool.start()
        if pin_m > 0 and pin_n > 0 and pin_k > 0:
            pool.pin_all(pin_m, pin_n, pin_k)
        _aveo = _AveoEntry(
            nodes=nodes_t, pool=pool, pin_m=pin_m, pin_n=pin_n, pin_k=pin_k
        )
        return pool


def get_worker_pool(ve_ids: Sequence[int]):
    """Return shared VeWorkerPool for given ve_ids."""
    global _worker
    _ensure_env()
    ids = tuple(sorted(int(v) for v in ve_ids))
    if not ids:
        raise ValueError("ve_ids must be non-empty")
    with _lock:
        if _worker is not None and _worker.ve_ids == ids:
            return _worker.pool
        if _worker is not None:
            try:
                _worker.pool.stop()
            except Exception:
                pass
            _worker = None
        from uni_cute_tensor.backends.ve_worker import VeWorkerPool
        from uni_cute_tensor.partition.runner import set_shared_ve_pool

        pool = VeWorkerPool(list(ids))
        pool.start()
        set_shared_ve_pool(pool)
        _worker = _PoolEntry(ve_ids=ids, pool=pool)
        return pool


def has_aveo_session() -> bool:
    with _lock:
        return _aveo is not None


def has_worker_pool() -> bool:
    with _lock:
        return _worker is not None


def session_health() -> dict:
    """Lightweight health snapshot for uct-serve."""
    with _lock:
        info: dict = {
            "aveo": False,
            "aveo_nodes": [],
            "aveo_pin": None,
            "worker": False,
            "worker_ids": [],
            "ok": True,
            "detail": "",
        }
        if _aveo is not None:
            info["aveo"] = True
            info["aveo_nodes"] = list(_aveo.nodes)
            info["aveo_pin"] = {
                "m": _aveo.pin_m,
                "n": _aveo.pin_n,
                "k": _aveo.pin_k,
            }
            # pool object presence
            if _aveo.pool is None:
                info["ok"] = False
                info["detail"] = "aveo pool missing"
        if _worker is not None:
            info["worker"] = True
            info["worker_ids"] = list(_worker.ve_ids)
            if _worker.pool is None:
                info["ok"] = False
                info["detail"] = "worker pool missing"
        return info


def shutdown_sessions() -> None:
    """Stop all shared sessions (safe to call multiple times)."""
    global _aveo, _worker, _closed
    with _lock:
        if _aveo is not None:
            try:
                _aveo.pool.stop()
            except Exception:
                pass
            _aveo = None
        if _worker is not None:
            try:
                from uni_cute_tensor.partition.runner import clear_shared_ve_pool

                clear_shared_ve_pool()
                _worker.pool.stop()
            except Exception:
                pass
            _worker = None
        _closed = True


@contextmanager
def session_scope(
    *,
    mode: str = "aveo_pin",
    ve_nodes: Optional[Sequence[int]] = None,
    pin_shape: Optional[tuple[int, int, int]] = None,
) -> Iterator[Any]:
    """Context that ensures a shared session exists; does not stop it on exit.

    Call ``shutdown_sessions()`` at process end (atexit registered).
    """
    if mode in ("aveo_pin", "aveo", "pin"):
        nodes = list(ve_nodes or [1])
        pm = pn = pk = 0
        if pin_shape:
            pm, pn, pk = pin_shape
        yield get_aveo_pool(nodes, pin_m=pm, pin_n=pn, pin_k=pk)
    elif mode in ("pool", "worker"):
        ids = list(ve_nodes or [1])
        yield get_worker_pool(ids)
    else:
        yield None


def dgemm_shared_aveo(
    a,
    b,
    *,
    ve_node: int = 1,
    pin: bool = True,
    pin_mode: str = "grow",
):
    """GEMM via shared AVEO session (optionally pinned)."""
    import numpy as np

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    n = b.shape[1]
    if pin:
        pool = get_aveo_pool(
            [ve_node], pin_m=m, pin_n=n, pin_k=k, pin_mode=pin_mode
        )
        return pool.dgemm_pinned(ve_node, a, b)
    pool = get_aveo_pool([ve_node], pin_mode=pin_mode)
    return pool.dgemm(ve_node, a, b)


# Clean up on interpreter exit
atexit.register(shutdown_sessions)
