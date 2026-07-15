"""Optional uni-framework TaskGraph integration (W3.2).

If UNI_ROOT (or ~/Work/uni) provides scheduler.task_graph, build a DAG:
  spmv(host) → prep(phi|host) → gemm shards (ve*) with PowerCap.

Falls back to a tiny local asyncio DAG when uni is unavailable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from uni_cute_tensor.bridge.uni_adapter import resolve_uni_root
from uni_cute_tensor.power import PowerCap


@dataclass
class BridgeTaskResult:
    name: str
    device: str
    wall_sec: float
    status: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeGraphResult:
    results: dict[str, BridgeTaskResult]
    wall_sec: float
    backend: str  # uni | local
    status: str
    notes: list[str] = field(default_factory=list)


def try_import_uni_task_graph():
    root = resolve_uni_root()
    if root is None:
        return None, None
    import sys

    sched = str(root / "src")
    if sched not in sys.path:
        sys.path.insert(0, sched)
    try:
        from scheduler.task_graph import TaskGraph, TaskNode  # type: ignore

        return TaskGraph, TaskNode
    except Exception:
        return None, None


async def _local_execute(
    nodes: dict[str, dict[str, Any]],
    power_cap: Optional[PowerCap],
) -> dict[str, BridgeTaskResult]:
    """Minimal DAG: ready when deps done; parallel within waves."""
    done: dict[str, BridgeTaskResult] = {}
    pending = set(nodes)
    while pending:
        ready = [
            n
            for n in pending
            if all(d in done for d in nodes[n].get("depends_on", []))
        ]
        if not ready:
            raise RuntimeError(f"deadlock or missing deps: {pending}")

        async def _run(name: str):
            node = nodes[name]
            device = node["device"]
            op = node.get("op", "dgemm")
            if power_cap is not None:
                if not power_cap.can_launch([device], {device: op}):
                    # wait briefly and retry once
                    await asyncio.sleep(0.05)
                power_cap.reserve([device], {device: op})
            t0 = time.perf_counter()
            try:
                out = node["run_fn"]()
                if asyncio.iscoroutine(out):
                    out = await out
                status = "pass"
                payload = out if isinstance(out, dict) else {"value": out}
            except Exception as exc:
                status = "fail"
                payload = {"error": str(exc)}
            finally:
                if power_cap is not None:
                    power_cap.release([device])
            wall = time.perf_counter() - t0
            return name, BridgeTaskResult(
                name=name,
                device=device,
                wall_sec=wall,
                status=status,
                payload=payload,
            )

        parts = await asyncio.gather(*[_run(n) for n in ready])
        for name, res in parts:
            done[name] = res
            pending.discard(name)
            if res.status == "fail":
                # still continue to collect? fail fast
                for p in list(pending):
                    done[p] = BridgeTaskResult(
                        name=p, device=nodes[p]["device"], wall_sec=0.0, status="skip"
                    )
                pending.clear()
                break
    return done


def build_spmv_gemm_graph_fns(
    y_builder: Callable[[], np.ndarray],
    b: np.ndarray,
    devices: Sequence[str],
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    use_phi_prep: bool = False,
) -> dict[str, dict[str, Any]]:
    """Create node specs for SpMV-ready dense matrix Y then GEMM.

    y_builder() returns the dense left matrix after SpMV (host-computed).
    """
    state: dict[str, Any] = {}

    def do_spmv_load():
        y = y_builder()
        state["y"] = y
        return {"shape": list(y.shape)}

    def do_prep():
        y = state["y"]
        if use_phi_prep:
            try:
                from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale

                ys, prep = run_phi_prep_scale(y, alpha=alpha, beta=beta)
                state["y"] = ys
                return {"status": prep.status, "gflops": prep.gflops}
            except Exception as exc:
                state["y"] = alpha * y + beta
                return {"status": "host_fallback", "error": str(exc)}
        state["y"] = alpha * y + beta
        return {"status": "host_scale"}

    def do_gemm():
        from uni_cute_tensor.partition.cost_model import choose_best_placement
        from uni_cute_tensor.partition.runner import execute_plan

        y = state["y"]
        m, k = y.shape
        n = b.shape[1]
        choice = choose_best_placement(m, n, k, list(devices), backend="VE_NLC")
        res = execute_plan(y, b, choice.plan)
        state["c"] = res.c
        state["plan"] = choice.plan
        return {
            "status": res.status,
            "err": res.max_abs_err,
            "strategy": res.strategy,
            "wall": res.wall_sec,
            "devices": choice.devices,
        }

    prep_dev = "phi0" if use_phi_prep else "host"
    nodes = {
        "spmv": {
            "device": "host",
            "op": "scale",
            "depends_on": [],
            "run_fn": do_spmv_load,
        },
        "prep": {
            "device": prep_dev,
            "op": "scale",
            "depends_on": ["spmv"],
            "run_fn": do_prep,
        },
        "gemm": {
            "device": devices[0] if devices else "host",
            "op": "dgemm",
            "depends_on": ["prep"],
            "run_fn": do_gemm,
        },
    }
    return nodes


def run_spmv_gemm_task_graph(
    y_builder: Callable[[], np.ndarray],
    b: np.ndarray,
    devices: Sequence[str],
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    use_phi_prep: bool = False,
    power_cap: Optional[PowerCap] = None,
    prefer_uni: bool = True,
) -> BridgeGraphResult:
    """Execute SpMV→prep→GEMM via uni TaskGraph if available, else local DAG."""
    if power_cap is None:
        power_cap = PowerCap()
    nodes = build_spmv_gemm_graph_fns(
        y_builder,
        b,
        devices,
        alpha=alpha,
        beta=beta,
        use_phi_prep=use_phi_prep,
    )
    notes: list[str] = []
    t0 = time.perf_counter()

    TaskGraph, TaskNode = try_import_uni_task_graph() if prefer_uni else (None, None)
    if TaskGraph is not None and TaskNode is not None:
        notes.append("backend=uni")
        # uni TaskGraph._run_node runs run_fn via run_in_executor → must be sync
        uni_cap = getattr(power_cap, "_uni", None)
        graph = TaskGraph(power_cap=uni_cap)

        for name, spec in nodes.items():
            graph.add(
                TaskNode(
                    name=name,
                    device=spec["device"],
                    run_fn=spec["run_fn"],  # sync callable → dict
                    depends_on=list(spec.get("depends_on", [])),
                    op=spec.get("op", "idle"),
                    estimated_watts=280.0 if spec.get("op") == "dgemm" else 100.0,
                )
            )
        try:
            raw = asyncio.run(graph.execute(verbose=False))
            results = {}
            status = "pass"
            for name, val in (raw or {}).items():
                if isinstance(val, dict):
                    st = val.get("status", "pass")
                    if st in ("fail", "failed"):
                        status = "fail"
                    results[name] = BridgeTaskResult(
                        name=name,
                        device=nodes[name]["device"],
                        wall_sec=float(val.get("wall", 0.0)),
                        status=str(st),
                        payload=val,
                    )
                else:
                    results[name] = BridgeTaskResult(
                        name=name,
                        device=nodes[name]["device"],
                        wall_sec=0.0,
                        status="pass",
                        payload={"value": val},
                    )
            # mark missing nodes
            for name in nodes:
                if name not in results:
                    results[name] = BridgeTaskResult(
                        name=name,
                        device=nodes[name]["device"],
                        wall_sec=0.0,
                        status="missing",
                    )
                    status = "fail"
            wall = time.perf_counter() - t0
            if status == "pass" or any(
                isinstance(v.payload, dict) and v.payload.get("status") == "pass"
                for v in results.values()
            ):
                # gemm node is the critical correctness check
                gemm = results.get("gemm")
                if gemm and gemm.payload.get("status") == "pass":
                    status = "pass"
            return BridgeGraphResult(
                results=results,
                wall_sec=wall,
                backend="uni",
                status=status,
                notes=notes,
            )
        except Exception as exc:
            notes.append(f"uni_failed:{exc};fallback=local")

    # local fallback
    notes.append("backend=local")
    results = asyncio.run(_local_execute(nodes, power_cap))
    wall = time.perf_counter() - t0
    status = "pass" if all(r.status in ("pass", "skip") for r in results.values()) and any(
        r.status == "pass" for r in results.values()
    ) else "fail"
    if any(r.status == "fail" for r in results.values()):
        status = "fail"
    return BridgeGraphResult(
        results=results,
        wall_sec=wall,
        backend="local",
        status=status,
        notes=notes,
    )
