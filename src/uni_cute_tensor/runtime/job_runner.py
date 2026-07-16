"""Job runner: load JSON job spec → execute → unified metrics (Phase 3 M2)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

JobPath = Union[str, Path]


@dataclass
class JobResult:
    job_type: str
    status: str
    wall_sec: float
    max_abs_err: float
    backend: str
    recommended: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: JobPath) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_job(path: JobPath) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            return dict(yaml.safe_load(text))
        except ImportError as exc:
            raise RuntimeError(
                "YAML jobs require PyYAML; use .json or pip install pyyaml"
            ) from exc
    return json.loads(text)


def _resolve_devices(spec: dict[str, Any], host_only: bool) -> list[str]:
    if host_only or spec.get("host_only"):
        return []
    from uni_cute_tensor.backends.ve_dgemm import ve_toolchain_available
    from uni_cute_tensor.bridge.uni_adapter import discover_devices, ve_device_names

    if not ve_toolchain_available():
        return []
    found = ve_device_names(discover_devices())
    want = spec.get("devices", "auto")
    if want in (None, "auto", ""):
        return list(found)
    if isinstance(want, str):
        names = [d.strip() for d in want.split(",") if d.strip()]
    else:
        names = list(want)
    return [d for d in names if d in found] or list(found)


def _run_dense_batch(spec: dict[str, Any], host_only: bool) -> JobResult:
    from uni_cute_tensor.partition.dispatch import recommend_backend
    from uni_cute_tensor.runtime.session import dgemm_shared_aveo, get_aveo_pool, shutdown_sessions
    from uni_cute_tensor.runtime.timeline import timeline_scope
    from uni_cute_tensor.backends.host_dgemm import host_dgemm

    m = int(spec.get("m", 512))
    n = int(spec.get("n", 512))
    k = int(spec.get("k", 512))
    batches = int(spec.get("batches", 8))
    seed = int(spec.get("seed", 0))
    backend = str(spec.get("backend", "auto"))
    devices = _resolve_devices(spec, host_only)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k))
    bs = [rng.standard_normal((k, n)) for _ in range(batches)]

    rec = recommend_backend(
        m, n, k, batches=batches, ve_devices=devices[:1] if devices else [], prefer_mode="pin"
    )
    use_host = (
        host_only
        or backend == "host"
        or not devices
        or (backend == "auto" and rec.recommended == "host" and not spec.get("force_ve"))
    )
    # force_ve for host-not-win demos
    if spec.get("force_ve") and devices:
        use_host = False

    notes = [f"recommend={rec.recommended} conf={rec.confidence}"]
    max_err = 0.0
    # fair host baseline (warm OpenBLAS path), timed separately
    host_dgemm(a, bs[0], backend="auto")
    t_h0 = time.perf_counter()
    for b in bs:
        host_dgemm(a, b, backend="auto")
    host_wall = time.perf_counter() - t_h0

    t0 = time.perf_counter()
    with timeline_scope(job_id=spec.get("name", "dense_batch")) as tl:
        if use_host:
            for b in bs:
                with tl.span("host_gemm", "kernel", device="host"):
                    c, r = host_dgemm(a, b, backend="auto")
                max_err = max(max_err, r.max_abs_err)
            be = "HOST"
            notes.append("path=host_dgemm")
        else:
            node = int(devices[0].replace("ve", ""))
            # optional cold oneshot baseline (few jobs) — shows residency value
            oneshot_wall = None
            if spec.get("compare_oneshot", True):
                from uni_cute_tensor.backends.ve_dgemm import run_ve_dgemm_shard

                n_cold = min(3, batches)
                t_c = time.perf_counter()
                for b in bs[:n_cold]:
                    run_ve_dgemm_shard(a, b, ve_id=node)
                oneshot_wall = (time.perf_counter() - t_c) * (batches / n_cold)

            get_aveo_pool([node], pin_m=m, pin_n=n, pin_k=k)
            dgemm_shared_aveo(a[:32, :32], bs[0][:32, :32], ve_node=node, pin=True)
            t0 = time.perf_counter()  # steady-state after session warm
            for b in bs:
                with tl.span("aveo_pin", "kernel", device=f"ve{node}"):
                    c, r = dgemm_shared_aveo(a, b, ve_node=node, pin=True)
                max_err = max(max_err, r.max_abs_err, float(np.max(np.abs(c - a @ b))))
            be = "VE_AVEO_PIN_SHARED"
            notes.append(f"path=shared_aveo_pin ve{node}")
    wall = time.perf_counter() - t0
    thr = batches / wall if wall > 0 else 0.0
    host_thr = batches / host_wall if host_wall > 0 else 0.0
    status = "pass" if max_err < 1e-8 else "fail"
    metrics = {
        "m": m,
        "n": n,
        "k": k,
        "batches": batches,
        "throughput_batches_per_sec": thr,
        "host_dgemm_wall_sec": host_wall,
        "host_dgemm_batches_per_sec": host_thr,
        "speedup_vs_host_dgemm": host_wall / wall if wall > 0 else 0.0,
        "timeline_phases": tl.summary(),
        "devices": devices,
        "force_ve": bool(spec.get("force_ve")),
    }
    if not use_host and oneshot_wall is not None and oneshot_wall > 0:
        metrics["oneshot_scaled_wall_sec"] = oneshot_wall
        metrics["oneshot_batches_per_sec"] = batches / oneshot_wall
        metrics["speedup_vs_oneshot"] = oneshot_wall / wall
        notes.append(
            f"vs_oneshot={metrics['speedup_vs_oneshot']:.2f}x "
            f"(resident pin thr={thr:.1f} vs oneshot~{metrics['oneshot_batches_per_sec']:.1f} b/s)"
        )
    # keep session warm unless asked to close
    if spec.get("shutdown_session"):
        shutdown_sessions()
    return JobResult(
        job_type="dense_batch",
        status=status,
        wall_sec=wall,
        max_abs_err=max_err,
        backend=be,
        recommended=rec.recommended,
        metrics=metrics,
        notes=notes + [rec.reason[:120]],
    )


def _run_sparse_dense(spec: dict[str, Any], host_only: bool) -> JobResult:
    from uni_cute_tensor.apps.spmv_dataprep import csr_spmv, stencil5_csr
    from uni_cute_tensor.partition.dispatch import recommend_backend
    from uni_cute_tensor.runtime.session import dgemm_shared_aveo, get_aveo_pool
    from uni_cute_tensor.runtime.timeline import timeline_scope
    from uni_cute_tensor.backends.host_dgemm import host_dgemm

    nx = int(spec.get("nx", 40))
    ny = int(spec.get("ny", 40))
    nrhs = int(spec.get("nrhs", 64))
    n_out = int(spec.get("n_out", 128))
    seed = int(spec.get("seed", 0))
    devices = _resolve_devices(spec, host_only)
    rng = np.random.default_rng(seed)
    csr = stencil5_csr(nx, ny)
    x = rng.standard_normal((csr.ncols, nrhs))
    w = rng.standard_normal((nrhs, n_out))

    t0 = time.perf_counter()
    max_err = 0.0
    with timeline_scope(job_id=spec.get("name", "sparse_dense")) as tl:
        with tl.span("spmv", "kernel", device="host"):
            y = csr_spmv(csr, x)
        rec = recommend_backend(
            y.shape[0], n_out, y.shape[1], batches=1, ve_devices=devices[:1] if devices else []
        )
        use_host = host_only or not devices or spec.get("backend") == "host"
        if use_host:
            with tl.span("gemm", "kernel", device="host"):
                c, r = host_dgemm(y, w, backend="auto")
            be = "HOST"
            max_err = r.max_abs_err
        else:
            node = int(devices[0].replace("ve", ""))
            get_aveo_pool([node], pin_m=y.shape[0], pin_n=n_out, pin_k=y.shape[1])
            with tl.span("gemm", "kernel", device=f"ve{node}"):
                c, r = dgemm_shared_aveo(y, w, ve_node=node, pin=True)
            be = "VE_AVEO_PIN_SHARED"
            max_err = max(r.max_abs_err, float(np.max(np.abs(c - y @ w))))
    wall = time.perf_counter() - t0
    return JobResult(
        job_type="sparse_dense",
        status="pass" if max_err < 1e-8 else "fail",
        wall_sec=wall,
        max_abs_err=max_err,
        backend=be,
        recommended=rec.recommended,
        metrics={
            "nx": nx,
            "ny": ny,
            "nnz": csr.nnz,
            "nrhs": nrhs,
            "n_out": n_out,
            "timeline_phases": tl.summary(),
        },
        notes=[f"stencil5 {nx}x{ny}", rec.reason[:100]],
    )


def _run_dataprep(spec: dict[str, Any], host_only: bool) -> JobResult:
    from uni_cute_tensor.apps.dataprep_clean import (
        clean_features,
        make_dirty_matrix,
        random_projection_matrix,
    )
    from uni_cute_tensor.partition.dispatch import recommend_backend
    from uni_cute_tensor.runtime.session import dgemm_shared_aveo, get_aveo_pool
    from uni_cute_tensor.runtime.timeline import timeline_scope
    from uni_cute_tensor.backends.host_dgemm import host_dgemm

    m = int(spec.get("m", 512))
    k = int(spec.get("k", 384))
    n_out = int(spec.get("n_out", 96))
    seed = int(spec.get("seed", 0))
    devices = _resolve_devices(spec, host_only)
    rng = np.random.default_rng(seed)
    dirty = make_dirty_matrix(m, k, rng=rng)
    W = random_projection_matrix(k, n_out, rng=rng)

    t0 = time.perf_counter()
    with timeline_scope(job_id=spec.get("name", "dataprep")) as tl:
        with tl.span("clean", "host", device="host"):
            cleaned, st = clean_features(dirty)
        mean = cleaned.mean(axis=0)
        std = cleaned.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        xn = (cleaned - mean) / std
        rec = recommend_backend(
            m, n_out, k, batches=1, ve_devices=devices[:1] if devices else []
        )
        use_host = host_only or not devices or spec.get("backend") == "host"
        if use_host:
            with tl.span("project", "kernel", device="host"):
                c, r = host_dgemm(xn, W, backend="auto")
            be = "HOST"
            err = r.max_abs_err
        else:
            node = int(devices[0].replace("ve", ""))
            get_aveo_pool([node], pin_m=m, pin_n=n_out, pin_k=k)
            with tl.span("project", "kernel", device=f"ve{node}"):
                c, r = dgemm_shared_aveo(xn, W, ve_node=node, pin=True)
            be = "VE_AVEO_PIN_SHARED"
            err = max(r.max_abs_err, float(np.max(np.abs(c - xn @ W))))
    wall = time.perf_counter() - t0
    return JobResult(
        job_type="dataprep",
        status="pass" if err < 1e-8 and not np.isnan(cleaned).any() else "fail",
        wall_sec=wall,
        max_abs_err=err,
        backend=be,
        recommended=rec.recommended,
        metrics={
            "m": m,
            "k": k,
            "n_out": n_out,
            "nan_filled": st.nan_count,
            "clipped": st.clipped_count,
            "timeline_phases": tl.summary(),
        },
        notes=[rec.reason[:100]],
    )


def run_job(
    job: Union[dict[str, Any], JobPath],
    *,
    host_only: bool = False,
    out_dir: Optional[JobPath] = None,
) -> JobResult:
    """Execute a job dict or path to JSON/YAML job file."""
    if not isinstance(job, dict):
        job = load_job(job)
    jtype = str(job.get("type") or job.get("job_type") or "dense_batch")
    if jtype in ("dense_batch", "dense"):
        result = _run_dense_batch(job, host_only)
    elif jtype in ("sparse_dense", "spmv_dense", "sparse"):
        result = _run_sparse_dense(job, host_only)
    elif jtype in ("dataprep", "dataprep_project"):
        result = _run_dataprep(job, host_only)
    else:
        raise ValueError(f"unknown job type: {jtype}")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.write_json(out / "metrics.json")
        # timeline not auto-written; phases in metrics
    return result


def list_bundled_jobs() -> list[Path]:
    root = Path(__file__).resolve().parents[3] / "jobs"
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))
