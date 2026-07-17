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


def load_array(path: Union[str, Path]) -> np.ndarray:
    """Load float64 2-D array from .npy or .npz (first array / key 'arr'/'a')."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        a = np.load(str(path))
    elif path.suffix == ".npz":
        z = np.load(str(path))
        key = "arr" if "arr" in z else ("a" if "a" in z else z.files[0])
        a = z[key]
    else:
        # raw row-major float64 needs shape in sidecar — require npy/npz
        raise ValueError(f"unsupported array format: {path.suffix} (use .npy/.npz)")
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    return a


def load_csr_npz(path: Union[str, Path]):
    """Load CSR from npz: indptr, indices, data, optional nrows/ncols."""
    from uni_cute_tensor.apps.spmv_dataprep import CSRMatrix

    path = Path(path)
    z = np.load(str(path))
    indptr = np.asarray(z["indptr"], dtype=np.int32)
    indices = np.asarray(z["indices"], dtype=np.int32)
    data = np.asarray(z["data"], dtype=np.float64)
    nrows = int(z["nrows"]) if "nrows" in z else int(indptr.shape[0] - 1)
    ncols = int(z["ncols"]) if "ncols" in z else (int(indices.max()) + 1 if indices.size else 0)
    return CSRMatrix(nrows=nrows, ncols=ncols, indptr=indptr, indices=indices, data=data)


def _want_phi(spec: dict[str, Any], host_only: bool) -> bool:
    """Phi is opt-in via job field phi / use_phi; disabled under host_only."""
    if host_only or spec.get("host_only"):
        return False
    return bool(spec.get("phi") or spec.get("use_phi"))


def _prep_scale(
    a: np.ndarray,
    *,
    use_phi: bool,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[np.ndarray, str, float]:
    """Return (scaled, note, prep_sec). Phi on demand with Host fallback."""
    t0 = time.perf_counter()
    if not use_phi:
        out = np.ascontiguousarray(alpha * a + beta, dtype=np.float64)
        return out, "host_scale", time.perf_counter() - t0
    try:
        from pathlib import Path

        from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale

        if not Path("/dev/mic0").exists():
            out = np.ascontiguousarray(alpha * a + beta, dtype=np.float64)
            return out, "host_scale(no_mic)", time.perf_counter() - t0
        out, prep = run_phi_prep_scale(a, alpha=alpha, beta=beta)
        if prep.status != "pass":
            out = np.ascontiguousarray(alpha * a + beta, dtype=np.float64)
            return out, f"host_scale(phi_fail:{prep.status})", time.perf_counter() - t0
        return (
            out,
            f"phi_scale gflops={prep.gflops:.2f}",
            time.perf_counter() - t0,
        )
    except Exception as exc:
        out = np.ascontiguousarray(alpha * a + beta, dtype=np.float64)
        return out, f"host_scale(phi_skip:{exc})", time.perf_counter() - t0


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
    alpha = float(spec.get("alpha", 1.0))
    beta = float(spec.get("beta", 0.0))
    devices = _resolve_devices(spec, host_only)
    use_phi = _want_phi(spec, host_only)
    pin_mode = str(spec.get("pin_mode", "grow"))
    rng = np.random.default_rng(seed)
    if spec.get("matrix_a"):
        a_raw = load_array(spec["matrix_a"])
        m, k = a_raw.shape
    else:
        a_raw = rng.standard_normal((m, k))
    if spec.get("matrix_b"):
        b0 = load_array(spec["matrix_b"])
        if b0.shape[0] != k:
            raise ValueError(f"matrix_b inner dim {b0.shape[0]} != k={k}")
        n = b0.shape[1]
        bs = [b0 for _ in range(batches)]
    else:
        bs = [rng.standard_normal((k, n)) for _ in range(batches)]
    a, prep_note, prep_sec = _prep_scale(a_raw, use_phi=use_phi, alpha=alpha, beta=beta)

    rec = recommend_backend(
        m, n, k, batches=batches, ve_devices=devices[:1] if devices else [], prefer_mode="pin"
    )
    use_host = (
        host_only
        or backend == "host"
        or not devices
        or (backend == "auto" and rec.recommended == "host" and not spec.get("force_ve"))
    )
    if spec.get("force_ve") and devices:
        use_host = False

    notes = [
        f"recommend={rec.recommended} conf={rec.confidence}",
        f"prep={prep_note}",
    ]
    max_err = 0.0
    oneshot_wall = None
    # fair host baseline (warm OpenBLAS path), timed separately
    host_dgemm(a, bs[0], backend="auto")
    t_h0 = time.perf_counter()
    for b in bs:
        host_dgemm(a, b, backend="auto")
    host_wall = time.perf_counter() - t_h0

    t0 = time.perf_counter()
    with timeline_scope(job_id=spec.get("name", "dense_batch")) as tl:
        with tl.span("prep_scale", "host" if "host" in prep_note else "kernel", device="phi0" if use_phi else "host"):
            pass  # prep already done; span records 0 — time in prep_sec metric
        if use_host:
            for b in bs:
                with tl.span("host_gemm", "kernel", device="host"):
                    c, r = host_dgemm(a, b, backend="auto")
                max_err = max(max_err, r.max_abs_err)
            be = "HOST"
            notes.append("path=host_dgemm")
        else:
            node = int(devices[0].replace("ve", ""))
            if spec.get("compare_oneshot", True):
                from uni_cute_tensor.backends.ve_dgemm import run_ve_dgemm_shard

                # ve_exec and AVEO cannot share a VE node; free any shared session first
                shutdown_sessions()
                n_cold = min(3, batches)
                t_c = time.perf_counter()
                for b in bs[:n_cold]:
                    run_ve_dgemm_shard(a, b, ve_id=node)
                oneshot_wall = (time.perf_counter() - t_c) * (batches / n_cold)

            get_aveo_pool([node], pin_m=m, pin_n=n, pin_k=k, pin_mode=pin_mode)
            dgemm_shared_aveo(
                a[:32, :32], bs[0][:32, :32], ve_node=node, pin=True, pin_mode=pin_mode
            )
            t0 = time.perf_counter()
            for b in bs:
                with tl.span("aveo_pin", "kernel", device=f"ve{node}"):
                    c, r = dgemm_shared_aveo(
                        a, b, ve_node=node, pin=True, pin_mode=pin_mode
                    )
                max_err = max(max_err, r.max_abs_err, float(np.max(np.abs(c - a @ b))))
            be = "VE_AVEO_PIN_SHARED"
            notes.append(f"path=shared_aveo_pin ve{node} pin_mode={pin_mode}")
    wall = time.perf_counter() - t0
    thr = batches / wall if wall > 0 else 0.0
    host_thr = batches / host_wall if host_wall > 0 else 0.0
    status = "pass" if max_err < 1e-8 else "fail"
    metrics = {
        "m": m,
        "n": n,
        "k": k,
        "batches": batches,
        "phi": use_phi,
        "pin_mode": pin_mode,
        "external_a": bool(spec.get("matrix_a")),
        "external_b": bool(spec.get("matrix_b")),
        "prep_sec": prep_sec,
        "prep_note": prep_note,
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
    alpha = float(spec.get("alpha", 1.0))
    beta = float(spec.get("beta", 0.0))
    devices = _resolve_devices(spec, host_only)
    use_phi = _want_phi(spec, host_only)
    pin_mode = str(spec.get("pin_mode", "grow"))
    rng = np.random.default_rng(seed)
    if spec.get("csr_path"):
        csr = load_csr_npz(spec["csr_path"])
        nx = ny = -1
    else:
        csr = stencil5_csr(nx, ny)
    if spec.get("matrix_x"):
        x = load_array(spec["matrix_x"])
        nrhs = x.shape[1] if x.ndim == 2 else 1
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    else:
        x = rng.standard_normal((csr.ncols, nrhs))
    if spec.get("matrix_w"):
        w = load_array(spec["matrix_w"])
        n_out = w.shape[1]
    else:
        w = rng.standard_normal((nrhs, n_out))

    t0 = time.perf_counter()
    max_err = 0.0
    prep_note = "none"
    prep_sec = 0.0
    with timeline_scope(job_id=spec.get("name", "sparse_dense")) as tl:
        with tl.span("spmv", "kernel", device="host"):
            y = csr_spmv(csr, x)
        y, prep_note, prep_sec = _prep_scale(y, use_phi=use_phi, alpha=alpha, beta=beta)
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
            get_aveo_pool(
                [node],
                pin_m=y.shape[0],
                pin_n=n_out,
                pin_k=y.shape[1],
                pin_mode=pin_mode,
            )
            with tl.span("gemm", "kernel", device=f"ve{node}"):
                c, r = dgemm_shared_aveo(
                    y, w, ve_node=node, pin=True, pin_mode=pin_mode
                )
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
            "phi": use_phi,
            "csr_external": bool(spec.get("csr_path")),
            "prep_sec": prep_sec,
            "prep_note": prep_note,
            "timeline_phases": tl.summary(),
        },
        notes=[
            f"csr={'external' if spec.get('csr_path') else f'stencil5 {nx}x{ny}'}",
            f"prep={prep_note}",
            rec.reason[:100],
        ],
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
    alpha = float(spec.get("alpha", 1.0))
    beta = float(spec.get("beta", 0.0))
    devices = _resolve_devices(spec, host_only)
    use_phi = _want_phi(spec, host_only)
    rng = np.random.default_rng(seed)
    dirty = make_dirty_matrix(m, k, rng=rng)
    W = random_projection_matrix(k, n_out, rng=rng)

    t0 = time.perf_counter()
    prep_note = "none"
    prep_sec = 0.0
    with timeline_scope(job_id=spec.get("name", "dataprep")) as tl:
        with tl.span("clean", "host", device="host"):
            cleaned, st = clean_features(dirty)
        mean = cleaned.mean(axis=0)
        std = cleaned.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        xn = (cleaned - mean) / std
        xn, prep_note, prep_sec = _prep_scale(xn, use_phi=use_phi, alpha=alpha, beta=beta)
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
            "phi": use_phi,
            "prep_sec": prep_sec,
            "prep_note": prep_note,
            "timeline_phases": tl.summary(),
        },
        notes=[f"prep={prep_note}", rec.reason[:100]],
    )


def _run_phi_prep_ve_gemm(spec: dict[str, Any], host_only: bool) -> JobResult:
    """Explicit hetero story: scale (Phi-on-demand) then multi-batch VE/Host GEMM."""
    # Reuse dense_batch with force phi semantics
    s = dict(spec)
    s.setdefault("type", "dense_batch")
    s["phi"] = bool(spec.get("phi", True))  # default on for this job type
    if host_only:
        s["phi"] = False
    result = _run_dense_batch(s, host_only)
    result.job_type = "phi_prep_ve_gemm"
    result.notes = [f"hetero_story phi_requested={spec.get('phi', True)}"] + list(
        result.notes
    )
    return result


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
    if jtype in ("dense_batch", "dense", "service_dense_stream"):
        result = _run_dense_batch(job, host_only)
    elif jtype in ("sparse_dense", "spmv_dense", "sparse", "service_sparse_dense"):
        result = _run_sparse_dense(job, host_only)
    elif jtype in ("dataprep", "dataprep_project"):
        result = _run_dataprep(job, host_only)
    elif jtype in ("phi_prep_ve_gemm", "hetero_prep_gemm"):
        result = _run_phi_prep_ve_gemm(job, host_only)
    else:
        raise ValueError(f"unknown job type: {jtype}")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.write_json(out / "metrics.json")
    return result


def list_bundled_jobs() -> list[Path]:
    root = Path(__file__).resolve().parents[3] / "jobs"
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))
