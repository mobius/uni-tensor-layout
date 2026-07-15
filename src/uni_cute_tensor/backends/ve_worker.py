"""Persistent VE DGEMM workers — amortize ve_exec process startup."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from uni_cute_tensor.backends.ve_dgemm import (
    VeRunResult,
    _parse_checksum,
    _parse_elapsed,
    _parse_gflops,
    _read_output,
    _ve_env,
    _ve_id_from_name,
    _write_a_shard,
    _write_b_shared,
    _write_combined,
    compile_ve_dgemm,
    ve_toolchain_available,
)
from uni_cute_tensor.partition.multi_device import PlacementPlan, partition_to_devices

NLC_ROOT = Path(os.environ.get("NLC_ROOT", "/opt/nec/ve/nlc/3.1.0"))
VE_EXEC = Path(os.environ.get("VE_EXEC", "/opt/nec/ve/bin/ve_exec"))
NCC = Path(os.environ.get("NCC", "/opt/nec/ve/bin/ncc"))

_WORKER_SRC = Path(__file__).resolve().parents[1] / "kernels" / "ve" / "dgemm_worker.c"
_BUILD_DIR = Path(__file__).resolve().parents[3] / "build" / "ve"
_WORKER_BIN = _BUILD_DIR / "dgemm_worker_ve"


def compile_ve_worker(*, force: bool = False) -> Path:
    if not ve_toolchain_available():
        raise RuntimeError("VE toolchain unavailable")
    if not _WORKER_SRC.is_file():
        raise FileNotFoundError(_WORKER_SRC)
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _WORKER_BIN.is_file() and not force:
        if _WORKER_BIN.stat().st_mtime >= _WORKER_SRC.stat().st_mtime:
            return _WORKER_BIN
    # reuse dgemm_rect compile flags
    inc = NLC_ROOT / "include"
    lib = NLC_ROOT / "lib"
    cmd = [
        str(NCC),
        "-O3",
        "-fopenmp",
        "-o",
        str(_WORKER_BIN),
        str(_WORKER_SRC),
        f"-I{inc}",
        f"-L{lib}",
        "-lcblas",
        "-lblas_openmp",
        f"-Wl,-rpath,{lib}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"worker ncc failed:\n{r.stderr}")
    return _WORKER_BIN


@dataclass
class _Worker:
    ve_id: int
    control_dir: Path
    proc: subprocess.Popen


class VeWorkerPool:
    """One long-lived ve_exec process per VE device."""

    def __init__(self, ve_ids: Sequence[int], *, base_dir: Optional[Path] = None):
        self.ve_ids = list(ve_ids)
        self.base_dir = Path(base_dir) if base_dir else Path(
            tempfile.mkdtemp(prefix="cct_ve_pool_")
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._workers: dict[int, _Worker] = {}
        self._own_dir = base_dir is None

    def start(self) -> None:
        binary = compile_ve_worker()
        # also ensure one-shot dgemm binary exists (not required for worker)
        try:
            compile_ve_dgemm()
        except Exception:
            pass
        env = _ve_env()
        for vid in self.ve_ids:
            cdir = self.base_dir / f"ve{vid}"
            cdir.mkdir(parents=True, exist_ok=True)
            # clear stale
            for name in ("job.go", "job.cmd", "job.status", "job.log"):
                p = cdir / name
                if p.exists():
                    p.unlink()
            proc = subprocess.Popen(
                [str(VE_EXEC), "-N", str(vid), str(binary), str(cdir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
            )
            self._workers[vid] = _Worker(ve_id=vid, control_dir=cdir, proc=proc)
            # wait ready message or short sleep
            time.sleep(0.15)
            if proc.poll() is not None:
                err = (proc.stderr.read() if proc.stderr else b"").decode()
                raise RuntimeError(f"worker ve{vid} exited early: {err}")

    def stop(self) -> None:
        for vid, w in list(self._workers.items()):
            try:
                self._submit_raw(vid, "QUIT\n", timeout=10.0)
            except Exception:
                pass
            try:
                w.proc.terminate()
                w.proc.wait(timeout=5)
            except Exception:
                try:
                    w.proc.kill()
                except Exception:
                    pass
        self._workers.clear()
        if self._own_dir and self.base_dir.exists():
            shutil.rmtree(self.base_dir, ignore_errors=True)

    def __enter__(self) -> "VeWorkerPool":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def _submit_raw(self, ve_id: int, cmd: str, *, timeout: float = 120.0) -> str:
        w = self._workers[ve_id]
        st = w.control_dir / "job.status"
        go = w.control_dir / "job.go"
        log = w.control_dir / "job.log"
        cmdp = w.control_dir / "job.cmd"
        for p in (st, go, log):
            if p.exists():
                p.unlink()
        cmdp.write_text(cmd, encoding="utf-8")
        go.write_text("", encoding="utf-8")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if st.is_file():
                status = st.read_text(encoding="utf-8").strip()
                text = log.read_text(encoding="utf-8") if log.is_file() else ""
                if status == "FAIL":
                    raise RuntimeError(f"ve{ve_id} job FAIL: {text}")
                return text
            if w.proc.poll() is not None:
                raise RuntimeError(f"ve{ve_id} worker died")
            time.sleep(0.002)
        raise TimeoutError(f"ve{ve_id} job timeout")

    def run_split(
        self,
        ve_id: int,
        a_path: Path,
        b_path: Path,
        out_path: Path,
        *,
        timeout: float = 300.0,
    ) -> str:
        return self._submit_raw(
            ve_id,
            f"SPLIT {a_path} {b_path} {out_path}\n",
            timeout=timeout,
        )

    def run_combined(
        self,
        ve_id: int,
        in_path: Path,
        out_path: Path,
        *,
        timeout: float = 300.0,
    ) -> str:
        return self._submit_raw(
            ve_id,
            f"COMBINED {in_path} {out_path}\n",
            timeout=timeout,
        )


def multi_ve_layout_dgemm_pooled(
    a: np.ndarray,
    b: np.ndarray,
    devices: Sequence[str],
    pool: VeWorkerPool,
    *,
    work_dir: Optional[Path] = None,
    share_b: bool = True,
) -> tuple[np.ndarray, PlacementPlan, list[VeRunResult]]:
    """Multi-VE DGEMM using an already-started VeWorkerPool."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dim mismatch")
    plan = partition_to_devices(m, n, list(devices), prefer_kind="ve")
    c = np.zeros((m, n), dtype=np.float64)
    results: list[VeRunResult] = []

    own = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="cct_pool_jobs_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    b_path = work_dir / "b_shared.bin"
    if share_b:
        _write_b_shared(b_path, b)

    def _one(shard):
        ve_id = _ve_id_from_name(shard.device)
        a_sh = a[shard.row_start : shard.row_end, :]
        sub = work_dir / shard.device
        sub.mkdir(exist_ok=True)
        out_path = sub / "out.bin"
        t0 = time.perf_counter()
        if share_b:
            a_path = (sub / "a.bin").resolve()
            _write_a_shard(a_path, a_sh)
            log = pool.run_split(
                ve_id, a_path, b_path.resolve(), out_path.resolve()
            )
            mode = "split-pooled"
        else:
            in_path = (sub / "in.bin").resolve()
            _write_combined(in_path, a_sh, b)
            log = pool.run_combined(ve_id, in_path, out_path.resolve())
            mode = "combined-pooled"
        wall = time.perf_counter() - t0
        c_sh = _read_output(out_path)
        err = float(np.max(np.abs(c_sh - a_sh @ b)))
        res = VeRunResult(
            device=shard.device,
            ve_id=ve_id,
            m=a_sh.shape[0],
            n=n,
            k=k,
            gflops=_parse_gflops(log),
            elapsed_sec=_parse_elapsed(log) or wall,
            checksum=_parse_checksum(log),
            max_abs_err=err,
            stdout=log,
            stderr="",
            status="pass" if err < 1e-8 else "fail",
            mode=mode,
        )
        return slice(shard.row_start, shard.row_end), c_sh, res

    with ThreadPoolExecutor(max_workers=len(plan.shards)) as ex:
        futs = [ex.submit(_one, s) for s in plan.shards]
        for fut in as_completed(futs):
            sl, c_sh, res = fut.result()
            c[sl, :] = c_sh
            results.append(res)
    results.sort(key=lambda r: r.ve_id)
    if own:
        shutil.rmtree(work_dir, ignore_errors=True)
    return c, plan, results
