"""uct-serve protocol tests (host-only, temp Unix socket)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_roundtrip_local_run():
    from uni_cute_tensor.runtime.job_runner import run_job
    from uni_cute_tensor.runtime.serve_protocol import encode_msg, request_run

    job = {
        "type": "dense_batch",
        "m": 48,
        "n": 40,
        "k": 32,
        "batches": 2,
        "backend": "host",
        "host_only": True,
    }
    req = request_run(job, id="t1", host_only=True)
    assert req["op"] == "run"
    r = run_job(job, host_only=True)
    assert r.status == "pass"
    assert b"\n" in encode_msg({"ok": True})


def test_serve_ping_and_job(tmp_path: Path):
    sock = tmp_path / "uct-test.sock"
    # start server
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from uni_cute_tensor.runtime.serve import run_server; "
                f"run_server({str(sock)!r}, host_only=True)"
            ),
        ],
        cwd=str(ROOT),
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # wait for socket
        for _ in range(50):
            if sock.exists():
                break
            time.sleep(0.05)
            if proc.poll() is not None:
                out = proc.stdout.read().decode() if proc.stdout else ""
                pytest.fail(f"server died: {out}")
        assert sock.exists(), "socket not created"

        from uni_cute_tensor.runtime.serve_protocol import (
            client_call,
            request_health,
            request_ping,
            request_run,
            request_shutdown,
        )

        pong = client_call(str(sock), request_ping(), timeout=30)
        assert pong.get("ok") is True
        assert pong.get("op") == "pong"

        health = client_call(str(sock), request_health(), timeout=30)
        assert health.get("ok") is True
        assert "sessions" in health

        job = {
            "type": "dense_batch",
            "m": 40,
            "n": 32,
            "k": 24,
            "batches": 2,
            "backend": "host",
        }
        res = client_call(
            str(sock), request_run(job, id="j1", host_only=True), timeout=60
        )
        assert res.get("ok") is True
        assert res.get("result", {}).get("status") == "pass"

        # second job — same server (session registry may stay empty for host)
        res2 = client_call(
            str(sock), request_run(job, id="j2", host_only=True), timeout=60
        )
        assert res2.get("ok") is True

        client_call(str(sock), request_shutdown(), timeout=10)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if sock.exists():
            sock.unlink()
