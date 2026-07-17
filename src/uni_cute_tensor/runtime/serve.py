"""uct-serve: local Unix-socket job daemon with shared VE sessions.

Phase 4 M2 — process stays up; multiple clients submit jobs without cold open.
Binds AF_UNIX only (never TCP / 0.0.0.0).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from uni_cute_tensor.runtime.job_runner import run_job
from uni_cute_tensor.runtime.serve_protocol import DEFAULT_SOCKET_PATH, encode_msg
from uni_cute_tensor.runtime.session import session_health, shutdown_sessions


class JobServer:
    def __init__(
        self,
        sock_path: str = DEFAULT_SOCKET_PATH,
        *,
        host_only: bool = False,
        preload_pin: Optional[tuple[int, int, int]] = None,
        ve_node: int = 1,
    ):
        self.sock_path = sock_path
        self.host_only = host_only
        self.preload_pin = preload_pin
        self.ve_node = ve_node
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._jobs_done = 0
        self._jobs_fail = 0
        self._t0 = time.time()

    def _preload(self) -> None:
        if self.host_only or not self.preload_pin:
            return
        try:
            from uni_cute_tensor.runtime.session import get_aveo_pool

            m, n, k = self.preload_pin
            get_aveo_pool([self.ve_node], pin_m=m, pin_n=n, pin_k=k)
            print(
                f"[uct-serve] preloaded AVEO pin ve{self.ve_node} shape=({m},{n},{k})",
                flush=True,
            )
        except Exception as exc:
            print(f"[uct-serve] preload skipped: {exc}", flush=True)

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
        op = str(req.get("op", "run"))
        rid = str(req.get("id", "0"))

        if op in ("ping", "noop"):
            return {
                "op": "pong",
                "id": rid,
                "ok": True,
                "uptime_sec": time.time() - self._t0,
            }

        if op == "health":
            return {
                "op": "health",
                "id": rid,
                "ok": True,
                "uptime_sec": time.time() - self._t0,
                "jobs_done": self._jobs_done,
                "jobs_fail": self._jobs_fail,
                "sessions": session_health(),
            }

        if op == "shutdown":
            self._stop.set()
            return {"op": "shutdown", "id": rid, "ok": True}

        if op != "run":
            return {
                "op": "error",
                "id": rid,
                "ok": False,
                "error": f"unknown op {op}",
            }

        job = req.get("job")
        if not isinstance(job, dict):
            return {
                "op": "error",
                "id": rid,
                "ok": False,
                "error": "job must be object",
            }

        host_only = bool(req.get("host_only", self.host_only))
        # serialize jobs so shared AVEO session is not used concurrently
        t_wait0 = time.perf_counter()
        with self._lock:
            queue_wait_sec = time.perf_counter() - t_wait0
            try:
                result = run_job(job, host_only=host_only, out_dir=None)
                self._jobs_done += 1
                if result.status != "pass":
                    self._jobs_fail += 1
                payload = result.to_dict()
                # surface queue wait for service SLOs (not power)
                metrics = dict(payload.get("metrics") or {})
                metrics["queue_wait_sec"] = queue_wait_sec
                payload["metrics"] = metrics
                return {
                    "op": "result",
                    "id": rid,
                    "ok": result.status == "pass",
                    "status": result.status,
                    "queue_wait_sec": queue_wait_sec,
                    "result": payload,
                }
            except Exception as exc:
                self._jobs_fail += 1
                return {
                    "op": "error",
                    "id": rid,
                    "ok": False,
                    "error": str(exc),
                    "queue_wait_sec": queue_wait_sec,
                    "traceback": traceback.format_exc()[-2000:],
                }

    def _client_thread(self, conn: socket.socket) -> None:
        """One request / one response per connection."""
        try:
            f = conn.makefile("rwb", buffering=0)
            buf = b""
            while b"\n" not in buf:
                chunk = f.read(4096)
                if not chunk:
                    return
                buf += chunk
                if len(buf) > 16 * 1024 * 1024:
                    f.write(
                        encode_msg(
                            {"op": "error", "ok": False, "error": "request too large"}
                        )
                    )
                    return
            line = buf.split(b"\n", 1)[0].decode("utf-8")
            try:
                req = json.loads(line) if line.strip() else {"op": "ping"}
            except Exception as exc:
                f.write(
                    encode_msg(
                        {"op": "error", "ok": False, "error": f"bad json: {exc}"}
                    )
                )
                return
            resp = self.handle_request(req)
            f.write(encode_msg(resp))
            f.flush()
        except Exception as exc:
            try:
                conn.sendall(
                    encode_msg({"op": "error", "ok": False, "error": str(exc)})
                )
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve_forever(self) -> None:
        path = Path(self.sock_path)
        if path.exists():
            path.unlink()

        self._preload()
        ss = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ss.bind(self.sock_path)
        try:
            os.chmod(self.sock_path, 0o600)
        except OSError:
            pass
        ss.listen(32)
        ss.settimeout(0.5)
        print(f"[uct-serve] listening on unix:{self.sock_path}", flush=True)
        print(f"[uct-serve] host_only={self.host_only}", flush=True)

        def _sig(_s, _f):
            self._stop.set()

        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

        try:
            while not self._stop.is_set():
                try:
                    conn, _addr = ss.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                threading.Thread(
                    target=self._client_thread, args=(conn,), daemon=True
                ).start()
        finally:
            try:
                ss.close()
            except Exception:
                pass
            p = Path(self.sock_path)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
            shutdown_sessions()
            print(
                f"[uct-serve] stopped jobs_done={self._jobs_done} fail={self._jobs_fail}",
                flush=True,
            )


def run_server(
    sock_path: str = DEFAULT_SOCKET_PATH,
    *,
    host_only: bool = False,
    preload: Optional[str] = None,
    ve_node: int = 1,
) -> None:
    pin = None
    if preload:
        parts = [int(x) for x in preload.split(",")]
        if len(parts) == 1:
            pin = (parts[0], parts[0], parts[0])
        elif len(parts) == 3:
            pin = (parts[0], parts[1], parts[2])
        else:
            raise ValueError("preload must be N or M,N,K")
    JobServer(
        sock_path, host_only=host_only, preload_pin=pin, ve_node=ve_node
    ).serve_forever()
