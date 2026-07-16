"""Line-delimited JSON protocol for uct-serve / uct-run --socket."""

from __future__ import annotations

import json
import socket
from typing import Any, Optional, TextIO, Union

# Default Unix socket path (local only; never bind 0.0.0.0)
DEFAULT_SOCKET_PATH = "/tmp/uct-serve.sock"


def encode_msg(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def read_msg(fp: TextIO, *, timeout_line: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Read one JSON object from a text file/socket makefile. None on EOF."""
    line = fp.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {"op": "noop"}
    return json.loads(line)


def write_msg(fp: TextIO, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def request_run(
    job: dict[str, Any],
    *,
    id: str = "0",
    host_only: bool = False,
) -> dict[str, Any]:
    return {"op": "run", "id": id, "job": job, "host_only": host_only}


def request_ping(id: str = "0") -> dict[str, Any]:
    return {"op": "ping", "id": id}


def request_health(id: str = "0") -> dict[str, Any]:
    return {"op": "health", "id": id}


def request_shutdown(id: str = "0") -> dict[str, Any]:
    return {"op": "shutdown", "id": id}


def client_call(
    sock_path: str,
    request: dict[str, Any],
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Connect to Unix socket, send one request, read one response."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        f = s.makefile("rwb", buffering=0)
        f.write(encode_msg(request))
        f.flush()
        # read response line
        buf = b""
        while b"\n" not in buf:
            chunk = f.read(4096)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0].decode("utf-8")
        if not line:
            raise ConnectionError("empty response from uct-serve")
        return json.loads(line)
    finally:
        try:
            s.close()
        except Exception:
            pass
