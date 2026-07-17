#!/usr/bin/env bash
# Stop uct-serve via protocol shutdown, then pidfile kill if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SOCKET="${UCT_SOCKET:-/tmp/uct-serve.sock}"
PIDFILE="${UCT_SERVE_PID:-$ROOT/artifacts/uct-serve.pid}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

PY="${ROOT}/env/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

if [[ -S "$SOCKET" ]]; then
  "$PY" -c "
from uni_cute_tensor.runtime.serve_protocol import client_call, request_shutdown
try:
    print(client_call('$SOCKET', request_shutdown(), timeout=10))
except Exception as e:
    print('shutdown rpc failed:', e)
" || true
  sleep 0.4
fi

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE" || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi
rm -f "$SOCKET"
echo "uct-serve stopped"
