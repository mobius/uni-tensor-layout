#!/usr/bin/env bash
# Start uct-serve with fixed defaults for daily use.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SOCKET="${UCT_SOCKET:-/tmp/uct-serve.sock}"
LOG="${UCT_SERVE_LOG:-$ROOT/artifacts/uct-serve.log}"
PIDFILE="${UCT_SERVE_PID:-$ROOT/artifacts/uct-serve.pid}"
PRELOAD="${UCT_PRELOAD:-256}"
VE_NODE="${UCT_VE_NODE:-1}"
HOST_ONLY="${UCT_HOST_ONLY:-0}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/nec/ve/veos/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VE_LD_LIBRARY_PATH="${VE_LD_LIBRARY_PATH:-/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "uct-serve already running pid=$(cat "$PIDFILE") socket=$SOCKET"
  exit 0
fi

PY="${ROOT}/env/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

# Build argv for serve_main
if [[ "$HOST_ONLY" == "1" || "$HOST_ONLY" == "true" ]]; then
  EXTRA="--host-only"
  PRE=""
else
  EXTRA=""
  PRE="--preload ${PRELOAD}"
fi

# shellcheck disable=SC2086
nohup "$PY" -c "
import sys
from uni_cute_tensor.cli import serve_main
sys.argv = ['uct-serve', '--socket', '${SOCKET}', '--ve-node', '${VE_NODE}'] + '''${EXTRA} ${PRE}'''.split()
raise SystemExit(serve_main())
" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
sleep 0.5
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "started uct-serve pid=$(cat "$PIDFILE") socket=$SOCKET log=$LOG"
else
  echo "failed to start; see $LOG" >&2
  tail -20 "$LOG" >&2 || true
  exit 1
fi
