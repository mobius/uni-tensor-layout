#!/usr/bin/env bash
# Hardware gate for cpu-cute-tensor. Prints capability summary; no secrets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "${ROOT}/env/.venv/bin/python" ]]; then
  PY="${ROOT}/env/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "No python found" >&2
  exit 1
fi

echo "== sysfs / devices =="
ls /dev/mic0 2>/dev/null && echo "mic0: present" || echo "mic0: absent"
ls /dev/ve0 /dev/ve1 /dev/ve2 2>/dev/null || echo "ve: missing some nodes"
command -v ncc >/dev/null && ncc --version | head -1 || echo "ncc: missing"
command -v micctrl >/dev/null && micctrl -s 2>/dev/null | head -3 || echo "micctrl: missing"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "nvidia-smi: absent (expected on this machine)"

echo
echo "== python probe =="
"$PY" -c "from cpu_cute_tensor.cli import check_hw_main; check_hw_main()"
