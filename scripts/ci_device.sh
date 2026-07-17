#!/usr/bin/env bash
# Optional device smoke on this machine class (Phi/VE). Skips cleanly if absent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/nec/ve/veos/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VE_LD_LIBRARY_PATH="${VE_LD_LIBRARY_PATH:-/opt/nec/ve/nlc/3.1.0/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/lib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"

PY="${ROOT}/env/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

echo "=== device pytest ==="
if [[ -d /sys/class/ve || -e /dev/mic0 ]]; then
  "$PY" -m pytest -q -m device || exit $?
else
  echo "no VE/Phi sysfs — skip device pytest"
fi

echo "=== short jobs smoke ==="
"$PY" -c "
from uni_cute_tensor.runtime.job_runner import run_job
r = run_job({'type':'dense_batch','m':64,'n':64,'k':64,'batches':2,'backend':'host','compare_oneshot':False}, host_only=True)
assert r.status=='pass', r
print('host dense_batch ok')
" 

if [[ -d /sys/class/ve ]]; then
  "$PY" -c "
from uni_cute_tensor.runtime.job_runner import run_job
from uni_cute_tensor.runtime.session import shutdown_sessions
r = run_job({'type':'dense_batch','m':128,'n':128,'k':128,'batches':4,'force_ve':True,'compare_oneshot':False,'phi':False})
print('ve dense', r.status, r.backend, r.metrics.get('throughput_batches_per_sec'))
assert r.status=='pass', r
shutdown_sessions()
" || exit $?
fi

echo "=== device smoke PASS ==="
