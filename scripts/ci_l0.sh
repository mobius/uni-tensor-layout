#!/usr/bin/env bash
# Local L0 CI: unit tests without Phi/VE device markers. No secrets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== audit ==="
bash scripts/audit_sensitive.sh

echo "=== L0 pytest (not device) ==="
python -m pytest -q -m "not device"

echo "=== public API import ==="
python - <<'PY'
import uni_cute_tensor as u
from uni_cute_tensor import (
    PowerCap,
    choose_best_placement,
    create_dataplane,
    host_dgemm,
    PlacementPlan,
)
assert u.__version__
print("version", u.__version__)
print("API ok")
PY

echo "=== L0 PASS ==="
