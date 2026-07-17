#!/usr/bin/env bash
# Full local check: audit + L0 + optional device smoke. No GitHub CI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/ci_l0.sh

if [[ "${UCT_SKIP_DEVICE:-0}" == "1" ]]; then
  echo "=== skip device (UCT_SKIP_DEVICE=1) ==="
  exit 0
fi

bash scripts/ci_device.sh
echo "=== ci_all PASS ==="
