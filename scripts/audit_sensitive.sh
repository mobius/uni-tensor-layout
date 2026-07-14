#!/usr/bin/env bash
# Scan workspace text for likely secrets before commit/push.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Auditing for sensitive patterns under ${ROOT} (excluding venv/git/self)..."
fail=0

run_rg() {
  local pat="$1"
  rg -n --hidden \
    -g '!.git/**' \
    -g '!.venv/**' \
    -g '!env/.venv/**' \
    -g '!*.png' \
    -g '!scripts/audit_sensitive.sh' \
    -e "$pat" . 2>/dev/null || true
}

run_grep() {
  local pat="$1"
  grep -RInE \
    --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=env \
    --exclude='audit_sensitive.sh' \
    -e "$pat" . 2>/dev/null || true
}

check_pat() {
  local pat="$1"
  local hits
  if command -v rg >/dev/null 2>&1; then
    hits="$(run_rg "$pat")"
  else
    hits="$(run_grep "$pat")"
  fi
  if [[ -n "${hits}" ]]; then
    echo "$hits"
    echo "FAIL pattern: $pat" >&2
    fail=1
  fi
}

check_pat 'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY'
check_pat 'AKIA[0-9A-Z]{16}'
check_pat 'ghp_[A-Za-z0-9]{20,}'
check_pat 'github_pat_[A-Za-z0-9_]{20,}'
check_pat 'xox[baprs]-'
check_pat 'password[[:space:]]*=[[:space:]]*['\''"][^'\''"]+['\''"]'
# micinfo serial lines (avoid checking this script itself)
check_pat 'Device Serial Number'
check_pat 'ADKC[0-9A-Z]{8,}'

if [[ $fail -ne 0 ]]; then
  echo "Sensitive patterns detected. Fix before push." >&2
  exit 1
fi
echo "Audit OK (no matched secret patterns)."
