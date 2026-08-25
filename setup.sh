#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ERR_LOG="${ROOT}/logs/setup_errors.log"
mkdir -p "${ROOT}/logs"

fail() {
  local code="${1:-1}"
  shift || true
  local msg="${*:-unknown error}"
  echo "[FXJEFE-SETUP][ERROR] ${msg}" >&2
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ERROR ${msg}" >>"${ERR_LOG}" 2>/dev/null || true
  exit "${code}"
}

warn() {
  echo "[FXJEFE-SETUP][WARN] $*" >&2
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WARN $*" >>"${ERR_LOG}" 2>/dev/null || true
}

info() {
  echo "[FXJEFE-SETUP] $*"
}

on_err() {
  local line="${1:-?}"
  local cmd="${2:-?}"
  fail 1 "line ${line}: command failed: ${cmd}"
}

trap 'on_err ${LINENO} "${BASH_COMMAND}"' ERR

info "root=${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  fail 2 "python3 not found in PATH"
fi

if [ ! -d fxjefe ]; then
  info "creating venv fxjefe"
  python3 -m venv fxjefe || fail 3 "venv creation failed"
else
  info "using existing venv fxjefe"
fi

if [ ! -f fxjefe/bin/activate ] && [ ! -f fxjefe/Scripts/activate ]; then
  fail 3 "fxjefe venv activate script missing"
fi

if [ -f fxjefe/bin/activate ]; then
  source fxjefe/bin/activate
elif [ -f fxjefe/Scripts/activate ]; then
  source fxjefe/Scripts/activate
fi

python -m pip install -U pip setuptools wheel || fail 4 "pip bootstrap failed"

if [ -f requirements_mac.txt ] && [ "$(uname -s)" = "Darwin" ]; then
  pip install -r requirements_mac.txt || warn "requirements_mac partial failure — minimal fallback"
  pip install numpy pandas flask joblib 2>/dev/null || true
elif [ -f requirements_linux.txt ]; then
  pip install -r requirements_linux.txt || warn "requirements_linux partial failure — minimal fallback"
  pip install numpy pandas flask joblib 2>/dev/null || true
else
  warn "no requirements_*.txt found — installing minimal set"
  pip install numpy pandas flask joblib || fail 4 "minimal pip install failed"
fi

info "SHA256 verification (required)"
if [ ! -f verify_checksums.py ]; then
  fail 5 "verify_checksums.py missing"
fi
if [ ! -f checksums.json ] && [ ! -f SHA256SUMS ]; then
  fail 5 "checksums.json or SHA256SUMS missing"
fi

set +e
python verify_checksums.py
sha_rc=$?
set -e
if [ "${sha_rc}" -ne 0 ]; then
  fail 5 "SHA256 verification failed (exit ${sha_rc}) — refuse to continue"
fi
info "SHA256 ALL VALID"

if [ -f runtime_lock.py ]; then
  set +e
  python runtime_lock.py
  lock_rc=$?
  set -e
  if [ "${lock_rc}" -ne 0 ]; then
    warn "runtime_lock returned ${lock_rc} (non-fatal)"
  else
    info "runtime_lock OK"
  fi
else
  warn "runtime_lock.py not present — skip"
fi

if command -v docker >/dev/null 2>&1; then
  set +e
  docker build -t fxjefe .
  docker_rc=$?
  set -e
  if [ "${docker_rc}" -ne 0 ]; then
    warn "docker build failed (exit ${docker_rc}) — host venv still usable"
  else
    info "docker image fxjefe built"
    info "run: docker run --rm -p 8080:8080 fxjefe"
  fi
else
  info "Docker not installed — host fxjefe venv ready"
fi

trap - ERR
info "DONE"
exit 0
