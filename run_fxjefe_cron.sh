#!/bin/bash
set -euo pipefail
export PATH="/usr/local/bin:/opt/homebrew/bin:/Library/Frameworks/Python.framework/Versions/3.11/bin:/usr/bin:/bin"
ROOT="$HOME/Documents/FXJEFE_Project"
cd "$ROOT"
if [ -f "$ROOT/venv/bin/activate" ]; then
  source "$ROOT/venv/bin/activate"
fi
if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
LOGDIR="$ROOT/logs/cron"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/run_$STAMP.log"
{
  echo "START $STAMP"
  python3 runtime_lock.py
  python3 signal_gate.py
  if [ -f pipelinerun_production.py ]; then
    python3 pipelinerun_production.py
  elif [ -f pipelinerun.py ]; then
    python3 pipelinerun.py
  fi
  echo "END $(date +%Y%m%d_%H%M%S)"
} >>"$LOG" 2>&1
