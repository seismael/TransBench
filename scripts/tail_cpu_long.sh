#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/firas/dev/TransBench"
LOG="$REPO_DIR/runs/cpu_long_latest.log"
PIDFILE="$REPO_DIR/runs/cpu_long_latest.pid"

if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE" || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "RUNNING pid=$pid"
  else
    echo "NOT RUNNING (pidfile present: $pid)"
  fi
else
  echo "No pidfile found"
fi

echo "--- last 50 lines: $LOG ---"
tail -n 50 "$LOG" 2>/dev/null || true
