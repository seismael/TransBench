#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/firas/dev/TransBench"
LOG="$REPO_DIR/runs/cpu_long_latest.log"
PIDFILE="$REPO_DIR/runs/cpu_long_latest.pid"

mkdir -p "$REPO_DIR/runs"
cd "$REPO_DIR"

# Stop existing run if present.
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping existing run pid=$old_pid"
    kill "$old_pid" || true
  fi
fi

# Start detached. Keep all bash syntax inside WSL.
nohup bash "$REPO_DIR/scripts/run_cpu_long.sh" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

echo "STARTED pid=$(cat "$PIDFILE")"
echo "LOG $LOG"
