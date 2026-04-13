#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/firas/dev/TransBench"
PIDFILE="$REPO_DIR/runs/cpu_long_latest.pid"

if [[ ! -f "$PIDFILE" ]]; then
  echo "No pidfile: nothing to stop"
  exit 0
fi

pid="$(cat "$PIDFILE" || true)"
if [[ -z "${pid:-}" ]]; then
  echo "Empty pidfile"
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  echo "Stopping pid=$pid"
  kill "$pid" || true
else
  echo "Not running pid=$pid"
fi
