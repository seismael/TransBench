#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/firas/dev/TransBench"
VENV_PY="$REPO_DIR/.venv/bin/python"
ARCHI="$REPO_DIR/.venv/bin/transbench"
CFG="$REPO_DIR/benchmarks.cpu.long.toml"
REPORTS_DIR="$REPO_DIR/reports"

cd "$REPO_DIR"

# Force CPU-only behavior and tune threads.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$(nproc)"
export MKL_NUM_THREADS="$(nproc)"

"$ARCHI" suite --config "$CFG" --reports-dir "$REPORTS_DIR"
"$ARCHI" make-manifest --reports-dir "$REPORTS_DIR"

echo "Done. Reports in: $REPORTS_DIR"
