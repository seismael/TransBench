#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv-sm61 ]]; then
  echo "Missing .venv-sm61. Create it first (torch==2.1.2+cu118)." >&2
  exit 1
fi

source .venv-sm61/bin/activate

# Keep it simple + predictable on older GPUs.
export CUDA_DEVICE_MAX_CONNECTIONS=1

transbench suite --config benchmarks.cuda.sm61.toml --reports-dir reports
transbench make-manifest --reports-dir reports

echo "Done. Manifest updated: reports/manifest.json"