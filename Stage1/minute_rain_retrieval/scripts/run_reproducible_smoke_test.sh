#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-cuda}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/smoke_test}

"$PYTHON" train.py \
  --dataset-path data/reproducible_v1/minute_rainfall_full.npz \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs 1 \
  --batch-size 128 \
  --patience 1 \
  --max-train-dry-ratio 1 \
  --selection-metric balanced_mae \
  "$@" 2>&1 | tee "$OUTPUT_DIR.console.txt"

printf '\nSmoke test completed. Metrics: %s/metrics.json\n' "$OUTPUT_DIR"
