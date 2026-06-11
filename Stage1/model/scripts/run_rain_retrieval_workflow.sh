#!/usr/bin/env bash
# End-to-end Stage1 rainfall retrieval workflow.
#
# Default flow:
#   1. Predict camera weather labels with a minute-level timestamp.
#   2. Build a timestamped NPZ dataset from link/weather DB + camera labels.
#   3. Train the recommended drybase + summary + instant-aux model.
#   4. Run same-satellite rainy/dry link analysis.
#   5. Export full predictions, rainy-only predictions, and MAE/MSE/RMSE metrics.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE1_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M)}"
EXPERIMENT="${EXPERIMENT:-rain_retrieval}"
DATASET_NAME="${DATASET_NAME:-pass_dataset_${EXPERIMENT}_${RUN_TS}}"

LABEL_DIR="${LABEL_DIR:-$STAGE1_ROOT/data/camera_labels}"
DATASET_DIR="${DATASET_DIR:-$ROOT/data/datasets}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-$ROOT/checkpoints}"
RESULT_BASE="${RESULT_BASE:-$STAGE1_ROOT/analysis/satellite_weather_diff/runs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

LABEL_CSV="$LABEL_DIR/${RUN_TS}_weather_labels.csv"
SLIM_LABEL_CSV="$LABEL_DIR/${RUN_TS}_weather_labels_slim.csv"
PASS_DATASET_PATH="${PASS_DATASET_PATH:-$DATASET_DIR/${DATASET_NAME}.npz}"
CHECKPOINTS="${CHECKPOINTS:-$CHECKPOINT_BASE/${DATASET_NAME}}"
RUN_RESULT_DIR="${RUN_RESULT_DIR:-$RESULT_BASE/${DATASET_NAME}}"
WORKFLOW_LOG="${WORKFLOW_LOG:-$LOG_DIR/${DATASET_NAME}_workflow.log}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/${DATASET_NAME}_train.log}"

mkdir -p "$LABEL_DIR" "$DATASET_DIR" "$CHECKPOINTS" "$RUN_RESULT_DIR" "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$WORKFLOW_LOG"
}

run_logged() {
  log "$*"
  "$@" 2>&1 | tee -a "$WORKFLOW_LOG"
}

log "Stage1 workflow started"
log "dataset_name=$DATASET_NAME"
log "run_ts=$RUN_TS"
log "label_csv=$SLIM_LABEL_CSV"
log "dataset_npz=$PASS_DATASET_PATH"
log "checkpoints=$CHECKPOINTS"
log "results=$RUN_RESULT_DIR"

export RUN_TS
export PASS_DATASET_PATH
export BUILD_NPZ="${BUILD_NPZ:-1}"
export REBUILD_NPZ="${REBUILD_NPZ:-1}"
run_logged bash scripts/predict_camera_weather.sh

if [[ ! -f "$SLIM_LABEL_CSV" ]]; then
  echo "Missing expected label CSV: $SLIM_LABEL_CSV" >&2
  exit 1
fi
if [[ ! -f "$PASS_DATASET_PATH" ]]; then
  echo "Missing expected dataset NPZ: $PASS_DATASET_PATH" >&2
  exit 1
fi

export IMAGE_LABEL_CSV="$SLIM_LABEL_CSV"
export CHECKPOINTS
export LOG_FILE="$TRAIN_LOG"
export VAL_STRATEGY="${VAL_STRATEGY:-stratified_all}"
export ITERATIONS="${ITERATIONS:-1}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export PATIENCE="${PATIENCE:-15}"
run_logged bash scripts/train_experiments.sh "$EXPERIMENT"

CKPT_DIR="$(
  find "$CHECKPOINTS" -type f -name checkpoint.pth -printf '%T@ %h\n' |
    sort -n |
    tail -1 |
    cut -d' ' -f2-
)"
if [[ -z "$CKPT_DIR" || ! -f "$CKPT_DIR/checkpoint.pth" ]]; then
  echo "No checkpoint.pth found under $CHECKPOINTS" >&2
  exit 1
fi
log "selected_checkpoint=$CKPT_DIR"

LINK_ANALYSIS_DIR="$RUN_RESULT_DIR/link_diff"
run_logged python3 "$STAGE1_ROOT/analysis/satellite_weather_diff/analyze_satellite_weather_diff.py" \
  --npz "$PASS_DATASET_PATH" \
  --out-dir "$LINK_ANALYSIS_DIR"

PRED_CSV="$RUN_RESULT_DIR/${DATASET_NAME}_predictions.csv"
TEST_CSV="$RUN_RESULT_DIR/${DATASET_NAME}_test_predictions.csv"
METRICS_CSV="$RUN_RESULT_DIR/${DATASET_NAME}_metrics.csv"
run_logged python3 scripts/evaluate_checkpoint_splits.py \
  --checkpoint-dir "$CKPT_DIR" \
  --batch-size "${EVAL_BATCH_SIZE:-128}" \
  --out-csv "$PRED_CSV" \
  --test-csv "$TEST_CSV" \
  --metrics-csv "$METRICS_CSV"

{
  echo "dataset_name,$DATASET_NAME"
  echo "run_ts,$RUN_TS"
  echo "label_csv,$SLIM_LABEL_CSV"
  echo "dataset_npz,$PASS_DATASET_PATH"
  echo "checkpoint_dir,$CKPT_DIR"
  echo "prediction_csv,$PRED_CSV"
  echo "test_prediction_csv,$TEST_CSV"
  echo "metrics_csv,$METRICS_CSV"
  echo "link_analysis_dir,$LINK_ANALYSIS_DIR"
  echo "workflow_log,$WORKFLOW_LOG"
  echo "train_log,$TRAIN_LOG"
} > "$RUN_RESULT_DIR/run_manifest.csv"

log "Stage1 workflow completed"
log "manifest=$RUN_RESULT_DIR/run_manifest.csv"
