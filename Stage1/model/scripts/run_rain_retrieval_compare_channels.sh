#!/usr/bin/env bash
# Compare channel-mixing (cm) and channel-wise/two-stage attention (cw)
# on the same Stage1 rainfall retrieval dataset.
#
# Default flow:
#   1. Build image labels and incrementally merge new DB rows into NPZ once.
#   2. Train channel-mixing model.
#   3. Train channel-wise model.
#   4. Export prediction CSVs and MAE/MSE/RMSE metric tables.
#
# Fast reuse:
#   REUSE_DATASET=1 PASS_DATASET_PATH=/path/to/existing.npz bash scripts/run_rain_retrieval_compare_channels.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE1_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M)}"
EXPERIMENT="${EXPERIMENT:-rain_retrieval_compare_channels}"
DATASET_NAME="${DATASET_NAME:-pass_dataset_${EXPERIMENT}_${RUN_TS}}"

LABEL_DIR="${LABEL_DIR:-$STAGE1_ROOT/data/camera_labels}"
DATASET_DIR="${DATASET_DIR:-$ROOT/data/datasets}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-$ROOT/checkpoints}"
RESULT_BASE="${RESULT_BASE:-$STAGE1_ROOT/analysis/satellite_weather_diff/runs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

LABEL_CSV="$LABEL_DIR/${RUN_TS}_weather_labels.csv"
SLIM_LABEL_CSV="$LABEL_DIR/${RUN_TS}_weather_labels_slim.csv"
PASS_DATASET_PATH="${PASS_DATASET_PATH:-$DATASET_DIR/${DATASET_NAME}.npz}"
RUN_RESULT_DIR="${RUN_RESULT_DIR:-$RESULT_BASE/${DATASET_NAME}}"
WORKFLOW_LOG="${WORKFLOW_LOG:-$LOG_DIR/${DATASET_NAME}_workflow.log}"

mkdir -p "$LABEL_DIR" "$DATASET_DIR" "$RUN_RESULT_DIR" "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$WORKFLOW_LOG"
}

run_logged() {
  log "$*"
  "$@" 2>&1 | tee -a "$WORKFLOW_LOG"
}

latest_existing_npz() {
  if [[ -n "${INCREMENTAL_SOURCE_NPZ:-}" ]]; then
    echo "$INCREMENTAL_SOURCE_NPZ"
    return
  fi
  if [[ -f "$PASS_DATASET_PATH" ]]; then
    echo "$PASS_DATASET_PATH"
    return
  fi
  find "$DATASET_DIR" -maxdepth 1 -type f -name "pass_dataset_*.npz" \
    ! -path "$PASS_DATASET_PATH" -printf '%T@ %p\n' 2>/dev/null |
    sort -n |
    tail -1 |
    cut -d' ' -f2-
}

log "Stage1 channel comparison workflow started"
log "dataset_name=$DATASET_NAME"
log "run_ts=$RUN_TS"
log "dataset_npz=$PASS_DATASET_PATH"
log "results=$RUN_RESULT_DIR"

if [[ "${REUSE_DATASET:-0}" = "1" ]]; then
  if [[ ! -f "$PASS_DATASET_PATH" ]]; then
    echo "REUSE_DATASET=1 but PASS_DATASET_PATH does not exist: $PASS_DATASET_PATH" >&2
    exit 1
  fi
  log "reuse_dataset=1, skip image labeling and NPZ rebuild"
  IMAGE_LABEL_CSV="${IMAGE_LABEL_CSV:-$LABEL_DIR/latest_weather_labels_slim.csv}"
else
  export RUN_TS
  export PASS_DATASET_PATH
  export BATCH_SIZE="${VISION_BATCH_SIZE:-64}"

  if [[ "${INCREMENTAL_NPZ:-1}" = "1" ]]; then
    export BUILD_NPZ=0
    run_logged bash scripts/predict_camera_weather.sh

    if [[ ! -f "$SLIM_LABEL_CSV" ]]; then
      echo "Missing expected label CSV: $SLIM_LABEL_CSV" >&2
      exit 1
    fi

    SOURCE_NPZ="$(latest_existing_npz || true)"
    if [[ -n "$SOURCE_NPZ" && -f "$SOURCE_NPZ" ]]; then
      log "incremental_npz=1 source_npz=$SOURCE_NPZ output_npz=$PASS_DATASET_PATH"
      run_logged python3 scripts/incremental_build_pass_dataset.py \
        --db-path "${DB_PATH:-/home/wdz/satellite_data/satellite_data.db}" \
        --existing-npz "$SOURCE_NPZ" \
        --output-path "$PASS_DATASET_PATH" \
        --image-csv "$SLIM_LABEL_CSV" \
        --image-tolerance "${IMAGE_TOLERANCE:-10min}" \
        --lookback-minutes "${INCREMENTAL_LOOKBACK_MINUTES:-20}"
    else
      log "incremental_npz=1 but no source NPZ found; building full dataset"
      run_logged python3 scripts/incremental_build_pass_dataset.py \
        --db-path "${DB_PATH:-/home/wdz/satellite_data/satellite_data.db}" \
        --existing-npz "$PASS_DATASET_PATH" \
        --output-path "$PASS_DATASET_PATH" \
        --image-csv "$SLIM_LABEL_CSV" \
        --image-tolerance "${IMAGE_TOLERANCE:-10min}"
    fi
  else
    export BUILD_NPZ=1
    export REBUILD_NPZ="${REBUILD_NPZ:-1}"
    run_logged bash scripts/predict_camera_weather.sh

    if [[ ! -f "$SLIM_LABEL_CSV" ]]; then
      echo "Missing expected label CSV: $SLIM_LABEL_CSV" >&2
      exit 1
    fi
  fi
  if [[ ! -f "$PASS_DATASET_PATH" ]]; then
    echo "Missing expected dataset NPZ: $PASS_DATASET_PATH" >&2
    exit 1
  fi
  IMAGE_LABEL_CSV="$SLIM_LABEL_CSV"
fi

COMMON_EXTRA="${EXTRA_SET:-}"
COMMON_EXTRA="${COMMON_EXTRA} data.num_workers=${DATA_NUM_WORKERS:-0}"
COMMON_EXTRA="${COMMON_EXTRA} features.link=[phyRssi,rssi,snr,lastCniValue]"
COMMON_EXTRA="${COMMON_EXTRA} image_weather.enabled=true image_weather.csv_path=$IMAGE_LABEL_CSV image_weather.tolerance=${IMAGE_TOLERANCE:-10min}"
DRYBASE_EXTRA="dry_baseline.enabled=true dry_baseline.method=mean dry_baseline.exclude_instant_rain=true dry_baseline.exclude_image_rain=true dry_baseline.image_rain_prob_threshold=${DRY_BASELINE_IMAGE_RAIN_PROB_THRESHOLD:-0.5}"
INSTANT_AUX_TARGETS="targets.auxiliary=[rain_rate_mean,rain_rate_max,rainy_ratio]"
MODEL_EXTRA="model.input_dim=21 model.feature_group_dims=[4,6,3,4,4] model.use_summary_token=true"
TRAIN_EXTRA="training.auxiliary_loss_weight=${AUXILIARY_LOSS_WEIGHT:-0.3}"

train_and_eval_variant() {
  local variant="$1"
  local use_channel_attention="$2"
  local ckpt_root="$CHECKPOINT_BASE/${DATASET_NAME}_${variant}"
  local train_log="$LOG_DIR/${DATASET_NAME}_${variant}_train.log"
  local result_dir="$RUN_RESULT_DIR/${variant}"

  mkdir -p "$ckpt_root" "$result_dir"

  log "training_variant=$variant use_channel_attention=$use_channel_attention"
  export PASS_DATASET_PATH
  export CHECKPOINTS="$ckpt_root"
  export LOG_FILE="$train_log"
  export VAL_STRATEGY="${VAL_STRATEGY:-stratified_all}"
  export ITERATIONS="${ITERATIONS:-1}"
  export EPOCHS="${EPOCHS:-100}"
  export BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE:-32}}"
  export PATIENCE="${PATIENCE:-15}"
  export USE_CHANNEL_ATTENTION="$use_channel_attention"
  export EXTRA_SET="$COMMON_EXTRA $DRYBASE_EXTRA $MODEL_EXTRA model.use_channel_attention=$use_channel_attention $INSTANT_AUX_TARGETS $TRAIN_EXTRA"

  run_logged bash scripts/train_default.sh

  local ckpt_dir
  ckpt_dir="$(
    find "$ckpt_root" -type f -name checkpoint.pth -printf '%T@ %h\n' |
      sort -n |
      tail -1 |
      cut -d' ' -f2-
  )"
  if [[ -z "$ckpt_dir" || ! -f "$ckpt_dir/checkpoint.pth" ]]; then
    echo "No checkpoint.pth found under $ckpt_root" >&2
    exit 1
  fi
  log "selected_${variant}_checkpoint=$ckpt_dir"

  local pred_csv="$result_dir/${DATASET_NAME}_${variant}_predictions.csv"
  local test_csv="$result_dir/${DATASET_NAME}_${variant}_test_predictions.csv"
  local metrics_csv="$result_dir/${DATASET_NAME}_${variant}_metrics.csv"
  run_logged python3 scripts/evaluate_checkpoint_splits.py \
    --checkpoint-dir "$ckpt_dir" \
    --batch-size "${EVAL_BATCH_SIZE:-128}" \
    --out-csv "$pred_csv" \
    --test-csv "$test_csv" \
    --metrics-csv "$metrics_csv"

  {
    echo "variant,$variant"
    echo "use_channel_attention,$use_channel_attention"
    echo "dataset_name,$DATASET_NAME"
    echo "run_ts,$RUN_TS"
    echo "label_csv,$IMAGE_LABEL_CSV"
    echo "dataset_npz,$PASS_DATASET_PATH"
    echo "checkpoint_dir,$ckpt_dir"
    echo "prediction_csv,$pred_csv"
    echo "test_prediction_csv,$test_csv"
    echo "metrics_csv,$metrics_csv"
    echo "train_log,$train_log"
  } > "$result_dir/run_manifest.csv"
}

train_and_eval_variant cm false
train_and_eval_variant cw true

COMBINED_METRICS="$RUN_RESULT_DIR/combined_metrics.csv"
python3 - "$RUN_RESULT_DIR" "$COMBINED_METRICS" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
frames = []
for variant in ("cm", "cw"):
    matches = sorted((run_dir / variant).glob("*_metrics.csv"))
    if not matches:
        continue
    df = pd.read_csv(matches[-1])
    df.insert(0, "variant", variant)
    frames.append(df)
if frames:
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    pivot = combined.pivot_table(
        index=["split", "subset"],
        columns="variant",
        values=["mae", "mse", "rmse", "n"],
        aggfunc="first",
    )
    pivot.to_csv(run_dir / "combined_metrics_pivot.csv")
PY

{
  echo "dataset_name,$DATASET_NAME"
  echo "run_ts,$RUN_TS"
  echo "label_csv,$IMAGE_LABEL_CSV"
  echo "dataset_npz,$PASS_DATASET_PATH"
  echo "cm_result_dir,$RUN_RESULT_DIR/cm"
  echo "cw_result_dir,$RUN_RESULT_DIR/cw"
  echo "combined_metrics,$COMBINED_METRICS"
  echo "combined_metrics_pivot,$RUN_RESULT_DIR/combined_metrics_pivot.csv"
  echo "workflow_log,$WORKFLOW_LOG"
} > "$RUN_RESULT_DIR/run_manifest.csv"

log "Stage1 channel comparison workflow completed"
log "combined_metrics=$COMBINED_METRICS"
