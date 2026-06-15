#!/bin/bash
# Stage1 single-run launcher.
#
# Common overrides:
#   LR=5e-5 BATCH_SIZE=16 PATCH_LEN=4 STRIDE=2 bash scripts/train_default.sh
#   RAIN_FILTER_MIN=1e-6 bash scripts/train_default.sh
#   SATELLITE_IDS=1558 bash scripts/train_default.sh
#   DRY_BASELINE_ENABLED=true bash scripts/train_default.sh
#   EXTRA_SET="model.dropout=0.2 training.iterations=1" bash scripts/train_default.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/default.yaml}
PYTHON=${PYTHON:-python3}
LOG_DIR=${LOG_DIR:-logs}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/train_${TIMESTAMP}.log"}
mkdir -p "$LOG_DIR"

declare -a OVERRIDES=()
declare -a RUN_FLAGS=()

add_override() {
    local env_name=$1
    local cfg_key=$2
    local value=${!env_name-}
    if [ -n "$value" ]; then
        OVERRIDES+=(--set "${cfg_key}=${value}")
    fi
}

# Data and output paths.
add_override DB_PATH data.db_path
add_override PASS_DATASET_PATH data.pass_dataset_path
add_override CHECKPOINTS checkpoints
add_override VAL_STRATEGY data.val_strategy
add_override RAIN_FILTER_MIN data.rain_filter_min
add_override SATELLITE_IDS data.satellite_filter_ids
add_override DRY_BASELINE_ENABLED dry_baseline.enabled
add_override DRY_BASELINE_RAIN_THRESHOLD dry_baseline.rain_threshold
add_override DRY_BASELINE_METHOD dry_baseline.method
add_override DRY_BASELINE_ADD_SUMMARY dry_baseline.add_summary
add_override DRY_BASELINE_EXCLUDE_INSTANT_RAIN dry_baseline.exclude_instant_rain
add_override DRY_BASELINE_EXCLUDE_IMAGE_RAIN dry_baseline.exclude_image_rain
add_override DRY_BASELINE_IMAGE_RAIN_PROB_THRESHOLD dry_baseline.image_rain_prob_threshold
add_override DRY_BASELINE_TIME_SCALE_HOURS dry_baseline.time_scale_hours
add_override DRY_BASELINE_TIME_WEIGHT dry_baseline.time_weight
add_override DRY_BASELINE_POSITION_WEIGHT dry_baseline.position_weight

# Model shape and ablation knobs.
add_override MAX_SEQ_LEN model.max_seq_len
add_override PATCH_LEN model.patch_len
add_override STRIDE model.stride
add_override D_MODEL model.d_model
add_override N_HEADS model.n_heads
add_override E_LAYERS model.e_layers
add_override D_LAYERS model.d_layers
add_override D_FF model.d_ff
add_override DROPOUT model.dropout
add_override USE_CHANNEL_ATTENTION model.use_channel_attention

# Training knobs.
add_override SEED training.seed
add_override ITERATIONS training.iterations
add_override BATCH_SIZE training.batch_size
add_override EPOCHS training.epochs
add_override LR training.lr
add_override WEIGHT_DECAY training.weight_decay
add_override PATIENCE training.patience
add_override RAINFALL_LOSS_WEIGHT training.rainfall_loss_weight
add_override AUXILIARY_LOSS_WEIGHT training.auxiliary_loss_weight
add_override GRAD_CLIP training.grad_clip
add_override USE_COSINE training.use_cosine
add_override TMAX training.tmax
add_override LRADJ training.lradj
add_override DECAY_FAC training.decay_fac

if [ -n "${EXTRA_SET-}" ]; then
    read -r -a EXTRA_OVERRIDES <<< "$EXTRA_SET"
    for item in "${EXTRA_OVERRIDES[@]}"; do
        OVERRIDES+=(--set "$item")
    done
fi

if [ "${REBUILD_CACHE:-0}" = "1" ]; then
    CACHE_PATH=${PASS_DATASET_PATH:-/home/wdz/BT/Stage1/rain_retrieval/model/data/pass_dataset.npz}
    rm -f "$CACHE_PATH"
    echo "Removed cached pass dataset: $CACHE_PATH"
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    RUN_FLAGS+=(--dry-run)
fi

{
    echo "Stage1 training started at $(date)"
    echo "Config: $CONFIG"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "Log file: $LOG_FILE"
    echo "Overrides: ${OVERRIDES[*]:-(none)}"
    echo "========================================"
} | tee "$LOG_FILE"

"$PYTHON" main.py --config "$CONFIG" "${OVERRIDES[@]}" "${RUN_FLAGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "Stage1 training completed at $(date)" | tee -a "$LOG_FILE"
