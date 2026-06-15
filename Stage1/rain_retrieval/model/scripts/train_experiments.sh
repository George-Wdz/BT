#!/bin/bash
# Reproducible Stage1 experiment launcher.
#
# Recommended:
#   bash scripts/train_experiments.sh rain_retrieval
#
# Ablations:
#   bash scripts/train_experiments.sh ablate_no_instant_aux
#   bash scripts/train_experiments.sh ablate_no_summary
#   bash scripts/train_experiments.sh ablate_no_drybase
#
# Common overrides:
#   VAL_STRATEGY=stratified_all ITERATIONS=1 bash scripts/train_experiments.sh rain_retrieval
#   EPOCHS=150 LR=5e-5 AUXILIARY_LOSS_WEIGHT=0.2 bash scripts/train_experiments.sh rain_retrieval

set -euo pipefail

EXP=${1:-rain_retrieval}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
IMAGE_LABEL_CSV=${IMAGE_LABEL_CSV:-/home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv}
PASS_DATASET_DEFAULT="$ROOT/data/pass_dataset_link4_img_latest.npz"

COMMON_EXTRA=${EXTRA_SET:-}
COMMON_EXTRA="${COMMON_EXTRA} data.num_workers=${DATA_NUM_WORKERS:-0}"
COMMON_EXTRA="${COMMON_EXTRA} features.link=[phyRssi,rssi,snr,lastCniValue]"
COMMON_EXTRA="${COMMON_EXTRA} image_weather.enabled=true image_weather.csv_path=$IMAGE_LABEL_CSV image_weather.tolerance=10min"

DRYBASE_EXTRA="dry_baseline.enabled=true dry_baseline.method=mean dry_baseline.exclude_instant_rain=true dry_baseline.exclude_image_rain=true dry_baseline.image_rain_prob_threshold=${DRY_BASELINE_IMAGE_RAIN_PROB_THRESHOLD:-0.5}"
INSTANT_AUX_TARGETS="targets.auxiliary=[rain_rate_mean,rain_rate_max,rainy_ratio]"

case "$EXP" in
  rain_retrieval|link4_img_drybase_instant_aux_cm)
    export USE_CHANNEL_ATTENTION=false
    export CHECKPOINTS=${CHECKPOINTS:-"$ROOT/checkpoints_stage1_rain_retrieval"}
    export PASS_DATASET_PATH=${PASS_DATASET_PATH:-"$PASS_DATASET_DEFAULT"}
    export LOG_FILE=${LOG_FILE:-"$ROOT/logs/train_${RUN_TS}_${EXP}.log"}
    export EXTRA_SET="$COMMON_EXTRA $DRYBASE_EXTRA model.input_dim=21 model.feature_group_dims=[4,6,3,4,4] model.use_summary_token=true $INSTANT_AUX_TARGETS training.auxiliary_loss_weight=${AUXILIARY_LOSS_WEIGHT:-0.3}"
    ;;

  ablate_no_instant_aux|link4_img_drybase_latest_summary_cm)
    export USE_CHANNEL_ATTENTION=false
    export CHECKPOINTS=${CHECKPOINTS:-"$ROOT/checkpoints_stage1_ablate_no_instant_aux"}
    export PASS_DATASET_PATH=${PASS_DATASET_PATH:-"$PASS_DATASET_DEFAULT"}
    export LOG_FILE=${LOG_FILE:-"$ROOT/logs/train_${RUN_TS}_${EXP}.log"}
    export EXTRA_SET="$COMMON_EXTRA $DRYBASE_EXTRA model.input_dim=21 model.feature_group_dims=[4,6,3,4,4] model.use_summary_token=true targets.auxiliary=[]"
    ;;

  ablate_no_summary|link4_img_drybase_latest_cm)
    export USE_CHANNEL_ATTENTION=false
    export CHECKPOINTS=${CHECKPOINTS:-"$ROOT/checkpoints_stage1_ablate_no_summary"}
    export PASS_DATASET_PATH=${PASS_DATASET_PATH:-"$PASS_DATASET_DEFAULT"}
    export LOG_FILE=${LOG_FILE:-"$ROOT/logs/train_${RUN_TS}_${EXP}.log"}
    export EXTRA_SET="$COMMON_EXTRA $DRYBASE_EXTRA model.input_dim=21 model.feature_group_dims=[4,6,3,4,4] model.use_summary_token=false targets.auxiliary=[]"
    ;;

  ablate_no_drybase|link4_img_latest_cm)
    export USE_CHANNEL_ATTENTION=false
    export CHECKPOINTS=${CHECKPOINTS:-"$ROOT/checkpoints_stage1_ablate_no_drybase"}
    export PASS_DATASET_PATH=${PASS_DATASET_PATH:-"$PASS_DATASET_DEFAULT"}
    export LOG_FILE=${LOG_FILE:-"$ROOT/logs/train_${RUN_TS}_${EXP}.log"}
    export DRY_BASELINE_ENABLED=false
    export EXTRA_SET="$COMMON_EXTRA dry_baseline.enabled=false model.input_dim=17 model.feature_group_dims=[4,6,3,4] model.use_summary_token=true $INSTANT_AUX_TARGETS training.auxiliary_loss_weight=${AUXILIARY_LOSS_WEIGHT:-0.3}"
    ;;

  *)
    echo "Unknown experiment: $EXP" >&2
    echo "Available: rain_retrieval ablate_no_instant_aux ablate_no_summary ablate_no_drybase" >&2
    exit 2
    ;;
esac

export ITERATIONS=${ITERATIONS:-3}

bash scripts/train_default.sh
