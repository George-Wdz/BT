#!/usr/bin/env bash
# GPT2 baseline workflow for Stage1 rainfall retrieval.
#
# Examples:
#   bash scripts/run_gpt2_rain_workflow.sh --cuda-visible-devices 0
#   bash scripts/run_gpt2_rain_workflow.sh --ddp --cuda-visible-devices 0,1,2,3
#   bash scripts/run_gpt2_rain_workflow.sh --dry-run --ddp --cuda-visible-devices 0,1
#
# Extra arguments are forwarded to train_gpt2_rain.py, for example:
#   bash scripts/run_gpt2_rain_workflow.sh --ddp --cuda-visible-devices 0,1 \
#     --set data.val_strategy=time --set model.freeze_gpt2=ln_wpe

set -euo pipefail

cd /home/wdz/BT/Stage1/gpt2_rain_retrieval

# -----------------------------
# Runtime options
# -----------------------------
PYTHON=python3
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"    # Training GPUs, e.g. 0 or 0,1,2,3
eval_cuda_visible_devices=0                          # Evaluation uses one GPU
ddp=0                                                # 0=single process, 1=torchrun DDP
dry_run=0                                            # 1=print commands only

# -----------------------------
# Paths
# -----------------------------
run_ts="$(date +%Y%m%d_%H%M)"
checkpoint_base="/home/wdz/BT/Stage1/gpt2_rain_retrieval/checkpoints/gpt2_rain_${run_ts}"
result_dir="/home/wdz/BT/Stage1/gpt2_rain_retrieval/runs/gpt2_rain_${run_ts}"
log_dir="/home/wdz/BT/Stage1/gpt2_rain_retrieval/logs"

dataset_npz="/home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260626_1804.npz"
image_label_csv="/home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv"
gpt2_model_dir="/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/gpt2"

# -----------------------------
# Experiment options
# -----------------------------
val_strategy=stratified_before_test                  # stratified_all / stratified_before_test / time
position_columns="[longitude,latitude,satAltitude,posLongitude,posLatitude,altitude,slant_range_km,elevation_deg,azimuth_sin,azimuth_cos]"
input_dim=25
feature_group_dims="[4,10,3,4,4]"

gpt2_layers=6
freeze_gpt2=all                                      # all / ln_wpe / none

iterations=3
batch_size=32                                        # Per-GPU batch size when --ddp is enabled
epochs=100
patience=15
lr=0.0001

# -----------------------------
# Script-only flags
# -----------------------------
train_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ddp)
      ddp=1
      shift
      ;;
    --no-ddp)
      ddp=0
      shift
      ;;
    --cuda-visible-devices)
      cuda_visible_devices="$2"
      shift 2
      ;;
    --eval-cuda-visible-devices)
      eval_cuda_visible_devices="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      train_args+=("$1")
      shift
      ;;
  esac
done

count_visible_gpus() {
  local visible="${1// /}"
  local old_ifs="$IFS"
  IFS=","
  read -ra ids <<< "$visible"
  IFS="$old_ifs"
  echo "${#ids[@]}"
}

nproc_per_node=1
launcher=("$PYTHON")
if [[ "$ddp" -eq 1 ]]; then
  nproc_per_node="$(count_visible_gpus "$cuda_visible_devices")"
  if [[ "$nproc_per_node" -lt 2 ]]; then
    echo "[ERROR] --ddp needs at least two visible GPUs; got '$cuda_visible_devices'." >&2
    exit 1
  fi
  launcher=(torchrun --standalone --nproc_per_node "$nproc_per_node")
fi

mkdir -p "$checkpoint_base" "$result_dir" "$log_dir"

echo "[INFO] run_ts=$run_ts"
echo "[INFO] ddp=$ddp nproc_per_node=$nproc_per_node cuda_visible_devices=$cuda_visible_devices"
echo "[INFO] checkpoint_base=$checkpoint_base"
echo "[INFO] result_dir=$result_dir"

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
  "${launcher[@]}" train_gpt2_rain.py
  --config configs/default.yaml
  --set checkpoints="$checkpoint_base"
  --set data.pass_dataset_path="$dataset_npz"
  --set data.val_strategy="$val_strategy"
  --set image_weather.csv_path="$image_label_csv"
  --set "features.position=$position_columns"
  --set model.input_dim="$input_dim"
  --set "model.feature_group_dims=$feature_group_dims"
  --set model.gpt2_model_dir="$gpt2_model_dir"
  --set model.gpt2_layers="$gpt2_layers"
  --set model.freeze_gpt2="$freeze_gpt2"
  --set training.iterations="$iterations"
  --set training.batch_size="$batch_size"
  --set training.epochs="$epochs"
  --set training.patience="$patience"
  --set training.lr="$lr"
)

eval_cmd=(
  env
  CUDA_VISIBLE_DEVICES="$eval_cuda_visible_devices"
  "$PYTHON" evaluate_gpt2_checkpoint.py
  --checkpoint-dir "<best_checkpoint>"
  --batch-size 128
  --out-csv "$result_dir/gpt2_rain_${run_ts}_predictions.csv"
  --test-csv "$result_dir/gpt2_rain_${run_ts}_test_predictions.csv"
  --metrics-csv "$result_dir/gpt2_rain_${run_ts}_metrics.csv"
)

if [[ "$dry_run" -eq 1 ]]; then
  echo "[DRY-RUN] train command:"
  printf ' %q' "${python_cmd[@]}" "${train_args[@]}"
  printf '\n'
  echo "[DRY-RUN] eval command:"
  printf ' %q' "${eval_cmd[@]}"
  printf '\n'
  exit 0
fi

"${python_cmd[@]}" "${train_args[@]}" | tee "$log_dir/gpt2_rain_${run_ts}_train.log"

best_checkpoint="$(cat "$checkpoint_base/best_iteration_checkpoint.txt")"
eval_cmd[5]="$best_checkpoint"

"${eval_cmd[@]}" | tee "$log_dir/gpt2_rain_${run_ts}_eval.log"

{
  echo "run_ts,$run_ts"
  echo "ddp,$ddp"
  echo "nproc_per_node,$nproc_per_node"
  echo "cuda_visible_devices,$cuda_visible_devices"
  echo "checkpoint_base,$checkpoint_base"
  echo "best_checkpoint,$best_checkpoint"
  echo "result_dir,$result_dir"
  echo "train_log,$log_dir/gpt2_rain_${run_ts}_train.log"
  echo "eval_log,$log_dir/gpt2_rain_${run_ts}_eval.log"
} > "$result_dir/run_manifest.csv"
