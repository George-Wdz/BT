#!/usr/bin/env bash
# GPT2 baseline workflow for Stage1 rainfall retrieval.
#
# 默认复用已有 NPZ，只比较模型结构，不重新生成照片标签或重建数据集。
# 用法示例：
#   单卡：
#     bash scripts/run_gpt2_rain_workflow.sh --cuda-visible-devices 0
#
#   多卡 DDP，会自动按可见 GPU 数设置 torchrun --nproc_per_node：
#     bash scripts/run_gpt2_rain_workflow.sh --ddp --cuda-visible-devices 0,1,2,3
#
#   追加训练配置，参数会原样传给 train_gpt2_rain.py：
#     bash scripts/run_gpt2_rain_workflow.sh --ddp --cuda-visible-devices 0,1 \
#       --set data.val_strategy=time --set model.freeze_gpt2=ln_wpe
#
#   只打印命令，不启动训练：
#     bash scripts/run_gpt2_rain_workflow.sh --dry-run --ddp --cuda-visible-devices 0,1

set -euo pipefail

cd /home/wdz/BT/Stage1/gpt2_rain_retrieval

# -----------------------
# 可调配置
# -----------------------
PYTHON=python3
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"          # 训练可见 GPU；多卡写逗号分隔，如 0,1,2,3
eval_cuda_visible_devices=0                                # 评估使用单卡即可
ddp=0                                                      # 0=单进程训练；1=DDP 多卡训练
dry_run=0                                                  # 1=只打印最终命令，不执行

dataset_npz=/home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260626_1804.npz
image_label_csv=/home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv
gpt2_model_dir=/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/gpt2

val_strategy=stratified_before_test                        # stratified_all / stratified_before_test / time
gpt2_layers=6                                              # 使用 GPT2 前几层
freeze_gpt2=all                                            # all / ln_wpe / none
iterations=3
batch_size=32                                              # DDP 时表示每张 GPU 的 batch size
epochs=100
patience=15
lr=0.0001

position_columns="[longitude,latitude,satAltitude,posLongitude,posLatitude,altitude,slant_range_km,elevation_deg,azimuth_sin,azimuth_cos]"
input_dim=25
feature_group_dims="[4,10,3,4,4]"

extra_args=()
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
      extra_args+=("$1")
      shift
      ;;
  esac
done

count_visible_gpus() {
  local visible="$1"
  if [[ -n "$visible" && "$visible" != "all" ]]; then
    "$PYTHON" - "$visible" <<'PY'
import sys
visible = sys.argv[1].strip()
items = [x.strip() for x in visible.split(",") if x.strip()]
print(len(items))
PY
  else
    "$PYTHON" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
  fi
}

nproc_per_node=1
if [[ "$ddp" -eq 1 ]]; then
  nproc_per_node="$(count_visible_gpus "$cuda_visible_devices")"
  if [[ "$nproc_per_node" -lt 2 ]]; then
    echo "[ERROR] --ddp requires at least 2 visible GPUs, got: $cuda_visible_devices" >&2
    exit 1
  fi
fi

RUN_TS="$(date +%Y%m%d_%H%M)"
CHECKPOINT_BASE="/home/wdz/BT/Stage1/gpt2_rain_retrieval/checkpoints/gpt2_rain_${RUN_TS}"
RESULT_DIR="/home/wdz/BT/Stage1/gpt2_rain_retrieval/runs/gpt2_rain_${RUN_TS}"
LOG_DIR="/home/wdz/BT/Stage1/gpt2_rain_retrieval/logs"
mkdir -p "$CHECKPOINT_BASE" "$RESULT_DIR" "$LOG_DIR"

echo "[INFO] run_ts=$RUN_TS"
echo "[INFO] ddp=$ddp nproc_per_node=$nproc_per_node cuda_visible_devices=$cuda_visible_devices"
echo "[INFO] checkpoint_base=$CHECKPOINT_BASE"
echo "[INFO] result_dir=$RESULT_DIR"

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
)

if [ "$ddp" -eq 1 ]; then
  python_cmd+=(
    torchrun
    --standalone
    --nproc_per_node "$nproc_per_node"
    train_gpt2_rain.py
  )
else
  python_cmd+=(
    "$PYTHON"
    train_gpt2_rain.py
  )
fi

python_cmd+=(
  --config configs/default.yaml
  --set checkpoints="$CHECKPOINT_BASE"
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

if [[ "$dry_run" -eq 1 ]]; then
  echo "[DRY-RUN] train command:"
  printf ' %q' "${python_cmd[@]}" "${extra_args[@]}"
  printf '\n'
  echo "[DRY-RUN] eval command:"
  printf ' %q' env CUDA_VISIBLE_DEVICES="$eval_cuda_visible_devices" "$PYTHON" evaluate_gpt2_checkpoint.py \
    --checkpoint-dir '<best_checkpoint>' \
    --batch-size 128 \
    --out-csv "$RESULT_DIR/gpt2_rain_${RUN_TS}_predictions.csv" \
    --test-csv "$RESULT_DIR/gpt2_rain_${RUN_TS}_test_predictions.csv" \
    --metrics-csv "$RESULT_DIR/gpt2_rain_${RUN_TS}_metrics.csv"
  printf '\n'
  exit 0
fi

"${python_cmd[@]}" "${extra_args[@]}" | tee "$LOG_DIR/gpt2_rain_${RUN_TS}_train.log"

BEST_CHECKPOINT="$(cat "$CHECKPOINT_BASE/best_iteration_checkpoint.txt")"

env CUDA_VISIBLE_DEVICES="$eval_cuda_visible_devices" "$PYTHON" evaluate_gpt2_checkpoint.py \
  --checkpoint-dir "$BEST_CHECKPOINT" \
  --batch-size 128 \
  --out-csv "$RESULT_DIR/gpt2_rain_${RUN_TS}_predictions.csv" \
  --test-csv "$RESULT_DIR/gpt2_rain_${RUN_TS}_test_predictions.csv" \
  --metrics-csv "$RESULT_DIR/gpt2_rain_${RUN_TS}_metrics.csv" \
  | tee "$LOG_DIR/gpt2_rain_${RUN_TS}_eval.log"

{
  echo "run_ts,$RUN_TS"
  echo "ddp,$ddp"
  echo "nproc_per_node,$nproc_per_node"
  echo "cuda_visible_devices,$cuda_visible_devices"
  echo "checkpoint_base,$CHECKPOINT_BASE"
  echo "best_checkpoint,$BEST_CHECKPOINT"
  echo "result_dir,$RESULT_DIR"
  echo "train_log,$LOG_DIR/gpt2_rain_${RUN_TS}_train.log"
  echo "eval_log,$LOG_DIR/gpt2_rain_${RUN_TS}_eval.log"
} > "$RESULT_DIR/run_manifest.csv"
