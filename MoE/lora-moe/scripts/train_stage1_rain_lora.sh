#!/usr/bin/env bash
# Launch Stage1 rainfall retrieval LoRA training.
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1 MAX_TRAIN_SAMPLES=64 MAX_STEPS=20 bash scripts/train_stage1_rain_lora.sh
#   OUTPUT_DIR=outputs/stage1_rain_lora_v1 EPOCHS=1 MAX_STEPS=0 bash scripts/train_stage1_rain_lora.sh
#
# 超参数速查：
#   CUDA_VISIBLE_DEVICES     使用哪些 GPU。Qwen2.5-14B bf16 建议至少 2 张 4090，正式训练建议 4 张以上。
#   MODEL_DIR                Qwen2.5-14B-Instruct 基座路径。保持 Instruct 版本以保留中文回答能力。
#   STAGE1_CHECKPOINT_DIR    Stage1 反演模型 checkpoint 目录，必须包含 checkpoint.pth 和 meta.pt。
#   PASS_DATASET_PATH        Stage1 NPZ 数据集路径。留空时使用 Stage1 meta.pt 里记录的数据集。
#   OUTPUT_DIR               A2B2 LoRA adapter、Stage1 projector、训练参数保存目录。
#   DEVICE_MAP               Qwen 多卡切分方式。14B 推荐 auto。
#   DTYPE                    权重精度。4090 推荐 bfloat16。
#   BATCH_SIZE               每步样本数。14B 默认 1，显存足够也先通过 GRAD_ACCUM_STEPS 增大有效 batch。
#   GRAD_ACCUM_STEPS         梯度累积步数。有效 batch = BATCH_SIZE * GRAD_ACCUM_STEPS。
#   EPOCHS                   训练轮数。演示版 1 轮起步；Stage1 数据变好后再增加。
#   MAX_STEPS                最大优化步数。smoke/demo 用 20-100；设 0 表示按 EPOCHS 跑完。
#   MAX_TRAIN_SAMPLES        限制训练样本数。smoke 用 64；正式设 0。
#   MAX_VAL_SAMPLES          限制验证样本数。默认 128，避免验证太慢。
#   SAVE_STEPS               每隔多少优化步保存 checkpoint。设 0 表示只保存最终产物。
#   EVAL_STEPS               每隔多少优化步跑 val loss。设 0 表示不做中途验证。
#   EARLY_STOPPING_PATIENCE  早停容忍次数。必须配合 EVAL_STEPS>0。
#   LEARNING_RATE            LoRA+projector 学习率。推荐 2e-4；输出不稳定降到 1e-4。
#   NUM_STAGE1_TOKENS        Stage1 反演特征映射成几个 soft token。推荐 8 起步。
#   PROJECTOR_HIDDEN_DIM     projector MLP 隐层宽度。Stage1 特征较小，1024 足够。
#   LORA_R                   LoRA 秩。A2B2 演示版推荐 8；欠拟合再试 16。
#   LORA_ALPHA               LoRA 缩放。常用 2*r，例如 r=8 时 alpha=16。
#   LORA_DROPOUT             小数据推荐 0.05，过拟合可试 0.1。
#   LORA_TARGET_MODULES      LoRA 注入层。默认 q_proj,v_proj，优先保持 Qwen 语言能力。
#   NO_RAIN_THRESHOLD        输出口径阈值。Stage1 预测小于该值时训练回答记为无雨/0mm，默认 0.05。

set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${USER}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON=${PYTHON:-python}
MODEL_DIR=${MODEL_DIR:-/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct}
STAGE1_CHECKPOINT_DIR=${STAGE1_CHECKPOINT_DIR:-/home/wdz/BT/Stage1/model/checkpoints/pass_dataset_rain_retrieval_compare_channels_compare_cm_cw_20260612_1140_cm/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0}
PASS_DATASET_PATH=${PASS_DATASET_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_smoke}

DEVICE_MAP=${DEVICE_MAP:-auto}
DTYPE=${DTYPE:-bfloat16}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-20}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-64}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-128}
NUM_WORKERS=${NUM_WORKERS:-0}
SAVE_STEPS=${SAVE_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-0}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-1e-4}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_STAGE1_TOKENS=${NUM_STAGE1_TOKENS:-8}
PROJECTOR_HIDDEN_DIM=${PROJECTOR_HIDDEN_DIM:-1024}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,v_proj}
NO_RAIN_THRESHOLD=${NO_RAIN_THRESHOLD:-0.05}

LOG_DIR=${LOG_DIR:-logs}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/stage1_rain_lora_${TIMESTAMP}.log"}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

declare -a CMD=(
  "$PYTHON" -m lora_moe.train.stage1_rain_lora
  --model-dir "$MODEL_DIR"
  --stage1-checkpoint-dir "$STAGE1_CHECKPOINT_DIR"
  --pass-dataset-path "$PASS_DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-val-samples "$MAX_VAL_SAMPLES"
  --num-workers "$NUM_WORKERS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  --learning-rate "$LEARNING_RATE"
  --num-stage1-tokens "$NUM_STAGE1_TOKENS"
  --projector-hidden-dim "$PROJECTOR_HIDDEN_DIM"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --lora-target-modules "$LORA_TARGET_MODULES"
  --no-rain-threshold "$NO_RAIN_THRESHOLD"
)

{
  echo "LoRA-MoE Stage1 rainfall training started at $(date)"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "MODEL_DIR: $MODEL_DIR"
  echo "STAGE1_CHECKPOINT_DIR: $STAGE1_CHECKPOINT_DIR"
  echo "PASS_DATASET_PATH: ${PASS_DATASET_PATH:-<from meta.pt>}"
  echo "OUTPUT_DIR: $OUTPUT_DIR"
  echo "DEVICE_MAP: $DEVICE_MAP"
  echo "DTYPE: $DTYPE"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
  echo "EPOCHS: $EPOCHS"
  echo "MAX_STEPS: $MAX_STEPS"
  echo "MAX_TRAIN_SAMPLES: $MAX_TRAIN_SAMPLES"
  echo "MAX_VAL_SAMPLES: $MAX_VAL_SAMPLES"
  echo "NUM_WORKERS: $NUM_WORKERS"
  echo "SAVE_STEPS: $SAVE_STEPS"
  echo "EVAL_STEPS: $EVAL_STEPS"
  echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
  echo "EARLY_STOPPING_MIN_DELTA: $EARLY_STOPPING_MIN_DELTA"
  echo "LEARNING_RATE: $LEARNING_RATE"
  echo "NUM_STAGE1_TOKENS: $NUM_STAGE1_TOKENS"
  echo "PROJECTOR_HIDDEN_DIM: $PROJECTOR_HIDDEN_DIM"
  echo "LORA_R: $LORA_R"
  echo "LORA_ALPHA: $LORA_ALPHA"
  echo "LORA_DROPOUT: $LORA_DROPOUT"
  echo "LORA_TARGET_MODULES: $LORA_TARGET_MODULES"
  echo "NO_RAIN_THRESHOLD: $NO_RAIN_THRESHOLD"
  echo "LOG_FILE: $LOG_FILE"
  echo "Command: ${CMD[*]} $*"
  echo "========================================"
} | tee "$LOG_FILE"

"${CMD[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "LoRA-MoE Stage1 rainfall training completed at $(date)" | tee -a "$LOG_FILE"
