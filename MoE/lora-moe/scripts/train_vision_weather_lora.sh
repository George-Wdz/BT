#!/usr/bin/env bash
# Launch vision-weather LoRA training.
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1 MAX_TRAIN_SAMPLES=64 MAX_STEPS=20 bash scripts/train_vision_weather_lora.sh
#   OUTPUT_DIR=outputs/vision_weather_lora_v1 EPOCHS=1 bash scripts/train_vision_weather_lora.sh
#
# 超参数速查：
#   CUDA_VISIBLE_DEVICES   使用哪些 GPU。Qwen2.5-14B bf16 通常至少 2 张 4090；正式训练建议 4 张。
#   MODEL_DIR              基座模型路径。当前默认 Qwen2.5-14B-Instruct，不建议换成 base 模型。
#   SPLIT_ROOT             图像数据 split 根目录，要求包含 train/val/test 子目录。
#   VISION_WEIGHTS         已训练天气分类器权重。首版默认冻结它作为视觉编码器。
#   OUTPUT_DIR             LoRA adapter、projector、训练参数保存目录。每次正式实验建议换新目录。
#   DEVICE_MAP             Qwen 的多卡切分方式。14B 推荐 auto；单卡只有量化后才可能考虑 none/cuda。
#   DTYPE                  Qwen 权重精度。4090 推荐 bfloat16；显卡不支持 bf16 时再用 float16。
#   BATCH_SIZE             每步每进程喂给模型的样本数。14B 先用 1，别急着加。
#   GRAD_ACCUM_STEPS       梯度累积步数。有效 batch = BATCH_SIZE * GRAD_ACCUM_STEPS。
#   EPOCHS                 训练轮数。小数据先 1 轮，防止模板化和过拟合。
#   MAX_STEPS              最大优化步数。smoke test 用 20；正式训练设 0 表示按 EPOCHS 跑完。
#   MAX_TRAIN_SAMPLES      限制训练样本数。smoke test 用 64；正式训练设 0 表示用完整 train split。
#   MAX_VAL_SAMPLES        限制验证样本数。早停/验证时使用，默认 128，避免验证太慢。
#   SAVE_STEPS             每隔多少个优化步保存一个 checkpoint。设 0 表示只保存最终产物。
#   EVAL_STEPS             每隔多少个优化步跑一次 val loss。设 0 表示不做中途验证。
#   EARLY_STOPPING_PATIENCE 早停容忍次数。必须配合 EVAL_STEPS>0；设 0 表示关闭早停。
#   LEARNING_RATE          LoRA+projector 学习率。推荐从 2e-4 起；不稳定降到 1e-4。
#   NUM_VISUAL_TOKENS      每张图映射成几个软 token。推荐 8 起步；复杂任务再试 16。
#   PROJECTOR_HIDDEN_DIM   projector MLP 隐层宽度。当前视觉特征简单，1024 足够。
#   LORA_R                 LoRA 秩。推荐 8 起步；欠拟合再试 16。
#   LORA_ALPHA             LoRA 缩放。常用 2*r，例如 r=8 时 alpha=16。
#   LORA_DROPOUT           LoRA dropout。小数据推荐 0.05；过拟合可试 0.1。
#   LORA_TARGET_MODULES    LoRA 注入层。默认只微调 q_proj,v_proj，优先保持语言能力。

set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${USER}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON=${PYTHON:-python}
MODEL_DIR=${MODEL_DIR:-/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct}
SPLIT_ROOT=${SPLIT_ROOT:-/home/wdz/BT/Stage1/vision_weather/data/split}
VISION_WEIGHTS=${VISION_WEIGHTS:-/home/wdz/BT/Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt}
OUTPUT_DIR=${OUTPUT_DIR:-/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_smoke}

DEVICE_MAP=${DEVICE_MAP:-auto}
DTYPE=${DTYPE:-bfloat16}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-20}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-64}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-128}
SAVE_STEPS=${SAVE_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-0}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-1e-4}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS:-8}
PROJECTOR_HIDDEN_DIM=${PROJECTOR_HIDDEN_DIM:-1024}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,v_proj}

LOG_DIR=${LOG_DIR:-logs}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/vision_weather_lora_${TIMESTAMP}.log"}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

declare -a CMD=(
  "$PYTHON" -m lora_moe.train.vision_weather_lora
  --model-dir "$MODEL_DIR"
  --split-root "$SPLIT_ROOT"
  --vision-weights "$VISION_WEIGHTS"
  --output-dir "$OUTPUT_DIR"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-val-samples "$MAX_VAL_SAMPLES"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  --learning-rate "$LEARNING_RATE"
  --num-visual-tokens "$NUM_VISUAL_TOKENS"
  --projector-hidden-dim "$PROJECTOR_HIDDEN_DIM"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --lora-target-modules "$LORA_TARGET_MODULES"
)

{
  echo "LoRA-MoE vision-weather training started at $(date)"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "MODEL_DIR: $MODEL_DIR"
  echo "SPLIT_ROOT: $SPLIT_ROOT"
  echo "VISION_WEIGHTS: $VISION_WEIGHTS"
  echo "OUTPUT_DIR: $OUTPUT_DIR"
  echo "DEVICE_MAP: $DEVICE_MAP"
  echo "DTYPE: $DTYPE"
  echo "BATCH_SIZE: $BATCH_SIZE"
  echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
  echo "EPOCHS: $EPOCHS"
  echo "MAX_STEPS: $MAX_STEPS"
  echo "MAX_TRAIN_SAMPLES: $MAX_TRAIN_SAMPLES"
  echo "MAX_VAL_SAMPLES: $MAX_VAL_SAMPLES"
  echo "SAVE_STEPS: $SAVE_STEPS"
  echo "EVAL_STEPS: $EVAL_STEPS"
  echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
  echo "EARLY_STOPPING_MIN_DELTA: $EARLY_STOPPING_MIN_DELTA"
  echo "LEARNING_RATE: $LEARNING_RATE"
  echo "NUM_VISUAL_TOKENS: $NUM_VISUAL_TOKENS"
  echo "PROJECTOR_HIDDEN_DIM: $PROJECTOR_HIDDEN_DIM"
  echo "LORA_R: $LORA_R"
  echo "LORA_ALPHA: $LORA_ALPHA"
  echo "LORA_DROPOUT: $LORA_DROPOUT"
  echo "LORA_TARGET_MODULES: $LORA_TARGET_MODULES"
  echo "LOG_FILE: $LOG_FILE"
  echo "Command: ${CMD[*]} $*"
  echo "========================================"
} | tee "$LOG_FILE"

"${CMD[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "LoRA-MoE vision-weather training completed at $(date)" | tee -a "$LOG_FILE"
