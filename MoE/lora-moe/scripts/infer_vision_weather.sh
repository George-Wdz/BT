#!/usr/bin/env bash
# Run image-weather inference with Qwen + LoRA adapter + visual projector.

set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${USER}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON=${PYTHON:-python}
MODEL_DIR=${MODEL_DIR:-/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1}
SPLIT_ROOT=${SPLIT_ROOT:-/home/wdz/LLaMA-Factory/leo_model/data/vision_weather/split}
DEVICE_MAP=${DEVICE_MAP:-auto}
DTYPE=${DTYPE:-bfloat16}
SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
USE_BEST=${USE_BEST:-1}
SAVE_JSONL=${SAVE_JSONL:-}

declare -a CMD=(
  "$PYTHON" -m lora_moe.infer.vision_weather
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --split-root "$SPLIT_ROOT"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --split "$SPLIT"
  --limit "$LIMIT"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [ "$USE_BEST" = "1" ]; then
  CMD+=(--use-best)
fi
if [ -n "${IMAGE:-}" ]; then
  CMD+=(--image "$IMAGE")
fi
if [ -n "${IMAGE_DIR:-}" ]; then
  CMD+=(--image-dir "$IMAGE_DIR")
fi
if [ -n "$SAVE_JSONL" ]; then
  CMD+=(--save-jsonl "$SAVE_JSONL")
fi

echo "Command: ${CMD[*]} $*"
"${CMD[@]}" "$@"

