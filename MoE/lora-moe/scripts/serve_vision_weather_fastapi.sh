#!/usr/bin/env bash
# Serve Qwen + vision-weather LoRA + projector with FastAPI.

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
# Do not use $HOST here: conda/build environments may set HOST=x86_64-conda-linux-gnu.
SERVE_HOST=${SERVE_HOST:-0.0.0.0}
PORT=${PORT:-8010}
DEVICE_MAP=${DEVICE_MAP:-auto}
DTYPE=${DTYPE:-bfloat16}

declare -a CMD=(
  "$PYTHON" -m lora_moe.serve.vision_weather_fastapi
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --host "$SERVE_HOST"
  --port "$PORT"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --use-best
)

if [ -n "${ADAPTER_DIR:-}" ]; then
  CMD+=(--adapter-dir "$ADAPTER_DIR")
fi
if [ -n "${PROJECTOR_PATH:-}" ]; then
  CMD+=(--projector-path "$PROJECTOR_PATH")
fi
if [ -n "${VISION_WEIGHTS:-}" ]; then
  CMD+=(--vision-weights "$VISION_WEIGHTS")
fi

echo "Command: ${CMD[*]} $*"
"${CMD[@]}" "$@"
