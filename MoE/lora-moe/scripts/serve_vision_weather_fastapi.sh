#!/usr/bin/env bash
# Serve Qwen + vision-weather LoRA + projector with FastAPI.
#
# 常用启动：
#   CUDA_VISIBLE_DEVICES=0,1 OUTPUT_DIR=outputs/vision_weather_lora_qv_v1 bash scripts/serve_vision_weather_fastapi.sh
#   PORT=8010 SERVE_HOST=0.0.0.0 bash scripts/serve_vision_weather_fastapi.sh
#
# 关键参数：
#   OUTPUT_DIR       视觉 A1B1 训练输出目录，里面应有 adapter/projector.pt 或 best/adapter/best/projector.pt。
#   ADAPTER_DIR      手动指定 LoRA adapter 目录；设置后覆盖 OUTPUT_DIR 自动推断。
#   PROJECTOR_PATH   手动指定 projector.pt；设置后覆盖 OUTPUT_DIR 自动推断。
#   VISION_WEIGHTS   手动指定 WeatherClassifier 权重；默认读取 projector.pt 中记录的路径。
#   SERVE_HOST       FastAPI 监听地址。服务器演示通常用 0.0.0.0。
#   PORT             FastAPI 端口，默认 8010。
#   DEVICE_MAP       Qwen 多卡切分方式，14B 推荐 auto。
#   DTYPE            Qwen 权重精度，4090 推荐 bfloat16。

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
