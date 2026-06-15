#!/usr/bin/env bash
# Run image-weather inference with Qwen + LoRA adapter + visual projector.
#
# 常用启动：
#   CUDA_VISIBLE_DEVICES=0,1 OUTPUT_DIR=outputs/vision_weather_lora_qv_v1 bash scripts/infer_vision_weather.sh
#   IMAGE=/path/to/image.jpg bash scripts/infer_vision_weather.sh
#   IMAGE_DIR=/path/to/images LIMIT=50 SAVE_JSONL=/tmp/vision_weather.jsonl bash scripts/infer_vision_weather.sh
#
# 关键参数：
#   OUTPUT_DIR       视觉 A1B1 训练输出目录，里面应有 adapter/projector.pt 或 best/adapter/best/projector.pt。
#   USE_BEST         设为 1 时加载 best 权重；设为 0 时加载最终 adapter/projector.pt。
#   IMAGE            单张图片路径。设置后优先推理这张图片。
#   IMAGE_DIR        图片目录。未设置 IMAGE 时批量推理该目录下图片。
#   SPLIT_ROOT       图像天气数据集 split 根目录。未设置 IMAGE/IMAGE_DIR 时从 split 中抽样。
#   SPLIT            使用 train/val/test 哪个 split，默认 test。
#   LIMIT            批量推理最多样本数。
#   DEVICE_MAP       Qwen 多卡切分方式，14B 推荐 auto。
#   MAX_NEW_TOKENS   生成回答的最大 token 数。

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
SPLIT_ROOT=${SPLIT_ROOT:-/home/wdz/BT/Stage1/vision_weather/data/split}
VISION_WEIGHTS=${VISION_WEIGHTS:-/home/wdz/BT/Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt}
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
  --vision-weights "$VISION_WEIGHTS"
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
