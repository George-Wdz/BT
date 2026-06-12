#!/usr/bin/env bash
# Serve Qwen + Stage1 rainfall A2B2 LoRA + projector with FastAPI.
#
# 常用启动：
#   CUDA_VISIBLE_DEVICES=0,1,2,3 OUTPUT_DIR=outputs/stage1_rain_lora_a2b2_v1 bash scripts/serve_stage1_rain_fastapi.sh
#
# 关键参数：
#   OUTPUT_DIR          Stage1 A2B2 训练输出目录，里面应有 adapter/projector.pt 或 best/adapter/best/projector.pt。
#   USE_BEST            设为 1 时默认加载 best/adapter 和 best/projector.pt；设为 0 加载最终 adapter/projector.pt。
#   SENSOR_DB_PATH      实时卫星数据库路径。
#   POLL_INTERVAL_S     后台线程轮询数据库间隔，默认 30 秒。
#   STALE_AFTER_S       最新 phy_data 超过该秒数视为无最新卫星过境。
#   LOOKBACK_HOURS      每次轮询只读取最近多少小时 DB 数据，不全库重建。
#   MAX_PASSES          每次最多保留最近多少个 pass 做 Stage1 推理。
#   NO_RAIN_THRESHOLD   输出口径阈值。Stage1 预测小于该值时展示为无雨/0mm，默认 0.06。
#   DEVICE_MAP          Qwen 多卡切分方式，14B 推荐 auto。

set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${USER}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON=${PYTHON:-python}
MODEL_DIR=${MODEL_DIR:-/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1}
SENSOR_DB_PATH=${SENSOR_DB_PATH:-/home/wdz/satellite_data/satellite_data.db}
IMAGE_WEATHER_CSV=${IMAGE_WEATHER_CSV:-/home/wdz/BT/Stage1/data/camera_labels/latest_weather_labels_slim.csv}
IMAGE_TOLERANCE=${IMAGE_TOLERANCE:-10min}

# Do not use $HOST here: conda/build environments may set HOST=x86_64-conda-linux-gnu.
SERVE_HOST=${SERVE_HOST:-0.0.0.0}
PORT=${PORT:-8011}
DEVICE_MAP=${DEVICE_MAP:-auto}
DTYPE=${DTYPE:-bfloat16}
USE_BEST=${USE_BEST:-1}
POLL_INTERVAL_S=${POLL_INTERVAL_S:-30}
STALE_AFTER_S=${STALE_AFTER_S:-180}
LOOKBACK_HOURS=${LOOKBACK_HOURS:-4}
MAX_PASSES=${MAX_PASSES:-8}
PASS_GAP_THRESHOLD_S=${PASS_GAP_THRESHOLD_S:-60}
MIN_PASS_POINTS=${MIN_PASS_POINTS:-10}
NO_RAIN_THRESHOLD=${NO_RAIN_THRESHOLD:-0.06}

declare -a CMD=(
  "$PYTHON" -m lora_moe.serve.stage1_rain_fastapi
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --db-path "$SENSOR_DB_PATH"
  --image-weather-csv "$IMAGE_WEATHER_CSV"
  --image-tolerance "$IMAGE_TOLERANCE"
  --host "$SERVE_HOST"
  --port "$PORT"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
  --poll-interval-s "$POLL_INTERVAL_S"
  --stale-after-s "$STALE_AFTER_S"
  --lookback-hours "$LOOKBACK_HOURS"
  --max-passes "$MAX_PASSES"
  --pass-gap-threshold-s "$PASS_GAP_THRESHOLD_S"
  --min-pass-points "$MIN_PASS_POINTS"
  --no-rain-threshold "$NO_RAIN_THRESHOLD"
)

if [[ "$USE_BEST" = "1" ]]; then
  CMD+=(--use-best)
fi
if [[ -n "${ADAPTER_DIR:-}" ]]; then
  CMD+=(--adapter-dir "$ADAPTER_DIR")
fi
if [[ -n "${PROJECTOR_PATH:-}" ]]; then
  CMD+=(--projector-path "$PROJECTOR_PATH")
fi

echo "Command: ${CMD[*]} $*"
"${CMD[@]}" "$@"
