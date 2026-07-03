#!/usr/bin/env bash
# Serve Qwen + vision-weather LoRA-MoE expert with FastAPI.

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

python_cmd=(
  python -m lora_moe.serve.vision_weather_fastapi
  --cuda-visible-devices 4,5                                                               # 服务使用的 GPU 编号；多卡可写 0,1,2,3
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v3                 # 视觉天气 LoRA 输出目录
  --host 0.0.0.0                                                                            # FastAPI 监听地址
  --port 8010                                                                               # FastAPI 端口
  --device-map auto                                                                         # Qwen 多卡切分方式
  --dtype bfloat16                                                                          # Qwen 权重精度
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt # 冻结视觉分类器权重，需与 projector.pt 的 input_dim 匹配
  --use-best                                                                                # 加载 best adapter/projector
)

"${python_cmd[@]}" "$@"
