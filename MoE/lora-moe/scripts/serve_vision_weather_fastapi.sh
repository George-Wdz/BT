#!/usr/bin/env bash
# Serve Qwen + vision-weather LoRA-MoE expert with FastAPI.

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0,1                                                                 # 服务使用的 GPU 编号；如需沿用当前环境，删除本行
  python -m lora_moe.serve.vision_weather_fastapi
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1                 # 视觉天气 LoRA 输出目录
  --host 0.0.0.0                                                                            # FastAPI 监听地址
  --port 8010                                                                               # FastAPI 端口
  --device-map auto                                                                         # Qwen 多卡切分方式
  --dtype bfloat16                                                                          # Qwen 权重精度
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt # 冻结视觉分类器权重
  --use-best                                                                                # 加载 best adapter/projector
)

"${python_cmd[@]}" "$@"
