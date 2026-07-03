#!/usr/bin/env bash
# Run image-weather inference with Qwen + LoRA adapter + visual projector.

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

python_cmd=(
  python -m lora_moe.infer.vision_weather
  --cuda-visible-devices 0,1                                                               # 推理使用的 GPU 编号；多卡可写 0,1,2,3
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1                 # 视觉天气 LoRA 输出目录
  --split-root /home/wdz/BT/Stage1/vision_weather/data/split                                # 默认批量推理的数据集 split 根目录
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights//home/wdz/BT/Stage1/vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt # 冻结视觉分类器权重
  --device-map auto                                                                         # Qwen 多卡切分方式
  --dtype bfloat16                                                                          # Qwen 权重精度
  --split test                                                                              # 默认从 train/val/test 哪个 split 抽样
  --limit 20                                                                                # 批量推理最多样本数
  --max-new-tokens 32                                                                       # 生成回答最大 token 数
  --use-best                                                                                # 加载 best adapter/projector
)

"${python_cmd[@]}" "$@"
