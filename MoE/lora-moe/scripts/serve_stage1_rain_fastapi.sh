#!/usr/bin/env bash
# Serve Qwen + Stage1 rainfall LoRA-MoE expert with FastAPI.

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

python_cmd=(
  python -m lora_moe.serve.stage1_rain_fastapi
  --cuda-visible-devices 0,1,2,3                                                           # 服务使用的 GPU 编号；多卡可写 0,1,2,3
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v2                  # Stage1 LoRA 输出目录
  --stage1-checkpoint-dir /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints/pass_dataset_rain_retrieval_20260612_1116/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0 # Stage1 checkpoint 目录
  --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260612_1116.npz # dry baseline 使用的 NPZ 数据集
  --db-path /home/wdz/satellite_data/satellite_data.db                                      # 在线卫星数据库路径
  --image-weather-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv # 在线图像天气标签 CSV
  --image-tolerance 10min                                                                   # 图像标签和过境时间匹配窗口
  --host 0.0.0.0                                                                            # FastAPI 监听地址
  --port 8011                                                                               # FastAPI 端口
  --device-map auto                                                                         # Qwen 多卡切分方式
  --dtype bfloat16                                                                          # Qwen 权重精度
  --poll-interval-s 30                                                                      # 后台轮询数据库间隔
  --stale-after-s 180                                                                       # 最新数据超过该秒数认为无实时过境
  --lookback-hours 4                                                                        # 每次读取最近多少小时数据
  --max-passes 8                                                                            # 每次最多保留最近 pass 数
  --pass-gap-threshold-s 60                                                                 # 分割 pass 的时间间隔阈值
  --min-pass-points 10                                                                      # 有效 pass 的最少采样点数
  --no-rain-threshold 0.05                                                                  # 展示口径中小于该值按无雨处理
  --use-best                                                                                # 加载 best adapter/projector
)

"${python_cmd[@]}" "$@"
