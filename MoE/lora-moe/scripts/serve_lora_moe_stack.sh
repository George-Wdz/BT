#!/usr/bin/env bash
# Serve vision expert, Stage1 rainfall expert, and LoRA-MoE gateway together.
#
# 前端只访问 Gateway：
#   http://服务器IP:8020
#
# 内部专家端口：
#   vision_weather -> http://127.0.0.1:8121
#   stage1_rain    -> http://127.0.0.1:8122

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

mkdir -p logs/gateway_stack

vision_cmd=(
  python -m lora_moe.serve.vision_weather_fastapi
  --cuda-visible-devices 2,3                                                               # 视觉专家使用的 GPU 编号
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v3                 # 视觉天气 LoRA 输出目录
  --host 127.0.0.1                                                                         # 内部专家服务只监听本机
  --port 8121                                                                              # 视觉专家内部端口
  --device-map auto                                                                        # Qwen 多卡切分方式
  --dtype bfloat16                                                                         # Qwen 权重精度
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt # 冻结视觉分类器权重，需与 projector.pt 的 input_dim 匹配
  --use-best                                                                               # 加载 best adapter/projector
)

stage1_cmd=(
  python -m lora_moe.serve.stage1_rain_fastapi
  --cuda-visible-devices 0,1                                                              # Stage1 反演专家使用的 GPU 编号
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v2                  # Stage1 LoRA 输出目录
  --stage1-checkpoint-dir /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints/pass_dataset_rain_retrieval_20260612_1116/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0 # Stage1 checkpoint 目录
  --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260612_1116.npz # dry baseline 使用的 NPZ 数据集
  --db-path /home/wdz/satellite_data/satellite_data.db                                     # 在线卫星数据库路径
  --image-weather-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv # 在线图像天气标签 CSV
  --image-tolerance 10min                                                                  # 图像标签和过境时间匹配窗口
  --host 127.0.0.1                                                                         # 内部专家服务只监听本机
  --port 8122                                                                              # Stage1 反演专家内部端口
  --device-map auto                                                                        # Qwen 多卡切分方式
  --dtype bfloat16                                                                         # Qwen 权重精度
  --poll-interval-s 30                                                                     # 后台轮询数据库间隔
  --stale-after-s 180                                                                      # 最新数据超过该秒数认为无实时过境
  --lookback-hours 4                                                                       # 每次读取最近多少小时数据
  --max-passes 8                                                                           # 每次最多保留最近 pass 数
  --pass-gap-threshold-s 60                                                                # 分割 pass 的时间间隔阈值
  --min-pass-points 10                                                                     # 有效 pass 的最少采样点数
  --no-rain-threshold 0.05                                                                 # 展示口径中小于该值按无雨处理
  --use-best                                                                               # 加载 best adapter/projector
)

gateway_cmd=(
  python -m lora_moe.serve.lora_moe_gateway_fastapi
  --host 0.0.0.0                                                                           # Gateway 对外监听地址
  --port 8020                                                                              # Gateway 对外端口
  --vision-url http://127.0.0.1:8121                                                       # 视觉专家内部地址
  --stage1-url http://127.0.0.1:8122                                                       # Stage1 反演专家内部地址
  --request-timeout-s 300                                                                  # 等待后端专家推理的超时时间
)

pids=()

"${vision_cmd[@]}" > logs/gateway_stack/vision_weather.log 2>&1 &
pids+=("$!")

"${stage1_cmd[@]}" > logs/gateway_stack/stage1_rain.log 2>&1 &
pids+=("$!")

"${gateway_cmd[@]}" > logs/gateway_stack/gateway.log 2>&1 &
pids+=("$!")

echo "LoRA-MoE stack started."
echo "Gateway:        http://0.0.0.0:8020"
echo "Vision expert:  http://127.0.0.1:8121"
echo "Stage1 expert:  http://127.0.0.1:8122"
echo "Logs:           /home/wdz/BT/MoE/lora-moe/logs/gateway_stack"
echo "PIDs:           ${pids[*]}"
echo "Press Ctrl+C to stop all services."

trap 'kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true' EXIT INT TERM
wait
