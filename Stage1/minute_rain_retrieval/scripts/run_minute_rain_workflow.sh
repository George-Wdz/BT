#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# --rebuild-dataset: 1=用当前数据库重建最新 NPZ；0=复用 dataset-path 的已有 NPZ。
# --evaluate-only: 1=跳过训练，使用 output-dir/best.pt 导出预测；正常训练设为 0。
# --max-train-dry-ratio: 3=每轮保留全部有雨样本，随机抽取最多 3 倍无雨样本；-1=不下采样。
# --min-snr-db: 构建阶段硬过滤低 SNR 点；默认不设置，避免漏掉强降雨弱信号。
# --snr-quality-mode: none=原始输入；hard_mask=模型内硬掩码；soft_gate=模型内连续质量门控。
# 每次运行自动建立“时间+数据配置+划分配置”目录，并归档 full/train/val/test。
# --dataset-path 仅在 --rebuild-dataset 0 时用于指定待复用的数据；重建时无需填写。
python3 run_workflow.py \
  --mode all \
  --rebuild-dataset 1 \
  --evaluate-only 0 \
  --db-path /home/wdz/satellite_data/satellite_data.db \
  --archive-root /home/wdz/BT/Stage1/minute_rain_retrieval/data/training_runs \
  --image-csv /home/wdz/BT/Stage1/data/camera/labels/latest_weather_labels_slim.csv \
  --terminal-id 01-31-0005-0001 \
  --min-phy-points 3 \
  --position-tolerance-seconds 5 \
  --weather-tolerance-seconds 5 \
  --image-tolerance-seconds 600 \
  --split-strategy stratified_all \
  --split-seed 42 \
  --archive-quality-views 0 \
  --cuda-visible-devices 0 \
  --epochs 80 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --patience 12 \
  --max-train-dry-ratio 3 \
  --selection-metric balanced_mae \
  --heavy-rain-threshold 0.1 \
  --heavy-rain-loss-weight 2.0 \
  --probability-threshold 0.5 \
  --classification-weight 0.5 \
  --snr-quality-mode none \
  --snr-threshold-db -10 \
  --snr-gate-temperature-db 2 \
  "$@"
