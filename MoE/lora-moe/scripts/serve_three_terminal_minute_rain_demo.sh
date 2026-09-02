#!/usr/bin/env bash
# 三终端分钟降雨反演服务。沿用前台读取的统一历史库路径。
set -euo pipefail

PYTHON=${PYTHON:-python3}

python_cmd=(
  "$PYTHON" /home/wdz/BT/Stage1/minute_rain_retrieval/service.py
  --config-002 /home/wdz/BT/Stage1/terminal_002_rain_retrieval/config.yaml
  --config-003 /home/wdz/BT/Stage1/terminal_003_rain_retrieval/config.yaml
  --checkpoint-path /home/wdz/BT/Stage1/minute_rain_retrieval/weights/deployed/position_model/best.pt
  --fallback-checkpoint-path /home/wdz/BT/Stage1/minute_rain_retrieval/weights/deployed/no_position_fallback/best.pt
  --transfer-checkpoint-path /home/wdz/BT/Stage1/minute_rain_retrieval/weights/deployed/new_terminal_transfer/best.pt
  --device cuda
  --history-db-path /home/wdz/BT/MoE/lora-moe/data/runtime/rain_retrieval_history.sqlite3
  --backup-db-001 /home/wdz/BT/Stage1/data/source_backups/terminal_001/acquisition_recovered_to_20260824.sqlite3
  --backup-db-002 /home/wdz/BT/Stage1/data/source_backups/terminal_002/phy_recovered_to_20260901.sqlite3
  --backup-db-003 /home/wdz/BT/Stage1/data/source_backups/terminal_003/phy_recovered_to_20260901.sqlite3
  --poll-interval-s 30
  --worker-lookback-hours 24
  --worker-max-samples 256
  --min-phy-points 3
  --fallback-min-phy-points 3
  --position-tolerance-s 5
  --weather-tolerance-s 5
  --image-tolerance-s 600
  --camera-input-dir /home/wdz/BT/Stage1/data/camera/images
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt
  --vision-full-csv /home/wdz/BT/Stage1/data/camera/labels/latest_weather_labels.csv
  --vision-slim-csv /home/wdz/BT/Stage1/data/camera/labels/latest_weather_labels_slim.csv
  --vision-refresh-interval-s 30
  --vision-max-images-per-refresh 8192
  --vision-batch-size 256
  --vision-num-workers 8
  --link-analysis-dir /home/wdz/BT/Stage1/link_reliability_analysis/artifacts
  --host 0.0.0.0
  --port 8041
)

"${python_cmd[@]}" "$@"
