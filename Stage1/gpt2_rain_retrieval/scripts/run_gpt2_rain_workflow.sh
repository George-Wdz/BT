#!/usr/bin/env bash
# GPT2 baseline workflow for Stage1 rainfall retrieval.

set -euo pipefail

cd /home/wdz/BT/Stage1/gpt2_rain_retrieval

python_cmd=(
  python3 run_workflow.py
  --cuda-visible-devices 0
  --eval-cuda-visible-devices 0
  # --ddp
  --dataset-npz /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260626_1804.npz
  --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv
  --gpt2-model-dir /home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/gpt2
  --val-strategy stratified_before_test
  --position-columns "[longitude,latitude,satAltitude,posLongitude,posLatitude,altitude,slant_range_km,elevation_deg,azimuth_sin,azimuth_cos]"
  --input-dim 25
  --feature-group-dims "[4,10,3,4,4]"
  --gpt2-layers 6
  --freeze-gpt2 all
  --iterations 3
  --batch-size 32
  --epochs 100
  --patience 15
  --lr 0.0001
)

"${python_cmd[@]}" "$@"
