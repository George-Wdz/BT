#!/bin/bash
# Build the default 10-minute Stage2 weather table from DB + Stage1 pass index.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
FREQ=${FREQ:-10min}
DB_PATH=${DB_PATH:-/home/wdz/satellite_data/satellite_data.db}
PASS_INDEX=${PASS_INDEX:-/home/wdz/BT/Stage1/model/data/pass_dataset.index.csv}
OUTPUT=${OUTPUT:-/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv}

"$PYTHON" build_stage2_weather_table.py \
    --db-path "$DB_PATH" \
    --pass-index "$PASS_INDEX" \
    --freq "$FREQ" \
    --output "$OUTPUT"
