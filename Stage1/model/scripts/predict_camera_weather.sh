#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/home/wdz/LLaMA-Factory}"
VISION_DIR="${VISION_DIR:-$LLAMA_FACTORY_ROOT/leo_model/vision}"
INPUT_DIR="${INPUT_DIR:-/home/wdz/BT/Stage1/data/camera}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wdz/BT/Stage1/data/camera_labels}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WEIGHTS="${WEIGHTS:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M)}"

BUILD_NPZ="${BUILD_NPZ:-1}"
REBUILD_NPZ="${REBUILD_NPZ:-1}"
DB_PATH="${DB_PATH:-/home/wdz/satellite_data/satellite_data.db}"
PASS_DATASET_PATH="${PASS_DATASET_PATH:-$ROOT/data/pass_dataset_link4_img_latest.npz}"
IMAGE_TOLERANCE="${IMAGE_TOLERANCE:-10min}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$PASS_DATASET_PATH")"

timestamp="$RUN_TS"
out_csv="$OUTPUT_DIR/${timestamp}_weather_labels.csv"
slim_csv="$OUTPUT_DIR/${timestamp}_weather_labels_slim.csv"
latest_csv="$OUTPUT_DIR/latest_weather_labels.csv"
latest_slim_csv="$OUTPUT_DIR/latest_weather_labels_slim.csv"

cmd=(
  "$PYTHON_BIN"
  "$VISION_DIR/predict_weather_labels.py"
  --input-dir "$INPUT_DIR"
  --output-dir "$VISION_DIR"
  --save-csv "$out_csv"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$WEIGHTS" ]]; then
  cmd+=(--weights "$WEIGHTS")
fi

"${cmd[@]}"

cp "$out_csv" "$latest_csv"

"$PYTHON_BIN" - "$out_csv" "$slim_csv" "$latest_slim_csv" <<'PY'
import sys
import pandas as pd

src, slim_dst, latest_slim_dst = sys.argv[1:4]
cols = [
    "timestamp",
    "pred_label",
    "pred_idx",
    "confidence",
    "prob_sunny",
    "prob_cloudy",
    "prob_rain",
]
df = pd.read_csv(src, usecols=cols)
df.to_csv(slim_dst, index=False)
df.to_csv(latest_slim_dst, index=False)
PY

echo "$(date): weather labels exported: $out_csv"
echo "$(date): slim labels exported: $slim_csv"
echo "$(date): latest slim copy: $latest_slim_csv"

if [[ "$BUILD_NPZ" = "1" ]]; then
  if [[ "$REBUILD_NPZ" = "1" && -f "$PASS_DATASET_PATH" ]]; then
    rm -f "$PASS_DATASET_PATH"
    echo "$(date): removed old npz: $PASS_DATASET_PATH"
  fi

  "$PYTHON_BIN" - "$ROOT" "$DB_PATH" "$PASS_DATASET_PATH" "$slim_csv" "$IMAGE_TOLERANCE" <<'PY'
import sys
from pathlib import Path

root, db_path, output_path, image_csv, tolerance = sys.argv[1:6]
sys.path.insert(0, root)

from data.preprocessing import build_pass_dataset

feature_cols = {
    "link": ["phyRssi", "rssi", "snr", "lastCniValue"],
    "position": [
        "longitude",
        "latitude",
        "satAltitude",
        "posLongitude",
        "posLatitude",
        "altitude",
    ],
    "ground_weather": ["temperature", "humidity", "pressure"],
}
image_weather_cfg = {
    "enabled": True,
    "csv_path": image_csv,
    "tolerance": tolerance,
}

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
dataset = build_pass_dataset(
    db_path=db_path,
    output_path=output_path,
    feature_cols=feature_cols,
    strict_source_filters=False,
    image_weather_cfg=image_weather_cfg,
)
print(f"built_npz={output_path}")
print(f"passes={len(dataset)}")
PY
fi
