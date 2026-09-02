#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python3 "$ROOT/build_identity_map.py" \
  --db-path /home/wdz/satellite_data/satellite_data.db \
  --output-dir "$ROOT/analysis/latest"

python3 "$ROOT/assess_position_recovery.py" \
  --db-path /home/wdz/satellite_data/satellite_data.db \
  --mapping-path "$ROOT/analysis/latest/satellite_id_mapping.csv" \
  --daily-catalog-path "$ROOT/analysis/latest/daily_position_catalog_scores.csv" \
  --tolerance-seconds 5 \
  --output-path "$ROOT/analysis/latest/position_recovery_audit.json"
