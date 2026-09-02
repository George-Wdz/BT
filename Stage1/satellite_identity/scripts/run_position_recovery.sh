#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

"$PYTHON" "$ROOT/audit_raw_position_quality.py" \
  --csv-path /home/wdz/BT/Stage1/position_data.csv \
  --output-dir "$ROOT/analysis/position_quality"

# Only positions with an accepted physical identity, <=10 km measured ECEF
# anchor error, and a historical TLE within 7 days are exported.
"$PYTHON" "$ROOT/repair_raw_positions_from_tle.py" \
  --csv-path /home/wdz/BT/Stage1/position_data.csv \
  --output-path "$ROOT/analysis/position_recovery/repaired_positions.csv" \
  --max-anchor-error-km 10 \
  --max-tle-age-days 7
