# Stage1.5: Pass-to-Series Bridge

[English](README.md) | [中文](README_CN.md)

Stage1.5 converts irregular Stage1 pass-level rainfall outputs into regular time-series tables for Stage2 forecasting. It does not train a model.

## Inputs

| Source | Use |
| --- | --- |
| `weather_station` table | Regular weather aggregation and fixed-window rainfall targets. |
| Stage1 pass index or prediction CSV | Pass-level retrieval features aggregated by time bucket. |

Default pass input:

```text
/home/wdz/BT/Stage1/rain_retrieval/model/data/pass_dataset.index.csv
```

For formal Stage2 training, prefer Stage1 model predictions such as `pred_pass_rainfall_mm` instead of oracle labels.

## Target

For a bucket ending at time `t` with width `freq`:

```text
rain_window_mm = rainfall_cumulative(t) - rainfall_cumulative(t - freq)
```

For the default 10-minute table, the target is `rain_10min_mm`.

## Stage1 Feature Aggregation

Passes are assigned by pass end time:

```text
bucket = ceil(pass_end, freq)
```

This matches online availability: a complete pass can only be used after it ends.

Aggregated Stage1 features include:

| Feature | Description |
| --- | --- |
| `stage1_rain_sum` | Sum of pass-level rainfall in the bucket. |
| `stage1_rain_mean` | Mean pass-level rainfall. |
| `stage1_rain_max` | Maximum pass-level rainfall. |
| `stage1_pass_count` | Number of passes in the bucket. |
| `stage1_has_pass` | Whether the bucket contains at least one pass. |

## Output

The output follows the GPT4TS `Dataset_Custom` format:

```text
date, feature_1, ..., feature_n, target
```

Default output:

```text
Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
```

Generated CSV and summary files are kept outside this repository and can be published separately with the corresponding dataset release.

## Usage

Generate the default 10-minute table:

```bash
cd /home/wdz/BT/Stage1.5
bash scripts/build_default.sh
```

Generate a 1-hour table:

```bash
FREQ=1h \
OUTPUT=/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_1h.csv \
bash scripts/build_default.sh
```

Direct Python call:

```bash
python3 build_stage2_weather_table.py \
  --db-path /home/wdz/satellite_data/satellite_data.db \
  --pass-index /home/wdz/BT/Stage1/rain_retrieval/model/data/pass_dataset.index.csv \
  --freq 10min \
  --output /home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
```
