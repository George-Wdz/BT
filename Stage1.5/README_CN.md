# Stage1.5：pass 结果转规则时序表

[English](README.md) | [中文](README_CN.md)

Stage1.5 将 Stage1 产生的不规则 pass 级降雨结果聚合到规则时间网格，生成 Stage2 可读取的时序表。它不训练模型。

## 输入

| 来源 | 用途 |
| --- | --- |
| `weather_station` 表 | 聚合常规气象特征，并计算固定窗口累计降雨目标。 |
| Stage1 pass index 或预测 CSV | 将 pass 级反演结果聚合为时间桶特征。 |

默认 pass 输入：

```text
/home/wdz/BT/Stage1/rain_retrieval/model/data/pass_dataset.index.csv
```

正式训练 Stage2 时，应优先使用 Stage1 模型预测列，例如 `pred_pass_rainfall_mm`，而不是 oracle 标签。

## 目标

对于结束时刻为 `t`、宽度为 `freq` 的时间桶：

```text
rain_window_mm = rainfall_cumulative(t) - rainfall_cumulative(t - freq)
```

默认 10 分钟表的目标列为 `rain_10min_mm`。

## Stage1 特征聚合

pass 按结束时间归入时间桶：

```text
bucket = ceil(pass_end, freq)
```

这个口径符合在线可用性：只有 pass 结束后，完整 pass 的反演结果才可使用。

聚合特征包括：

| 特征 | 说明 |
| --- | --- |
| `stage1_rain_sum` | 桶内 pass 雨量之和 |
| `stage1_rain_mean` | 桶内 pass 雨量均值 |
| `stage1_rain_max` | 桶内 pass 雨量最大值 |
| `stage1_pass_count` | 桶内 pass 数 |
| `stage1_has_pass` | 桶内是否有 pass |

## 输出

输出表符合 GPT4TS `Dataset_Custom` 格式：

```text
date, feature_1, ..., feature_n, target
```

默认输出：

```text
Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
```

生成的 CSV 和 summary 文件暂不纳入本仓库，后续可随数据集版本单独发布。

## 使用

生成默认 10 分钟表：

```bash
cd /home/wdz/BT/Stage1.5
bash scripts/build_default.sh
```

生成 1 小时表：

```bash
FREQ=1h \
OUTPUT=/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_1h.csv \
bash scripts/build_default.sh
```

直接调用 Python：

```bash
python3 build_stage2_weather_table.py \
  --db-path /home/wdz/satellite_data/satellite_data.db \
  --pass-index /home/wdz/BT/Stage1/rain_retrieval/model/data/pass_dataset.index.csv \
  --freq 10min \
  --output /home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
```
