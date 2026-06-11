# Stage1.5: Pass 反演结果转规则时序表

Stage1.5 是 Stage1 和 Stage2 之间的桥接层。它不训练模型，只负责把不规则的卫星 pass 结果落到规则时间网格，生成 Stage2 可读取的 CSV。

## 1. 输入

### 气象站数据

来自 SQLite：

```text
/home/wdz/satellite_data/satellite_data.db
```

使用 `weather_station` 表：

```text
datetime
temperature
humidity
pressure
wind_speed
wind_direction
rainfall
rainfall_cumulative
```

### Stage1 pass 结果

默认读取：

```text
/home/wdz/BT/Stage1/model/data/pass_dataset.index.csv
```

如果该文件不存在，Stage1.5 仍会生成表，但 `stage1_*` 特征全部为 0。

## 2. 时间网格和目标

默认粒度是 10 分钟。

每一行 `date` 表示时间桶结束时刻。例如：

```text
2026-05-24 11:40:00 -> (11:30, 11:40]
```

Stage2 预测目标：

```text
rain_10min_mm = rainfall_cumulative(t) - rainfall_cumulative(t - 10min)
```

自定义 `FREQ=1h` 时，目标可理解为：

```text
rain_1h_mm = rainfall_cumulative(t) - rainfall_cumulative(t - 1h)
```

## 3. Stage1 特征落桶

Stage1 pass 是不规则事件，Stage1.5 按 `pass_end` 归入时间桶：

```text
bucket = ceil(pass_end, freq)
```

这样符合在线可用性：只有 pass 结束后，完整 pass 的 Stage1 结果才可用。

桶内聚合字段：

| 字段 | 含义 |
| --- | --- |
| `stage1_rain_sum` | 桶内 pass 雨量求和 |
| `stage1_rain_mean` | 桶内 pass 雨量均值 |
| `stage1_rain_max` | 桶内 pass 雨量最大值 |
| `stage1_rain_rate_mean` | 桶内 pass 的瞬时雨强均值统计 |
| `stage1_rain_rate_max` | 桶内 pass 的瞬时雨强最大值统计 |
| `stage1_pass_count` | 桶内 pass 数 |
| `stage1_has_pass` | 桶内是否有 pass |

重要：`pass_dataset.index.csv` 中的 `pass_rainfall_mm` 是真实标签，不是模型预测。它适合格式验证或上限实验。正式给 Stage2 使用时，应改用 Stage1 模型推理结果列，例如：

```text
pred_pass_rainfall_mm
```

并通过参数指定：

```bash
--stage1-rain-col pred_pass_rainfall_mm
```

## 4. 输出

默认输出：

```text
Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
```

同时生成：

```text
stage1_5_weather_10min.summary.json
```

输出表满足 Stage2 `Dataset_Custom` 约定：

```text
date, feature_1, ..., feature_n, target
```

默认字段：

| 字段 | 含义 |
| --- | --- |
| `temperature` / `humidity` / `pressure` | 时间桶内气象站均值 |
| `wind_speed` | 时间桶内平均风速 |
| `wind_direction` | 按风速加权的圆周平均风向 |
| `wind_dir_sin` / `wind_dir_cos` | 风向周期编码 |
| `wind_east` / `wind_north` | 风向风速分量 |
| `rain_rate_mean` / `rain_rate_max` | 瞬时雨强读数的桶内统计 |
| `weather_rows` | 桶内气象站记录数 |
| `stage1_*` | Stage1 pass 级结果聚合特征 |
| `rain_10min_mm` | Stage2 预测目标 |

默认表除 `date` 外有 20 个数值列。

## 5. 运行

先确保 Stage1 已生成 pass index：

```bash
cd /home/wdz/BT/Stage1/model
python3 - <<'PY'
from data.preprocessing import build_pass_dataset
build_pass_dataset(
    "/home/wdz/satellite_data/satellite_data.db",
    "/home/wdz/BT/Stage1/model/data/pass_dataset.npz",
)
PY
```

生成 10 分钟表：

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
  --pass-index /home/wdz/BT/Stage1/model/data/pass_dataset.index.csv \
  --freq 10min \
  --output /home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
```

## 6. Stage2 接口

Stage2 单变量降雨预测基线：

```bash
cd /home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting
python main.py \
  --root_path ./datasets/weather/ \
  --data_path stage1_5_weather_10min.csv \
  --target rain_10min_mm \
  --features S \
  --freq 10min \
  --model_id stage1_5_rain_10min \
  --data custom
```

当前 Stage2 开源实现中：

- `features=S`：只用目标列预测目标列，适合先跑纯降雨基线。
- `features=M`：会把所有数值列都作为输入和输出一起预测。

如果目标是“多变量气象 + Stage1 反演特征输入，单变量未来降雨输出”，后续需要小改 Stage2 的 data loader 或训练切片逻辑。

## 7. 验证

```bash
python3 -m compileall -q /home/wdz/BT/Stage1.5
cd /home/wdz/BT/Stage1.5
python3 build_stage2_weather_table.py --output /tmp/stage1_5_weather_10min_check.csv
```

当前 DB 下，10 分钟表验证结果约为：

```text
rows: 1019
target: rain_10min_mm
positive target rows: 61
target sum: 42.2 mm
```
