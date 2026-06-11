# LEO 星地链路降雨反演与预测

本项目分三段组织：

```text
Stage1   星地链路 -> pass 级降雨反演
Stage1.5 pass 级反演结果 -> 规则时序表
Stage2   规则时序表 -> 未来降雨预测
```

Stage2 使用开源 GPT4TS/时序预测代码，本仓库不重写其方法文档；这里只说明它在项目流水线中的位置。第三方复现代码和大模型权重不作为本仓库自有代码上传，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 总体数据流

```text
/home/wdz/satellite_data/satellite_data.db
  ├─ phy_data
  ├─ position_data
  ├─ weather_data
  └─ weather_station
        │
        ▼
Stage1
  输入：卫星链路 + 卫星/终端位置 + 地面气象
  输出：每次卫星过境的 pass_rainfall_mm
        │
        ▼
Stage1.5
  输入：气象站累计雨量 + Stage1 pass 级结果
  输出：Stage2/GPT4TS/Long-term_Forecasting/datasets/weather/stage1_5_weather_10min.csv
        │
        ▼
Stage2
  输入：规则时间网格上的气象和 Stage1 聚合特征
  输出：未来 rain_10min_mm / rain_1h_mm 等固定窗口降雨量
```

## 阶段分工

| 阶段 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| Stage1 | 反演当前卫星过境期间的降雨 | `phy_data`、`position_data`、`weather_data`、`weather_station` | `pass_dataset.npz`、`pass_dataset.index.csv`、checkpoint |
| Stage1.5 | 把不规则 pass 结果落到规则时间桶 | `weather_station`、Stage1 pass index 或预测结果 | `stage1_5_weather_10min.csv` |
| Stage2 | 时序预测 | Stage1.5 生成的 CSV | 未来固定窗口降雨预测 |

## 推荐运行顺序

1. 构建并训练 Stage1：

```bash
cd /home/wdz/BT/Stage1/model
REBUILD_CACHE=1 bash scripts/train_default.sh
```

2. 生成 Stage1.5 规则时序表：

```bash
cd /home/wdz/BT/Stage1.5
bash scripts/build_default.sh
```

3. 用 Stage2 训练预测模型：

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

Stage2 当前开源实现中，`features=S` 是单变量目标预测；`features=M` 会预测所有数值列。若要严格实现“多变量输入、单变量降雨输出”，需要后续小改 Stage2 loader/training 口径。

## 关键口径

- Stage1 主标签：`pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)`
- Stage1.5 目标：`rain_10min_mm = rainfall_cumulative(t) - rainfall_cumulative(t - 10min)`
- `rainfall` 瞬时读数只作为雨强统计，不直接累加为目标。
- Stage1 pass 结果按 `pass_end` 归入 Stage1.5 时间桶，避免线上推理时偷看未来。

## 文档入口

- [Stage1 README](Stage1/README.md)：pass 级反演模型、数据构建、训练配置
- [Stage1.5 README](Stage1.5/README.md)：规则时间表生成、字段口径、Stage2 接口
- `Stage2/GPT4TS/README.md`：开源方法原始说明
