# Stage2：规则时序降雨预测

[English](README.md) | [中文](README_CN.md)

Stage2 在规则时间网格上预测未来降雨。预测后端使用第三方 GPT4TS / One Fits All 实现；该代码在本地作为依赖使用，暂不纳入本仓库。

上游项目：

```text
https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
```

## 输入

Stage2 读取 Stage1.5 生成的 CSV。表格符合 GPT4TS `Dataset_Custom` 格式：

```text
date, feature_1, ..., feature_n, target
```

常见目标列：

| 目标 | 含义 |
| --- | --- |
| `rain_10min_mm` | 过去 10 分钟累计降雨 |
| `rain_1h_mm` | 过去 1 小时累计降雨 |

## 在项目中的作用

Stage1 从卫星过境片段中反演降雨。Stage1.5 将 pass 级结果和气象站数据聚合成规则时间桶。Stage2 基于该结构化表学习未来降雨的时间演化。

第三方 GPT4TS 代码保持独立，以保留上游归属并控制仓库体积。依赖说明见 `THIRD_PARTY.md`。

## 示例

在本地 GPT4TS 目录中运行：

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

说明：

- `features=S` 是只使用目标列的单变量预测基线。
- `features=M` 在原始 GPT4TS 实现中会预测所有数值列。
- 如果需要“多变量输入、单变量降雨输出”，可能需要小改 data loader 或训练循环。
