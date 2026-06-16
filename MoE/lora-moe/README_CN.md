# LoRA-MoE：多编码器多 LoRA 原型

[English](README.md) | [中文](README_CN.md)

本目录保存项目自有的参数高效多模态适配代码。它是 LoRA-MoE 路线的早期原型：当前包含 `vision_weather` 专家和 `stage1_rain` 专家，分别用仓库内冻结编码器生成 soft tokens，再输入 Qwen 语言模型。

## 范围

| 项 | 说明 |
| --- | --- |
| 当前任务 | 相机图像天气识别；Stage1 卫星链路降雨反演 |
| 视觉类别 | `sunny`、`cloudy`、`rain` |
| 视觉编码器 | 冻结的 `Stage1/vision_weather/WeatherClassifier` encoder，对应 A1B1 |
| Stage1 编码器 | 冻结的 `PatchEncoderDecoder` 反演模型，对应 A2B2 |
| Projector | 可训练 MLP，将冻结编码器特征映射为 soft tokens |
| 语言模型 | Qwen2.5-14B-Instruct，除 LoRA 外冻结 |
| LoRA 注入层 | `q_proj`、`v_proj` |

该模块与 Stage1 数据构建中直接使用图像分类器导出概率的路径不同。它用于验证模型输入层的多模态适配，不直接替代完整降雨反演流程。

## 结构

```text
image
  -> frozen WeatherClassifier encoder
  -> visual feature vector
  -> trainable projector
  -> visual soft tokens
  -> Qwen2.5-14B-Instruct + vision_weather LoRA
  -> 中文天气回答
```

训练时只更新 projector 和 LoRA 参数。视觉编码器和基座语言模型保持冻结。

Stage1 反演专家结构：

```text
satellite pass features
  -> frozen Stage1 PatchEncoderDecoder
  -> pass-level retrieval feature vector
  -> trainable projector
  -> Stage1 soft tokens
  -> Qwen2.5-14B-Instruct + stage1_rain LoRA
  -> 中文降雨量回答
```

Stage1 首版训练目标很窄：让 Qwen 根据反演 token 输出“本次卫星过境降雨量约为 X 毫米”。回答目标使用冻结 Stage1 小模型自己的预测值，而不是雨量计真实标签，这样训练和在线推理保持一致。雨量计分辨率按 `NO_RAIN_THRESHOLD=0.06` 处理，预测值低于该阈值时按无雨/0mm 表达。这不代表 Stage1 小模型本身已经足够准确；后续 Stage1 权重更新后，应重新训练对应 projector 和 A2B2 LoRA。

## 目录

```text
MoE/lora-moe/
  configs/
  scripts/
  src/lora_moe/
    components.py
    datasets.py
    train/
    infer/
    serve/
```

生成的 adapter、projector 权重、日志和 checkpoint 暂不纳入本仓库。轻量 `Stage1/vision_weather` 默认分类权重已纳入仓库，因为视觉专家本地复现需要匹配的编码器 checkpoint。

## 训练

### 视觉天气 A1B1

Smoke test：

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
MAX_TRAIN_SAMPLES=64 \
MAX_STEPS=20 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_smoke \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

首版完整训练：

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_TRAIN_SAMPLES=0 \
MAX_STEPS=0 \
EPOCHS=1 \
GRAD_ACCUM_STEPS=16 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_v1 \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

### Stage1 反演 A2B2

Smoke test：

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
MAX_TRAIN_SAMPLES=64 \
MAX_STEPS=20 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_smoke \
conda run --no-capture-output -n smoe bash scripts/train_stage1_rain_lora.sh
```

演示版训练：

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_TRAIN_SAMPLES=0 \
MAX_STEPS=100 \
EPOCHS=1 \
GRAD_ACCUM_STEPS=16 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1 \
conda run --no-capture-output -n smoe bash scripts/train_stage1_rain_lora.sh
```

默认 Stage1 checkpoint：

```text
/home/wdz/BT/Stage1/rain_retrieval/model/checkpoints/pass_dataset_rain_retrieval_20260612_1116/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0
```

默认 Stage1 pass 数据集：

```text
/home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260612_1116.npz
```

如果之后 Stage1 小模型重训好了，请通过 `STAGE1_CHECKPOINT_DIR=/path/to/new/checkpoint_dir` 和 `PASS_DATASET_PATH=/path/to/new/pass_dataset.npz` 指向新产物，并用新的 `OUTPUT_DIR` 重新训练 A2B2。

关键默认参数：

| 参数 | 默认值 |
| --- | --- |
| `BATCH_SIZE` | `1` |
| `GRAD_ACCUM_STEPS` | smoke test 用 `8`，较大训练用 `16` |
| `NUM_VISUAL_TOKENS` | `8` |
| `NUM_STAGE1_TOKENS` | `8` |
| `PROJECTOR_HIDDEN_DIM` | `1024` |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` |
| `LORA_DROPOUT` | `0.05` |
| `LEARNING_RATE` | `2e-4` |
| `NO_RAIN_THRESHOLD` | `0.06` |

## 推理

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1 \
conda run --no-capture-output -n smoe bash scripts/infer_vision_weather.sh
```

可选输入：

```bash
IMAGE=/path/to/image.jpg bash scripts/infer_vision_weather.sh
IMAGE_DIR=/path/to/images bash scripts/infer_vision_weather.sh
SAVE_JSONL=/tmp/vision_weather_predictions.jsonl bash scripts/infer_vision_weather.sh
```

## 服务

当前服务按专家拆分部署，目的是先分别验证 A1B1 和 A2B2 的训练产物、在线特征构造和 GPU 显存占用。后续视觉、Stage1、Stage2 都稳定后，再合并成统一多专家 FastAPI，由一个入口做规则路由或学习路由。

视觉天气 A1B1：

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n smoe bash scripts/serve_vision_weather_fastapi.sh
```

Stage1 反演 A2B2：

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1 \
conda run --no-capture-output -n smoe bash scripts/serve_stage1_rain_fastapi.sh
```

默认端口：

```text
vision_weather: http://127.0.0.1:8010
stage1_rain:    http://127.0.0.1:8011
```

Stage1 A2B2 服务启动时会持久化加载：

- Qwen2.5-14B-Instruct；
- Stage1 A2B2 LoRA adapter；
- Stage1 projector；
- 冻结 Stage1 checkpoint；
- `meta.pt` 中的 `cfg`、`scaler_X`、`scaler_y`、`sat_mapper`；
- 从训练 split 固化出来的 dry baseline 状态。

后台线程默认每 30 秒读取 `/home/wdz/satellite_data/satellite_data.db` 的最近窗口数据，只构造最新 pass，不全库重建。没有最新卫星过境时，`/generate` 会直接返回“无最新卫星过境”。

关键在线参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `POLL_INTERVAL_S` | `30` | 后台轮询间隔 |
| `STALE_AFTER_S` | `180` | 最新 `phy_data` 超过该秒数视为无最新卫星过境 |
| `LOOKBACK_HOURS` | `4` | 每次只读取最近几小时数据库 |
| `MAX_PASSES` | `8` | 每次最多保留最近几个 pass |
| `USE_BEST` | `1` | 加载 `best/adapter` 和 `best/projector.pt` |
| `SENSOR_DB_PATH` | `/home/wdz/satellite_data/satellite_data.db` | 实时卫星数据库 |
| `NO_RAIN_THRESHOLD` | `0.06` | 低于该阈值的 Stage1 预测展示为无雨/0mm |

## 产物管理

以下产物暂不纳入本仓库：

- Qwen 基座权重；
- 除当前默认 `Stage1/vision_weather` checkpoint 外的新 WeatherClassifier 权重；
- Stage1 checkpoint 和 `meta.pt`；
- Stage1 NPZ 数据集；
- 在线卫星数据库；
- LoRA adapter 输出；
- `projector.pt`；
- 日志和中间 checkpoint；
- 图像数据集。

可共享模型产物后续可单独发布，例如存放在私有 Hugging Face Model 仓库或单位对象存储中。
