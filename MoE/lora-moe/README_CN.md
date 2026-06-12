# LoRA-MoE：视觉天气适配器

[English](README.md) | [中文](README_CN.md)

本目录保存项目自有的参数高效视觉天气适配代码。它是 LoRA-MoE 路线的早期原型：当前只实现 `vision_weather` 专家，用视觉特征生成 soft tokens，再输入 Qwen 语言模型。

## 范围

| 项 | 说明 |
| --- | --- |
| 当前任务 | 相机图像天气识别 |
| 类别 | `sunny`、`cloudy`、`rain` |
| 视觉编码器 | 冻结的 `WeatherClassifier` encoder |
| Projector | 可训练 MLP，将视觉特征映射为 soft tokens |
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

生成的 adapter、projector 权重、日志和 checkpoint 不进入 Git。

## 训练

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

关键默认参数：

| 参数 | 默认值 |
| --- | --- |
| `BATCH_SIZE` | `1` |
| `GRAD_ACCUM_STEPS` | smoke test 用 `8`，较大训练用 `16` |
| `NUM_VISUAL_TOKENS` | `8` |
| `PROJECTOR_HIDDEN_DIM` | `1024` |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` |
| `LORA_DROPOUT` | `0.05` |
| `LEARNING_RATE` | `2e-4` |

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

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n smoe bash scripts/serve_vision_weather_fastapi.sh
```

## 产物管理

不要提交：

- Qwen 基座权重；
- WeatherClassifier 权重；
- LoRA adapter 输出；
- `projector.pt`；
- 日志和中间 checkpoint；
- 图像数据集。

可共享模型产物应单独保存，例如私有 Hugging Face Model 仓库。

