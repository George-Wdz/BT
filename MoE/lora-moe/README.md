# LoRA-MoE: Visual Weather Adapter

[English](README.md) | [中文](README_CN.md)

This directory contains project-specific code for a parameter-efficient visual weather adapter. It is an early LoRA-MoE prototype: the current implementation focuses on the `vision_weather` expert and uses image features as soft tokens for a Qwen language model.

## Scope

| Item | Description |
| --- | --- |
| Current task | Weather recognition from camera images. |
| Classes | `sunny`, `cloudy`, `rain` |
| Vision encoder | Frozen `WeatherClassifier` encoder. |
| Projector | Trainable MLP that maps visual features to soft tokens. |
| Language model | Qwen2.5-14B-Instruct, frozen except LoRA parameters. |
| LoRA target modules | `q_proj`, `v_proj` |

This module is separate from the direct image-classifier path used by Stage1 data construction. It is intended to validate model-input-level multimodal adaptation, not to replace the full rainfall retrieval pipeline.

## Architecture

```text
image
  -> frozen WeatherClassifier encoder
  -> visual feature vector
  -> trainable projector
  -> visual soft tokens
  -> Qwen2.5-14B-Instruct + vision_weather LoRA
  -> Chinese weather response
```

Only the projector and LoRA parameters are trained. The vision encoder and base language model remain frozen.

## Directory Layout

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

Generated adapters, projector weights, logs, and checkpoints are kept outside this repository.

## Training

Smoke test:

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
MAX_TRAIN_SAMPLES=64 \
MAX_STEPS=20 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_smoke \
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

Full first-pass training:

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

Default key settings:

| Parameter | Default |
| --- | --- |
| `BATCH_SIZE` | `1` |
| `GRAD_ACCUM_STEPS` | `8` for smoke test, `16` for larger runs |
| `NUM_VISUAL_TOKENS` | `8` |
| `PROJECTOR_HIDDEN_DIM` | `1024` |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` |
| `LORA_DROPOUT` | `0.05` |
| `LEARNING_RATE` | `2e-4` |

## Inference

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
OUTPUT_DIR=/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1 \
conda run --no-capture-output -n smoe bash scripts/infer_vision_weather.sh
```

Optional inputs:

```bash
IMAGE=/path/to/image.jpg bash scripts/infer_vision_weather.sh
IMAGE_DIR=/path/to/images bash scripts/infer_vision_weather.sh
SAVE_JSONL=/tmp/vision_weather_predictions.jsonl bash scripts/infer_vision_weather.sh
```

## Service

```bash
cd /home/wdz/BT/MoE/lora-moe
CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n smoe bash scripts/serve_vision_weather_fastapi.sh
```

## Artifact Management

The following artifacts are not included in this repository:

- Qwen base weights;
- WeatherClassifier weights;
- LoRA adapter outputs;
- `projector.pt`;
- logs and intermediate checkpoints;
- image datasets.

Shareable model artifacts can be released separately, for example through a private Hugging Face Model repository or institutional object storage.
