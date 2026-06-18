# LoRA-MoE: Multi-Encoder Multi-LoRA Prototype

[English](README.md) | [中文](README_CN.md)

This directory contains project-specific code for parameter-efficient multimodal adaptation. It is an early LoRA-MoE prototype: the current implementation includes the `vision_weather` expert and the `stage1_rain` expert. Both use frozen project encoders to produce soft tokens for a Qwen language model.

## Scope

| Item | Description |
| --- | --- |
| Current tasks | Camera-image weather recognition; Stage1 satellite-link rainfall retrieval |
| Vision classes | `sunny`, `cloudy`, `rain` |
| Vision encoder | Frozen `Stage1/vision_weather/WeatherClassifier` encoder, corresponding to A1B1 |
| Stage1 encoder | Frozen `PatchEncoderDecoder` rainfall retrieval model, corresponding to A2B2 |
| Projector | Trainable MLP that maps frozen encoder features to soft tokens |
| Language model | Qwen2.5-14B-Instruct, frozen except LoRA parameters |
| LoRA target modules | `q_proj`, `v_proj` |

This module is separate from the direct image-classifier path used by Stage1 data construction. It validates model-input-level multimodal adaptation and does not replace the full rainfall retrieval pipeline.

## Architecture

Vision-weather expert:

```text
image
  -> frozen WeatherClassifier encoder
  -> visual feature vector
  -> trainable projector
  -> visual soft tokens
  -> Qwen2.5-14B-Instruct + vision_weather LoRA
  -> Chinese weather response
```

Stage1 rainfall retrieval expert:

```text
satellite pass features
  -> frozen Stage1 PatchEncoderDecoder
  -> pass-level retrieval feature vector
  -> trainable projector
  -> Stage1 soft tokens
  -> Qwen2.5-14B-Instruct + stage1_rain LoRA
  -> Chinese rainfall response
```

Only the projector and LoRA parameters are trained. The Stage1 encoders and the base language model remain frozen.

The first Stage1 training target is intentionally narrow: Qwen learns to produce short Chinese rainfall answers such as "the rainfall for this satellite pass is about X mm" from Stage1 tokens. The answer target uses the frozen Stage1 model prediction instead of the rain-gauge ground-truth label, so training and online inference stay consistent. The rain-gauge resolution is handled with `--no-rain-threshold 0.06`; predictions below this threshold are expressed as no rain / 0 mm. This does not mean the Stage1 model itself is already accurate enough. After the Stage1 checkpoint improves, retrain the corresponding projector and A2B2 LoRA.

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

Generated adapters, projector weights, logs, and checkpoints are not committed to this repository. The lightweight default `Stage1/vision_weather` classifier weight is committed because the vision expert depends on the matching encoder checkpoint for local reproducibility.

## Training

### Vision Weather A1B1

Smoke test:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh
```

First full pass:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_vision_weather_lora.sh \
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_v1 \
  --max-train-samples 0 \
  --max-steps 0 \
  --epochs 1 \
  --grad-accum-steps 16
```

### Stage1 Rainfall A2B2

Smoke test:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_stage1_rain_lora.sh
```

Demo training:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/train_stage1_rain_lora.sh \
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1 \
  --max-train-samples 0 \
  --max-steps 100 \
  --epochs 1 \
  --grad-accum-steps 16
```

Default Stage1 checkpoint:

```text
/home/wdz/BT/Stage1/rain_retrieval/model/checkpoints/pass_dataset_rain_retrieval_20260612_1116/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0
```

Default Stage1 pass dataset:

```text
/home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260612_1116.npz
```

After retraining a better Stage1 model, update `--stage1-checkpoint-dir`,
`--pass-dataset-path`, and `--output-dir` in
`scripts/train_stage1_rain_lora.sh`, or pass those command-line arguments after
the script name.

Default key settings:

| Parameter | Default |
| --- | --- |
| `--batch-size` | `1` |
| `--grad-accum-steps` | `8` for smoke test, `16` for larger runs |
| `--num-visual-tokens` | `8` |
| `--num-stage1-tokens` | `8` |
| `--projector-hidden-dim` | `1024` |
| `--lora-r` / `--lora-alpha` | `8` / `16` |
| `--lora-dropout` | `0.05` |
| `--learning-rate` | `2e-4` |
| `--no-rain-threshold` | `0.06` |

## Inference

Vision-weather batch inference:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/infer_vision_weather.sh
```

Optional inputs:

```bash
bash scripts/infer_vision_weather.sh --image /path/to/image.jpg
bash scripts/infer_vision_weather.sh --image-dir /path/to/images
bash scripts/infer_vision_weather.sh --save-jsonl /tmp/vision_weather_predictions.jsonl
```

## Service

Services are currently deployed per expert. This keeps A1B1 and A2B2 easier to validate independently, including trained artifacts, online feature construction, and GPU memory usage. After the vision, Stage1, and Stage2 experts are stable, they should be merged into one multi-expert FastAPI entry point with rule-based or learned routing.

Vision-weather A1B1:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/serve_vision_weather_fastapi.sh
```

Stage1 rainfall A2B2:

```bash
cd /home/wdz/BT/MoE/lora-moe
conda run --no-capture-output -n smoe bash scripts/serve_stage1_rain_fastapi.sh \
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/stage1_rain_lora_a2b2_v1
```

Default ports:

```text
vision_weather: http://127.0.0.1:8010
stage1_rain:    http://127.0.0.1:8011
```

The Stage1 A2B2 service persistently loads:

- Qwen2.5-14B-Instruct;
- Stage1 A2B2 LoRA adapter;
- Stage1 projector;
- frozen Stage1 checkpoint;
- `cfg`, `scaler_X`, `scaler_y`, and `sat_mapper` from `meta.pt`;
- dry-baseline state frozen from the training split.

The background worker reads the recent window of `/home/wdz/satellite_data/satellite_data.db` every 30 seconds by default. It only builds the latest passes and does not rebuild the full dataset. If no latest satellite pass is available, `/generate` directly returns "无最新卫星过境".

Key online parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--poll-interval-s` | `30` | Background polling interval |
| `--stale-after-s` | `180` | Treat latest `phy_data` older than this as no latest pass |
| `--lookback-hours` | `4` | DB window read on each poll |
| `--max-passes` | `8` | Maximum recent passes kept per update |
| `--use-best` | enabled | Load `best/adapter` and `best/projector.pt` |
| `--db-path` | `/home/wdz/satellite_data/satellite_data.db` | Online satellite DB |
| `--no-rain-threshold` | `0.06` | Display Stage1 predictions below this threshold as no rain / 0 mm |

## Artifact Management

The following artifacts are not included in this repository:

- Qwen base weights;
- generated WeatherClassifier weights other than the current default `Stage1/vision_weather` checkpoint;
- Stage1 checkpoints and `meta.pt`;
- Stage1 NPZ datasets;
- online satellite DB;
- LoRA adapter outputs;
- `projector.pt`;
- logs and intermediate checkpoints;
- image datasets.

Shareable model artifacts can be released separately, for example through a private Hugging Face model repository or institutional object storage.
