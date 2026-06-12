# Stage1: Pass-Level Rainfall Retrieval

[English](README.md) | [中文](README_CN.md)

Stage1 retrieves rainfall during each satellite pass from satellite-link telemetry, satellite/terminal position, ground weather, and optional camera-derived weather probabilities. It estimates current pass-level rainfall; future forecasting is handled by Stage2.

## Scope

| Item | Description |
| --- | --- |
| Unit sample | One satellite pass, segmented by satellite ID and temporal continuity. |
| Main target | `pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)` |
| Optional auxiliary targets | `rain_rate_mean`, `rain_rate_max`, `rainy_ratio` |
| Main model | Pass-based Patch Encoder-Decoder Transformer |
| Recommended workflow | `Stage1/model/scripts/run_rain_retrieval_workflow.sh` |

## Input Features

The current recommended configuration uses:

| Group | Features |
| --- | --- |
| Link | `phyRssi`, `rssi`, `snr`, `lastCniValue` |
| Position | satellite longitude/latitude/altitude and terminal longitude/latitude/altitude |
| Ground weather | `temperature`, `humidity`, `pressure` |
| Image weather | `prob_sunny`, `prob_cloudy`, `prob_rain`, `image_available` |
| Dry baseline delta | same link features minus the train-set dry baseline |

The resulting default input dimension is:

```text
4 + 6 + 3 + 4 + 4 = 21
```

## Data Alignment

Pass construction and alignment are implemented in `Stage1/model/data/preprocessing.py`.

| Step | Policy |
| --- | --- |
| Pass segmentation | Same satellite, adjacent link samples separated by less than 60 seconds. |
| Minimum pass length | 10 valid link samples. |
| Position alignment | Nearest timestamp within 5 seconds. |
| Ground weather alignment | Nearest timestamp within 60 seconds. |
| Image weather alignment | Nearest image-label timestamp around pass center, default tolerance 10 minutes. |
| Rainfall label | Daily cumulative rainfall interpolated at pass start/end, then differenced. |

Instantaneous rainfall is kept for diagnostics and auxiliary targets; it is not used as the primary accumulated-rainfall target.

## Model

The model is defined in `Stage1/model/models/patch_encoder_decoder.py`.

Core components:

- overlapping patch embedding for irregular pass sequences;
- satellite embedding with an unknown-satellite slot;
- optional summary token from pass-level statistics;
- Transformer encoder and learnable target-query decoder;
- nonnegative rainfall regression head;
- rain/no-rain classification head;
- optional auxiliary regression heads.

Two encoder modes are available:

| Mode | Config | Description |
| --- | --- | --- |
| channel-mixing | `model.use_channel_attention=false` | Concatenate all features before patch embedding. |
| channel-wise | `model.use_channel_attention=true` | Embed feature groups separately, then apply temporal and channel attention. |

## Training

Run the full Stage1 workflow:

```bash
cd /home/wdz/BT/Stage1/model
ITERATIONS=1 EPOCHS=100 BATCH_SIZE=32 PATIENCE=15 \
bash scripts/run_rain_retrieval_workflow.sh
```

The workflow:

1. exports camera weather labels;
2. builds a timestamped NPZ pass dataset;
3. trains the recommended rainfall-retrieval model;
4. evaluates train/validation/test splits;
5. writes a run manifest and metrics.

Run only the recommended experiment:

```bash
cd /home/wdz/BT/Stage1/model
VAL_STRATEGY=stratified_all \
ITERATIONS=1 EPOCHS=100 BATCH_SIZE=32 PATIENCE=15 \
bash scripts/train_experiments.sh rain_retrieval
```

Useful ablations:

```bash
bash scripts/train_experiments.sh ablate_no_instant_aux
bash scripts/train_experiments.sh ablate_no_summary
bash scripts/train_experiments.sh ablate_no_drybase
```

## Splits

Default split ratio:

```text
train / validation / test = 0.7 / 0.2 / 0.1
```

Available validation strategies:

| Strategy | Use case |
| --- | --- |
| `stratified_all` | Rain/dry stratification across all splits. Useful when rainy passes are rare and diagnosis needs rainy test samples. |
| `stratified_before_test` | Last 10% of time remains test; train/validation are stratified before that point. Better for online-style validation. |

Scalers, satellite ID mapping, and dry-baseline references are fitted from the training split only.

## Outputs

Generated files are kept outside this repository:

| Output | Description |
| --- | --- |
| `Stage1/model/data/**/*.npz` | Pass datasets. |
| `Stage1/model/checkpoints*/` | Model checkpoints. |
| `Stage1/model/logs/` | Training logs. |
| `Stage1/analysis/**/runs/` | Evaluation outputs. |

Datasets and model artifacts can be released separately, for example in a private Hugging Face repository or institutional object storage.
