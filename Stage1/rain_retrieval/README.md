# Stage1: Pass-Level Rainfall Retrieval

[English](README.md) | [中文](README_CN.md)

Stage1 retrieves rainfall during each satellite pass from satellite-link telemetry, satellite--ground propagation geometry, ground weather, and optional camera-derived weather probabilities. It estimates current pass-level rainfall; future forecasting is handled by Stage2.

## Scope

| Item | Description |
| --- | --- |
| Unit sample | One satellite pass, segmented by satellite ID and temporal continuity. |
| Main target | `pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)` |
| Optional auxiliary targets | `rain_rate_mean`, `rain_rate_max`, `rainy_ratio` |
| Main model | Pass-based Patch Encoder-Decoder Transformer |
| Recommended workflow | `Stage1/rain_retrieval/model/scripts/run_rain_retrieval_workflow.sh` |

## Input Features

The current recommended configuration uses:

| Group | Features |
| --- | --- |
| Link | `phyRssi`, `rssi`, `snr`, `lastCniValue` |
| Propagation geometry (geo4) | `slant_range_km`, `elevation_deg`, `azimuth_sin`, `azimuth_cos` |
| Ground weather | `temperature`, `humidity`, `pressure` |
| Image weather | `prob_sunny`, `prob_cloudy`, `prob_rain`, `image_available` |
| Dry baseline delta | same link features minus the train-set dry baseline |

The resulting default input dimension is:

```text
4 + 4 + 3 + 4 + 4 = 19
```

## Data Alignment

Pass construction and alignment are implemented in `Stage1/rain_retrieval/model/data/preprocessing.py`.

| Step | Policy |
| --- | --- |
| Pass segmentation | Same satellite, adjacent link samples separated by less than 60 seconds. |
| Minimum pass length | 10 valid link samples. |
| Position alignment | Nearest timestamp within 5 seconds. |
| Ground weather alignment | Nearest timestamp within 60 seconds. |
| Image weather alignment | Nearest image-label timestamp around pass center; the recommended workflow uses a 6-minute tolerance. |
| Rainfall label | Daily cumulative rainfall interpolated at pass start/end, then differenced. |

Instantaneous rainfall is kept for diagnostics and auxiliary targets; it is not used as the primary accumulated-rainfall target.

## Model

The model is defined in `Stage1/rain_retrieval/model/patch_encoder_decoder.py`.

Core components:

- overlapping patch embedding for irregular pass sequences;
- satellite embedding with an unknown-satellite slot;
- optional summary token from pass-level statistics;
- Transformer encoder and learnable target-query decoder;
- nonnegative rainfall regression head;
- rain/no-rain classification head;
- optional auxiliary regression heads.

Three fusion modes are available:

| Mode | Config | Description |
| --- | --- | --- |
| channel-mixing | `model.fusion_mode=cm` | Concatenate all features before patch embedding. |
| channel-wise | `model.fusion_mode=cw` | Encode groups separately, then apply temporal and modality attention. This is the default. |
| group-attention | `model.fusion_mode=ga` | Apply physical-group attention within each patch before temporal encoding. |

The current model is a single-path CW model. Rainfall regression, rain/no-rain classification, and auxiliary targets share the same multimodal encoder and target-query decoder; there is no separate CM regression branch. The default model uses lightweight modality-specific encoders, modality-quality gating, contextual conditioning, and one target-specific query/head for each auxiliary output.

Time-cycle and pass-length conditions are derived while loading the NPZ. Because geo4 is part of the default model input, the NPZ must contain the derived geometry columns; raw6-only legacy datasets must be rebuilt. Adaptive task weighting remains available for experiments but is disabled by default. The default dry baseline uses geometry-weighted top-k training-dry references blended with the same-satellite dry mean.

## Training

Run the full Stage1 workflow:

```bash
cd /home/wdz/BT/Stage1/rain_retrieval/model
bash scripts/run_rain_retrieval_workflow.sh
```

Common settings, including GPU selection, paths, feature groups, epoch count,
batch size, patience, and learning rate, are kept directly in the script's
`python_cmd` array. For temporary overrides, append command-line arguments:

```bash
bash scripts/run_rain_retrieval_workflow.sh --lr 0.0002 --epochs 50
```

The workflow:

1. exports camera weather labels;
2. builds a timestamped NPZ pass dataset;
3. trains the recommended rainfall-retrieval model;
4. evaluates train/validation/test splits;
5. writes a run manifest and metrics.

Run feature ablations on an existing pass dataset:

```bash
cd /home/wdz/BT/Stage1/rain_retrieval/model
bash scripts/run_feature_ablation.sh \
  --pass-dataset-path data/datasets/pass_dataset_rain_retrieval_20260617_1806.npz
```

Default feature-ablation variants:

```text
full_a        = link + geo4 + ground_weather + image_weather + dry_delta
no_image      = link + geo4 + ground_weather + dry_delta
no_ground     = link + geo4 + image_weather + dry_delta
no_dry_delta  = link + geo4 + ground_weather + image_weather
```

For a single low-level training run, call `python main.py --set ...` directly.

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

`evaluate_checkpoint_splits.py` reports overall/rainy/dry regression metrics,
rain-classification precision, recall, F1, PR-AUC, ROC-AUC and false-alarm rate,
plus rainfall-severity, satellite, image-availability and optional elevation slices.

## Outputs

Generated files are kept outside this repository:

| Output | Description |
| --- | --- |
| `Stage1/rain_retrieval/model/data/**/*.npz` | Pass datasets. |
| `Stage1/rain_retrieval/model/checkpoints*/` | Model checkpoints. |
| `Stage1/rain_retrieval/model/logs/` | Training logs. |
| `Stage1/rain_retrieval/analysis/**/runs/` | Evaluation outputs. |

Datasets and model artifacts can be released separately, for example in a private Hugging Face repository or institutional object storage.
