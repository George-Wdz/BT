# GPT2 Rain Retrieval Baseline

This directory contains a controlled GPT2/GPT4TS-style baseline for Stage1
pass-level rainfall retrieval.

The experiment reuses the existing Stage1 NPZ pipeline:

- same pass segmentation and labels
- same train/val/test split logic
- same dry baseline / `dry_delta` construction
- same feature-group configuration
- same loss terms and metrics

The only intended change is the model backbone. Continuous pass patches are
projected into GPT2 hidden space and passed to a local GPT2 model through
`inputs_embeds`. By default GPT2 is frozen, and only the numeric adapters,
satellite embedding, summary token, and regression heads are trained.

## Run

```bash
cd /home/wdz/BT/Stage1/gpt2_rain_retrieval
bash scripts/run_gpt2_rain_workflow.sh --cuda-visible-devices 0
```

DDP multi-GPU training:

```bash
bash scripts/run_gpt2_rain_workflow.sh --ddp --cuda-visible-devices 0,1,2,3
```

The script automatically sets `torchrun --nproc_per_node` from the visible GPU
list, so `--cuda-visible-devices 0,1,2,3` means four training processes.

Dry-run without starting training:

```bash
bash scripts/run_gpt2_rain_workflow.sh --dry-run --ddp --cuda-visible-devices 0,1
```

Useful overrides:

```bash
bash scripts/run_gpt2_rain_workflow.sh \
  --set data.val_strategy=time \
  --set model.gpt2_layers=3 \
  --set model.freeze_gpt2=ln_wpe
```

`model.freeze_gpt2` options:

- `all`: freeze all GPT2 weights
- `ln_wpe`: train GPT2 LayerNorm and position embeddings only
- `none`: train the GPT2 backbone except the unused text token embedding table

Recommended order:

1. `all`: lowest-cost controlled baseline. GPT2 is only used as a frozen
   sequence backbone.
2. `ln_wpe`: small adaptation of GPT2 internals. This is a middle ground if
   the frozen baseline is close to useful.
3. `none`: GPT2 backbone fine-tuning. Since numeric soft tokens are passed
   through `inputs_embeds`, the GPT2 text token embedding table is not used
   and remains frozen. This is the most expensive and easiest to overfit on
   the current small rainfall dataset.

The default local GPT2 path is:

```text
/home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/gpt2
```

## Optimizer And LR

Training uses `AdamW`.

Default optimizer settings:

- `--lr 0.0001`: initial learning rate
- `--weight-decay 0.00001`: AdamW weight decay
- `--use-cosine 1`: enable `CosineAnnealingLR`
- `--tmax 20`: cosine scheduler `T_max`

With the default cosine scheduler, learning rate is stepped once per epoch and
decays toward `eta_min=1e-8`. Set `--use-cosine 0` to fall back to the Stage1
step-style scheduler controlled by `training.lradj` and `training.decay_fac` in
`configs/default.yaml`.

## Multi-GPU

The workflow uses PyTorch DDP through `torchrun` when `--ddp` is passed.

`training.batch_size` is interpreted as the per-GPU batch size. With two GPUs
and `training.batch_size=32`, the effective global batch size is about 64.

When DDP is enabled, the train set uses `DistributedSampler`. The rainy sample
weight is still applied through the loss function, but the single-process
`WeightedRandomSampler` is not used.

## Trainable Parameters

With the default `model.freeze_gpt2=all`, GPT2 itself is frozen. The trained
modules are:

- numeric patch projector
- pass summary projector
- satellite embedding and projection
- rainfall regression head
- rain/no-rain classification head
- auxiliary target head

This makes the baseline a low-cost test of whether a frozen GPT2 sequence
backbone helps after mapping numeric pass patches into GPT2 hidden space.

## Outputs

Each run writes:

- checkpoints: `Stage1/gpt2_rain_retrieval/checkpoints/`
- metrics and predictions: `Stage1/gpt2_rain_retrieval/runs/`
- logs: `Stage1/gpt2_rain_retrieval/logs/`
