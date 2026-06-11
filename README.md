# BT: Satellite-Link Rainfall Retrieval and Forecasting

[English](README.md) | [中文](README_CN.md)

This repository contains project-specific code and documentation for rainfall retrieval and forecasting from LEO satellite-link observations, ground weather data, camera-derived weather cues, and time-series forecasting models.

The repository is a lightweight code backup. Raw data, database snapshots, model weights, checkpoints, logs, and third-party reproduced repositories are intentionally excluded.

## Project Structure

| Path | Role |
| --- | --- |
| `Stage1/` | Pass-level rainfall retrieval from satellite-link observations. |
| `Stage1.5/` | Bridge from irregular pass-level retrievals to regular Stage2 weather tables. |
| `MoE/lora-moe/` | Project-specific LoRA/soft-token prototype for visual weather adaptation. |
| `docs/` | Methodology draft and rendered architecture figures. |
| `THIRD_PARTY.md` | Third-party dependency notes and upstream links. |

## Pipeline

```text
SQLite sensor database
  -> Stage1: satellite pass rainfall retrieval
  -> Stage1.5: pass outputs aggregated to regular time buckets
  -> Stage2: rainfall forecasting with GPT4TS-style time-series models
```

Main target definitions:

- Stage1: `pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)`
- Stage1.5/Stage2: fixed-window rainfall, e.g. `rain_10min_mm = rainfall_cumulative(t) - rainfall_cumulative(t - 10min)`

Instantaneous `rainfall` is used as diagnostic or auxiliary information, not as the primary accumulated-rainfall target.

## Third-Party Code

The following reproduced third-party repositories are used locally but are not vendored into this GitHub repository:

| Local role | Upstream GitHub |
| --- | --- |
| GPT4TS / One Fits All time-series forecasting backend | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All |
| LLaMA-MoE reference and serving experiments | https://github.com/pjlab-sys4nlp/llama-moe |

Keep the original licenses, citations, and installation instructions from each upstream project. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Data and Artifacts

Do not commit:

- SQLite databases and backups
- raw camera images
- generated CSV/NPZ datasets
- model weights and checkpoints
- logs and local caches
- downloaded third-party repositories

Shareable datasets or model artifacts should be stored separately, for example in a private Hugging Face Dataset/Model repository.

## Documentation

- [Stage1 README](Stage1/README.md)
- [Stage1.5 README](Stage1.5/README.md)
- [Methodology draft](docs/methodology.md)
