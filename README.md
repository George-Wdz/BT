# BT: Satellite-Link Rainfall Retrieval and Forecasting

[English](README.md) | [中文](README_CN.md)

This repository contains LEO satellite-link rainfall retrieval, visual weather classification, link reliability analysis, and forecasting experiments. The active Stage1 deliverable is **minute-level rainfall retrieval**: observations in the minute preceding a rain-gauge anchor are mapped to one accumulated rainfall estimate and one rain probability.

The legacy pass-level Stage1 retrieval has been removed. `Stage1.5`, `Stage2`, and other MoE code are separate experiments and are not required by the current minute-level service.

## Quick Start

```bash
git clone https://github.com/George-Wdz/BT.git
cd BT
python -m pip install -r Stage1/requirements.txt

cd Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python -m pytest -q
```

See the [Chinese Stage1 guide](Stage1/README_CN.md) for training commands, input schema, outputs, architecture, tests, limitations, and the demonstration sequence.

Git includes a fixed NPZ dataset, exported train/validation/test splits, and image-classification labels, so a clone can train and evaluate immediately. Raw SQLite databases, camera images, deployed weights, and history databases are not stored in Git; these local artifacts are required only for the optional online service.
