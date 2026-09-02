# BT: Satellite-Link Rainfall Retrieval and Forecasting

[English](README.md) | [中文](README_CN.md)

This repository contains LEO satellite-link rainfall retrieval, visual weather classification, link reliability analysis, and forecasting experiments. The active Stage1 deliverable is **minute-level rainfall retrieval**: observations in the minute preceding a rain-gauge anchor are mapped to one accumulated rainfall estimate and one rain probability.

The legacy pass-level Stage1 retrieval has been removed. `Stage1.5`, `Stage2`, and other MoE code are separate experiments and are not required by the current minute-level service.

## Quick Check

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python -m pytest -q
bash scripts/run_reproducible_smoke_test.sh
```

Start the dashboard service with:

```bash
cd /home/wdz/BT/MoE/lora-moe
PYTHON=/path/to/python bash scripts/serve_three_terminal_minute_rain_demo.sh
```

See the [Chinese Stage1 handoff guide](Stage1/README_CN.md) for the verified environment, required local artifacts, input schema, outputs, architecture, tests, limitations, and demo sequence.

Git includes a fixed NPZ dataset, train/validation/test audit splits, and image-classification labels, so a clone can train and evaluate immediately. Raw SQLite databases, camera images, deployed weights, and history databases are not stored in Git; these local artifacts are still required for online serving.
