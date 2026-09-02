# Stage1 Minute-Level Rainfall Retrieval

Stage1 currently predicts one accumulated rainfall value for each rain-gauge-anchored preceding-minute window. The legacy pass-level retrieval implementation has been removed.

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
python check_delivery.py --training-only
python -m pytest -q
```

The complete handoff guide is available in [README_CN.md](README_CN.md), with detailed runtime instructions in [minute_rain_retrieval/README_CN.md](minute_rain_retrieval/README_CN.md).
