# LoRA-MoE

This directory retains the visual-weather LoRA expert, generic Qwen services,
and the minute-rainfall dashboard entry point. The legacy pass-level Stage1
rainfall expert has been removed; minute retrieval is trained and served from
`Stage1/minute_rain_retrieval/`.

```bash
bash scripts/train_vision_weather_lora.sh
bash scripts/infer_vision_weather.sh
bash scripts/serve_vision_weather_fastapi.sh
bash scripts/serve_three_terminal_minute_rain_demo.sh
```

See `scripts/README_CN.md` for the maintained entry points.
