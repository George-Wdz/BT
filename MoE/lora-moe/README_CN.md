# LoRA-MoE

该目录保留视觉天气LoRA专家、通用Qwen服务和一分钟降雨反演前端入口。
旧Stage1过境级雨量专家已经移除；分钟反演模型由
`Stage1/minute_rain_retrieval/`独立训练和部署。

## 视觉天气LoRA

视觉分类编码器产生特征，经projector转换为soft tokens并与文本token embedding
拼接；Qwen主干冻结原始参数，仅训练指定LoRA层和projector。

```bash
bash scripts/train_vision_weather_lora.sh
bash scripts/infer_vision_weather.sh
bash scripts/serve_vision_weather_fastapi.sh
```

## 一分钟降雨服务

```bash
bash scripts/serve_three_terminal_minute_rain_demo.sh
```

该入口启动8041 FastAPI服务，加载三套分钟权重、在线视觉分类器、三终端协议
适配器及统一历史库。具体参数见 `scripts/serve_three_terminal_minute_rain_demo.sh`。

## 通用Qwen演示

`scripts/serve_dual_qwen_demo.sh` 仅用于展示两个普通Qwen服务通过Gateway交换
文本，不属于降雨反演链路。
