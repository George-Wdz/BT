# 脚本入口

当前保留视觉天气LoRA、通用Qwen演示和Stage1一分钟反演服务。旧过境级
降雨LoRA训练、评估和服务入口已经移除。

| 脚本 | 用途 |
| --- | --- |
| `train_vision_weather_lora.sh` | 训练视觉天气projector与LoRA |
| `infer_vision_weather.sh` | 离线验证视觉天气LoRA |
| `serve_vision_weather_fastapi.sh` | 启动视觉天气专家服务 |
| `serve_three_terminal_minute_rain_demo.sh` | 启动8041三终端一分钟反演服务 |
| `serve_dual_qwen_demo.sh` | 与业务无关的双Qwen通信演示 |

```bash
cd /home/wdz/BT/MoE/lora-moe
bash scripts/serve_three_terminal_minute_rain_demo.sh
```
