# BT：星地链路降雨反演与预测

[English](README.md) | [中文](README_CN.md)

本仓库包含LEO卫星链路降雨反演、视觉天气分类、链路可靠性分析和时序预测实验。当前可直接运行和交付的Stage1主流程是**一分钟降雨反演**：使用雨量计锚点前一分钟内的卫星链路、星地几何、地面温湿度气压和天空图像天气概率，输出该分钟累计降雨量及降雨概率。

旧的卫星过境级Stage1反演已经移除。`Stage1.5`、`Stage2`和MoE中的其他代码是独立实验模块，不是当前分钟反演服务的必经步骤。

## 当前交付入口

```bash
cd /home/wdz/BT/Stage1/minute_rain_retrieval
python check_delivery.py --db-path /home/wdz/satellite_data/satellite_data.db
python -m pytest -q
```

启动8041在线服务：

```bash
cd /home/wdz/BT/MoE/lora-moe
PYTHON=/path/to/python bash scripts/serve_three_terminal_minute_rain_demo.sh
```

完整安装、输入、输出和运行说明见 [Stage1中文README](Stage1/README_CN.md)。

## 目录

| 路径 | 作用 |
|---|---|
| `Stage1/minute_rain_retrieval/` | 当前分钟降雨数据构建、Transformer训练、归档、测试和在线服务 |
| `Stage1/vision_weather/` | 天空图像天气分类和在线标签生成 |
| `Stage1/rainfall_dashboard/` | FastAPI/ECharts页面和链路可靠性展示 |
| `Stage1/satellite_identity/` | 终端星历ID和物理卫星身份分析 |
| `Stage1/link_reliability_analysis/` | 原始链路质量和长期趋势分析 |
| `Stage1.5/` | 历史桥接实验，不属于当前分钟反演主流程 |
| `Stage2/` | 时序预测实验和第三方GPT4TS复现 |
| `MoE/lora-moe/` | 视觉LoRA原型、共享历史库代码和分钟服务启动入口 |
| `THIRD_PARTY.md` | 第三方来源、许可证和引用说明 |

## 数据与权重

Git仓库包含一分钟反演的固定NPZ训练集、验证集、测试集及图像分类标签，克隆后可以直接训练和评估。原始SQLite、相机照片、分钟模型权重和历史结果库不提交Git；启动完整8041在线服务仍需按Stage1 README准备这些本地制品。

## 交付文档

- [Stage1运行说明](Stage1/README_CN.md)
- [分钟项目详细README](Stage1/minute_rain_retrieval/README_CN.md)
- [设计说明](Stage1/minute_rain_retrieval/docs/DESIGN_CN.md)
- [测试说明](Stage1/minute_rain_retrieval/docs/TESTING_CN.md)
- [风险与限制](Stage1/minute_rain_retrieval/docs/RISKS_CN.md)
- [3～5分钟Demo](Stage1/minute_rain_retrieval/docs/DEMO_CN.md)
