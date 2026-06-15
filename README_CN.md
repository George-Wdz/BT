# BT：星地链路降雨反演与预测

[English](README.md) | [中文](README_CN.md)

本仓库保存星地链路降雨反演与预测相关的项目代码、配置和文档。项目使用 LEO 卫星链路观测、地面气象数据、相机天气信息和时序预测模型，构建从“过境级降雨反演”到“规则时间序列预测”的处理流程。

本仓库是轻量代码备份。原始数据、数据库快照、大模型权重、checkpoint、日志、本地缓存和第三方复现仓库暂不纳入本 GitHub 仓库。当前轻量 Stage1 视觉天气分类器默认权重作为小型复现产物已纳入仓库。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `Stage1/` | Stage1 组件：`vision_weather/` 为相机图像天气分类，`rain_retrieval/` 为 pass 级降雨反演。 |
| `Stage1.5/` | 将不规则 pass 级结果聚合到规则时间表。 |
| `MoE/lora-moe/` | 项目自有的视觉天气 LoRA / soft-token 原型。 |
| `docs/` | Methodology 草稿和架构图。 |
| `THIRD_PARTY.md` | 第三方依赖说明和上游链接。 |

## 流水线

```text
SQLite sensor database
  -> Stage1: satellite pass rainfall retrieval
  -> Stage1.5: pass outputs aggregated to regular time buckets
  -> Stage2: rainfall forecasting with GPT4TS-style time-series models
```

主要目标定义：

- Stage1：`pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)`
- Stage1.5 / Stage2：固定窗口累计降雨，例如 `rain_10min_mm = rainfall_cumulative(t) - rainfall_cumulative(t - 10min)`

瞬时 `rainfall` 用作诊断或辅助信息，不作为主要累计降雨目标。

## 第三方代码

以下第三方复现仓库在本地使用，但不直接纳入本 GitHub 仓库：

| 本地用途 | 上游 GitHub |
| --- | --- |
| GPT4TS / One Fits All 时序预测后端 | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All |
| LLaMA-MoE 参考代码和服务实验 | https://github.com/pjlab-sys4nlp/llama-moe |

使用时应保留上游项目的许可证、引用和安装说明。更多说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 数据与产物

以下内容通常不纳入本仓库：

- SQLite 数据库和备份
- 原始相机图像
- 生成的 CSV / NPZ 数据集
- 模型权重和 checkpoint，但当前轻量 `Stage1/vision_weather` 默认分类权重作为例外已纳入仓库
- 日志和本地缓存
- 下载或复现的第三方完整仓库

可共享的数据集和模型产物后续可单独发布，例如存放在私有 Hugging Face Dataset / Model 仓库或单位对象存储中。

## 文档

- [Stage1 README](Stage1/README.md) / [中文](Stage1/README_CN.md)
- [Stage1.5 README](Stage1.5/README.md) / [中文](Stage1.5/README_CN.md)
- [Stage2 README](Stage2/README.md) / [中文](Stage2/README_CN.md)
- [Methodology draft](docs/methodology.md)
