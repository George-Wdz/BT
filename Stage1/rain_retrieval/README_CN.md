# Stage1：pass 级降雨反演

[English](README.md) | [中文](README_CN.md)

Stage1 使用卫星链路遥测、星地传播几何、地面气象和可选的相机天气概率，反演一次卫星过境期间的降雨量。它解决的是当前过境窗口内的降雨反演，不负责未来预测；未来预测由 Stage2 处理。

## 范围

| 项 | 说明 |
| --- | --- |
| 样本单位 | 一次卫星过境 pass，按卫星 ID 和时间连续性切分。 |
| 主目标 | `pass_rainfall_mm = rainfall_cumulative(pass_end) - rainfall_cumulative(pass_start)` |
| 可选辅助目标 | `rain_rate_mean`、`rain_rate_max`、`rainy_ratio` |
| 主要模型 | pass-based Patch Encoder-Decoder Transformer |
| 推荐流程 | `Stage1/rain_retrieval/model/scripts/run_rain_retrieval_workflow.sh` |

## 输入特征

当前推荐配置使用：

| 分组 | 特征 |
| --- | --- |
| 链路 | `phyRssi`、`rssi`、`snr`、`lastCniValue` |
| 传播几何（geo4） | `slant_range_km`、`elevation_deg`、`azimuth_sin`、`azimuth_cos` |
| 地面气象 | `temperature`、`humidity`、`pressure` |
| 图像天气 | `prob_sunny`、`prob_cloudy`、`prob_rain`、`image_available` |
| dry baseline 差分 | 当前链路特征减去训练集无雨基线 |

默认输入维度：

```text
4 + 4 + 3 + 4 + 4 = 19
```

## 数据对齐

数据构建逻辑在 `Stage1/rain_retrieval/model/data/preprocessing.py`。

| 步骤 | 口径 |
| --- | --- |
| pass 切分 | 同一卫星，相邻链路样本间隔小于 60 秒。 |
| 最短 pass | 至少 10 个有效链路样本。 |
| 位置对齐 | 最近邻匹配，容忍 5 秒。 |
| 地面气象对齐 | 最近邻匹配，容忍 60 秒。 |
| 图像天气对齐 | 按 pass 中心匹配最近图像标签，推荐流程默认容忍 6 分钟。 |
| 雨量标签 | 对 `rainfall_cumulative` 在 pass 起止边界做日内插值，再差分。 |

瞬时 `rainfall` 只作为诊断字段或辅助监督，不作为主累计雨量目标。

## 模型

模型文件：

```text
Stage1/rain_retrieval/model/patch_encoder_decoder.py
```

核心模块：

- 对不规则 pass 序列做重叠 patch embedding；
- 加入 satellite embedding，并保留未知卫星槽位；
- 可选 pass 统计 summary token；
- Transformer encoder 和可学习 target-query decoder；
- 非负雨量回归头；
- rain/no-rain 分类头；
- 可选辅助回归头。

支持三种融合模式：

| 模式 | 配置 | 说明 |
| --- | --- | --- |
| channel-mixing | `model.fusion_mode=cm` | 拼接所有特征后做 patch embedding。 |
| channel-wise | `model.fusion_mode=cw` | 各特征组独立编码，再做时间注意力和模态注意力。当前默认。 |
| group-attention | `model.fusion_mode=ga` | patch 内先做物理特征组注意力，再进行时间编码。 |

当前模型是单路径 CW 模型：累计雨量回归、rain/no-rain 分类和辅助目标共享同一个
多模态编码器和 target-query decoder，不包含独立的 CM 回归路径。模型加入：

- 轻量模态专用编码器与质量门控；
- 卫星、时间和传播几何条件化 LayerNorm；
- 每个辅助目标独立的 target query 与输出头；

时间周期和 pass 长度在加载 NPZ 时动态派生。默认输入直接使用 geo4，因此 NPZ
必须包含几何派生列；只有 raw6 而没有 geo4 的旧 NPZ 需要重建。自适应多任务权重
保留为实验选项，但因小数据下波动较大，默认关闭。

默认 dry baseline 使用训练集内同卫星、几何相近的无雨 pass 做 top-k 加权，并与
同卫星无雨均值混合；验证集和测试集标签不参与 baseline 拟合。

## 训练

完整 Stage1 流程：

```bash
cd /home/wdz/BT/Stage1/rain_retrieval/model
bash scripts/run_rain_retrieval_workflow.sh
```

常用设置，包括 GPU、路径、输入特征组、epoch、batch size、patience 和学习率，
都直接写在脚本里的 `python_cmd` 数组中。临时覆盖某个参数时，可以把命令行参数追加到脚本后：

```bash
bash scripts/run_rain_retrieval_workflow.sh --lr 0.0002 --epochs 50
```

该流程会：

1. 导出相机天气标签；
2. 构建带时间戳的 NPZ pass 数据集；
3. 训练推荐的降雨反演模型；
4. 评估 train/validation/test；
5. 输出运行索引和指标。

在已有 pass 数据集上运行输入特征消融：

```bash
cd /home/wdz/BT/Stage1/rain_retrieval/model
bash scripts/run_feature_ablation.sh \
  --pass-dataset-path data/datasets/pass_dataset_rain_retrieval_20260617_1806.npz
```

默认特征消融组合：

```text
full_a        = link + geo4 + ground_weather + image_weather + dry_delta
no_image      = link + geo4 + ground_weather + dry_delta
no_ground     = link + geo4 + image_weather + dry_delta
no_dry_delta  = link + geo4 + ground_weather + image_weather
```

如果只需要低层单次训练，可以直接调用 `python main.py --set ...`。

## 数据划分

默认比例：

```text
train / validation / test = 0.7 / 0.2 / 0.1
```

可选策略：

| 策略 | 用途 |
| --- | --- |
| `stratified_all` | 对雨/无雨样本分别打乱并切分，适合雨样本稀少时做模型诊断。 |
| `stratified_before_test` | 最后 10% 时间段保留为 test，之前的数据做 train/validation 分层，更接近在线验证。 |

Scaler、卫星 ID 映射和 dry baseline 只使用训练集拟合，避免验证/测试信息泄漏。

## 输出

`evaluate_checkpoint_splits.py` 除总体及雨/无雨回归指标外，还报告降雨分类的
precision、recall、F1、PR-AUC、ROC-AUC、误报率，以及雨量等级、卫星、图像可用性
和（数据具备时）仰角区间切片指标。

以下产物暂不纳入本仓库：

| 输出 | 说明 |
| --- | --- |
| `Stage1/rain_retrieval/model/data/**/*.npz` | pass 数据集 |
| `Stage1/rain_retrieval/model/checkpoints*/` | 模型 checkpoint |
| `Stage1/rain_retrieval/model/logs/` | 训练日志 |
| `Stage1/rain_retrieval/analysis/**/runs/` | 评估结果 |

数据集和模型产物后续可单独发布，例如存放在私有 Hugging Face 仓库或单位对象存储中。
