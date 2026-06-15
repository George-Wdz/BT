# Stage1 视觉天气分类模型

本目录实现了 Stage1 使用的轻量视觉天气分类模块，已经迁入 BT 仓库，后续不再依赖外部工程路径。

- 源码：`/home/wdz/BT/Stage1/vision_weather`
- 默认权重：`weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt`
- 当前相机图片：`/home/wdz/BT/Stage1/rain_retrieval/data/camera`
- 当前自动标签输出：`/home/wdz/BT/Stage1/rain_retrieval/data/camera_labels`

模块功能：
- 输入：摄像头天空图像
- 输出：天气类别（默认从子目录自动推断；当前数据集为 3 类）
  - `sunny`（晴）
  - `cloudy`（多云）
  - `rain`（下雨）

## 三分类标注口径（当前版本）

为保证训练一致性，建议按以下标准进行人工标注：

- `sunny`：可清晰看到天空，且有明显晴朗特征（如太阳、月亮，或大面积清澈天空）。
- `cloudy`：可视天空占比偏少，云层较多且较厚，但无明确降雨证据。
- `rain`：画面存在明确降雨迹象（雨丝、雨滴痕迹、湿地反射明显且与降雨一致等）。

标注建议：
- 优先关注“是否在下雨”，若无法判断可先暂存为待复核样本，避免强行标注。
- 保持同一采集时段/同一机位的标注口径一致，减少标签漂移。
- 对高不确定样本建议双人复核，优先保证 `rain` 类质量。

## 为什么默认选 tiny_resnet

当前环境没有 `torchvision`，因此采用纯 PyTorch 的轻量 CNN（`tiny_resnet`）作为默认。
在简单分类场景中，它具备：
- 参数较少，训练稳定
- 对中小数据量较友好

当前阶段仅保留该轻量 CNN 路线，避免无效分支增加维护成本。

## 目录与脚本

- `dataset.py`: 图像数据读取与预处理、train/val 切分
- `models.py`: `WeatherClassifier`（仅 `tiny_resnet`）
- `train_weather_classifier.py`: 训练脚本（只负责训练过程）
- `eval_weather_classifier.py`: 独立验证脚本（可指定任意权重）

## 训练产物目录

训练产物按时间戳命名，统一落在 `logs/` 与 `weights/`：

```text
Stage1/vision_weather/
  logs/
    20260402_120000_train.csv
    20260402_121500_eval.csv (可选)
  weights/
    20260402_120000_train_best_model.pt
```

说明：
- `logs` 训练阶段默认只保存一个 CSV（epoch 的 train/val loss 与 acc）。
- `weights` 只保存训练得到的最佳模型权重（best model）。
- 当前默认权重 `weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt` 已纳入仓库，用于 Stage1 图像标签导出和 LoRA-MoE 视觉专家复现。

## 数据准备

本地项目目录可直接保存训练图片，但图片不推送到 GitHub。当前推荐结构：

```text
Stage1/vision_weather/data/raw/
  sunny/
  cloudy/
  rain/
```

当前本地 `raw/` 已从历史视觉分类目录同步过来，类别数量为：`sunny=8641`、`cloudy=8152`、`rain=1030`。`split/` 中保存了当前训练/验证/测试划分。

当前阶段优先训练分类能力，**不要求文件名携带时间戳**。
如后续需要与链路/环境时序对齐，可在训练时加 `--parse-timestamp` 开启从文件名解析 `timestamp_unix`。

## 训练命令（只训练）

```bash
python Stage1/vision_weather/train_weather_classifier.py \
  --data-dir Stage1/vision_weather/data/raw \
  --output-dir Stage1/vision_weather \
  --run-name weather_cls_v1 \
  --class-names sunny,cloudy,rain \
  --epochs 20 \
  --batch-size 32 \
  --val-ratio 0.1
```

说明：
- 脚本会按类别**分层随机切分** train/val。
- 每个 epoch 都会在 `val` 集上做验证（即日志中的 `val_loss/val_acc`）。
- 训练日志保存为 `logs/<timestamp>_<run_name>.csv`。
- 最佳权重保存为 `weights/<timestamp>_<run_name>_best_model.pt`。

## 验证命令（独立验证，可换新验证集）

```bash
python Stage1/vision_weather/eval_weather_classifier.py \
  --data-dir Stage1/vision_weather/data/split/test \
  --output-dir Stage1/vision_weather \
  --weights Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt \
  --class-names sunny,cloudy,rain \
  --save-csv
```

说明：
- `--weights` 可指定任意权重文件；不传则自动选 `weights/` 下最新文件。
- 验证默认在终端输出 overall + per-class 指标；`--save-csv` 可额外保存一份评估 CSV。

## 生成自动标签（对新图片批量推理）

当你把后续抓取的新图片放到一个文件夹后，可以用训练好的权重自动生成标签 CSV：

```bash
python Stage1/vision_weather/predict_weather_labels.py \
  --input-dir Stage1/rain_retrieval/data/camera \
  --output-dir Stage1/vision_weather \
  --weights Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt
```

说明：
- `--input-dir` 支持递归扫描，图片可按任意子目录组织。
- 不传 `--weights` 时，会自动使用 `weights/` 下最新模型。
- 结果默认保存到 `logs/<timestamp>_pred_labels.csv`。
- 输出包含：`pred_label`、`confidence` 以及每个类别的概率列（如 `prob_sunny`、`prob_cloudy`、`prob_rain`）。

## 数据集落盘划分（可视化 train/val/test）

如果你希望把划分结果直接落到目录，便于人工核查，可用：

```bash
python3 Stage1/vision_weather/prepare_weather_split.py \
  --source-dir Stage1/vision_weather/data/raw \
  --output-dir Stage1/vision_weather/data/split \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --class-names sunny,cloudy,rain \
  --mode hardlink \
  --overwrite
```

输出结构：

```text
Stage1/vision_weather/data/split/
  train/<class_name>/*
  val/<class_name>/*
  test/<class_name>/*
  manifests/
    split_manifest.csv
    split_summary.csv
```

说明：
- `hardlink` 模式不会复制图片内容，节省磁盘空间（Linux 推荐）。
- `split_manifest.csv` 可用于核查任意图片属于哪个 split。
