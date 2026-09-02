# 可复现数据集v1

该目录提供无需原始采集数据库即可执行训练和验证的固定数据版本。数据许可遵循仓库根目录`LICENSE`。

- `minute_rainfall_full.npz`：训练入口，包含全部样本及固定`split`字段。
- `train/`、`val/`、`test/`：从完整NPZ导出的独立划分、索引和摘要。
- `camera_weather_labels.csv`：构建该版本时使用的天空图像分类结果，不包含原始图像。
- `dataset_manifest.json`：数据语义、划分数量和SHA-256校验值。

独立划分文件用于检查样本，不应分别传给当前`train.py`。训练命令见项目`README_CN.md`。

`stratified_all`在全时间范围内按降雨类别分层随机划分，因此测试结果不代表严格未来时间泛化性能。
