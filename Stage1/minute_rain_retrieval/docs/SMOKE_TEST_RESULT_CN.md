# 可复现流程Smoke Test记录

## 执行环境

- 日期：2026-09-02
- Python：3.11.15
- NumPy：1.25.0
- PyTorch：2.0.1+cu117
- 设备：单张CUDA GPU
- 数据：`data/reproducible_v1/minute_rainfall_full.npz`

## 命令

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHON=/home/wdz/miniconda3/envs/smoe/bin/python \
OUTPUT_DIR=/tmp/minute_rain_smoke_20260902 \
bash scripts/run_reproducible_smoke_test.sh
```

## 控制台关键输出

```text
device=cuda trainable_parameters=1,152,202 total_parameters=1,152,202 input_dim=20
train_sampling rainy=879 dry_full=17607 dry_per_epoch=879 dry_to_rain=1.000
epoch=001 train_loss=0.947784 val_mae=0.025093 val_rainy_mae=0.099282 val_balanced_mae=0.060339 val_f1=0.2605
```

该测试只运行1个epoch，用于验证数据加载、反向传播、checkpoint保存和三划分评估链路，不代表正式模型性能。正式复现实验使用README中的80 epochs和3:1无雨下采样配置。

## 生成文件

```text
best.pt
best_val_predictions.csv
metrics.json
train_predictions.csv
val_predictions.csv
test_predictions.csv
val_rainy_predictions.csv
test_rainy_predictions.csv
val_test_rainy_predictions.csv
```

本次smoke test的测试集结果为MAE 0.02492 mm、rainy MAE 0.11805 mm、F1 0.2586。其意义仅是流程成功，不用于替换8041部署模型。

训练命令没有历史库参数，也没有部署权重路径，因此不会写入`rain_retrieval_history.sqlite3`或覆盖`weights/deployed/`。
