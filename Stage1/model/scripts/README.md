# Stage1 脚本说明

## 推荐流程

一键运行完整流程：

```bash
cd /home/wdz/BT/Stage1/model
ITERATIONS=1 EPOCHS=100 BATCH_SIZE=32 PATIENCE=15 \
bash scripts/run_rain_retrieval_workflow.sh
```

默认会生成同一个时间戳对应的一组文件：

- 视觉分类标签：`/home/wdz/BT/Stage1/data/camera_labels/YYYYMMDD_HHMM_weather_labels_slim.csv`
- NPZ 数据集：`/home/wdz/BT/Stage1/model/data/datasets/pass_dataset_rain_retrieval_YYYYMMDD_HHMM.npz`
- 最优权重：`/home/wdz/BT/Stage1/model/checkpoints/pass_dataset_rain_retrieval_YYYYMMDD_HHMM/`
- 训练日志：`/home/wdz/BT/Stage1/model/logs/pass_dataset_rain_retrieval_YYYYMMDD_HHMM_train.log`
- 评估结果：`/home/wdz/BT/Stage1/analysis/satellite_weather_diff/runs/pass_dataset_rain_retrieval_YYYYMMDD_HHMM/`

评估目录中包含：

- `*_predictions.csv`：train/val/test 全量过境预测
- `*_test_predictions.csv`：只包含测试集，列为卫星 ID、过境开始/结束时间、真实雨量、预测雨量、绝对误差
- `*_metrics.csv`：MAE、MSE、RMSE 等指标表
- `link_diff/`：同卫星雨/非雨链路差异分析
- `run_manifest.csv`：本次实验的标签、数据集、权重和结果路径索引

常用覆盖参数：

```bash
RUN_TS=20260610_1530 \
DATASET_NAME=pass_dataset_rain_retrieval_test \
ITERATIONS=1 EPOCHS=80 LR=5e-5 \
bash scripts/run_rain_retrieval_workflow.sh
```

只生成最新相机天气标签和 NPZ：

```bash
cd /home/wdz/BT/Stage1/model
BUILD_NPZ=1 REBUILD_NPZ=1 bash scripts/predict_camera_weather.sh
```

训练当前推荐模型：

```bash
VAL_STRATEGY=stratified_all \
ITERATIONS=1 EPOCHS=100 BATCH_SIZE=32 PATIENCE=15 \
bash scripts/train_experiments.sh rain_retrieval
```

评估并导出逐 pass 预测：

```bash
python3 scripts/evaluate_checkpoint_splits.py \
  --checkpoint-dir /home/wdz/BT/Stage1/model/checkpoints_stage1_rain_retrieval_stratified/stage1_cm_dm256_df512_eh8_el3_dl2_pl8_st4_bs32_lr0.0001_itr0 \
  --batch-size 128 \
  --out-csv /home/wdz/BT/Stage1/analysis/satellite_weather_diff/rain_retrieval_stratified_predictions.csv \
  --test-csv /home/wdz/BT/Stage1/analysis/satellite_weather_diff/rain_retrieval_stratified_test_predictions.csv \
  --metrics-csv /home/wdz/BT/Stage1/analysis/satellite_weather_diff/rain_retrieval_stratified_metrics.csv
```

## 当前推荐模型

`rain_retrieval` 包含：

- link 输入：`phyRssi,rssi,snr,lastCniValue`
- 相机天气分类概率：`prob_sunny,prob_cloudy,prob_rain,image_available`
- drybase 链路差分：当前链路指标减去同卫星 dry baseline
- summary token：过境级统计信息
- 瞬时雨量辅助任务：`rain_rate_mean,rain_rate_max,rainy_ratio`

## drybase 计算

只使用训练集构建 dry baseline，验证集和测试集不会参与 baseline 估计。

候选 dry pass 必须同时满足：

- `pass_rainfall_mm <= rain_threshold`
- `rain_rate_max <= rain_threshold`
- 如果有相机标签，则 `prob_rain < image_rain_prob_threshold`

对每颗卫星，把候选 dry pass 的链路指标在所有时间步上求均值，得到：

```text
dry_mean[satellite_id] = mean(link_features of dry candidate passes)
```

训练/验证/测试样本的 drybase 特征为：

```text
link_dry_delta = current_link_features - dry_mean[satellite_id]
```

如果某颗卫星在训练集中没有合格 dry pass，则使用训练集全局 dry mean 作为 fallback。
