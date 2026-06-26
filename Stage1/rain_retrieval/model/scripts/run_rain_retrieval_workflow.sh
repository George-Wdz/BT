#!/usr/bin/env bash
# End-to-end Stage1 rainfall retrieval workflow.
# The orchestration lives in scripts/run_workflow.py; this file keeps the
# experiment knobs visible, similar to Stage2/GPT4TS scripts.
#
# 默认行为：
#   1. 重新扫描 camera-input-dir 中的图片，生成本次 run_ts 对应的天气标签 CSV，并更新 latest_weather_labels*.csv。
#   2. 基于最新已有 pass_dataset_*.npz 和数据库新增行，增量构建一个新的时间戳 NPZ 数据集。
#   3. 使用新 NPZ 训练模型，并评估 train/val/test。
#
# --reuse-dataset:
#   0 = 使用新数据：重新生成照片标签，并增量构建新的时间戳 NPZ。
#   1 = 复用老数据：跳过照片标签生成和 NPZ 构建，直接使用已有 NPZ/latest_weather_labels_slim.csv。
#
# --incremental-npz:
#   0 = 全量重建 NPZ，重新用最新照片标签匹配所有历史 pass；补齐历史照片后应使用这个。
#   1 = 增量构建 NPZ，只重建最新数据库窗口并与旧 NPZ 合并；日常追加新数据时更快。
#
# 常用路径参数：
#   --dataset-dir       NPZ 数据集目录。未指定 --pass-dataset-path 时，新 NPZ 默认保存为：
#                       {dataset-dir}/pass_dataset_{experiment}_{run_ts}.npz
#   --pass-dataset-path 指定某一个 NPZ 文件。reuse-dataset=0 时表示“新 NPZ 输出路径”；
#                       reuse-dataset=1 时表示“复用这个已有 NPZ”。
#   --image-label-csv   指定已有照片天气标签 CSV。通常搭配 reuse-dataset=1 或复现实验使用；
#                       不指定且 reuse-dataset=0 时，会重新跑视觉模型并生成本次 run_ts 标签。
#   --label-dir         天气标签 CSV 输出目录。workflow 每次生成时间戳 CSV 后，会额外复制一份
#                       latest_weather_labels_slim.csv，方便后续复用最新标签。
#
# dry baseline 默认逻辑：
#   先要求气象站累计雨量/瞬时雨强为 0；若有相机标签，则排除视觉模型认为下雨的 pass。
#   默认不强制必须有相机标签，也不强制 sunny 概率阈值，这样 dry baseline 覆盖更稳定。

set -euo pipefail

cd /home/wdz/BT/Stage1/rain_retrieval/model

# --val-strategy 可选：
#   stratified_all         雨/无雨样本分别随机打乱后按 70/20/10 切分，train/val/test 都尽量有雨样本，适合当前雨样本稀少时做诊断。
#   stratified_before_test 先按时间保留最后 10% 作为 test，再把前 90% 的雨/无雨样本分层切成 train/val，更接近在线验证。
#   time                   完全按 pass 起始时间顺序切分，前 70% train、中间 20% val、最后 10% test。

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=4                                                                        # 训练使用的 GPU 编号；如需沿用当前环境，删除本行
  python3 run_workflow.py rain
  --experiment rain_retrieval                                                                    # 实验名称，参与默认 dataset_name 命名
  --db-path /home/wdz/satellite_data/satellite_data.db                                           # 卫星链路和气象数据库路径
  --camera-input-dir /home/wdz/BT/Stage1/rain_retrieval/data/camera                              # 新照片目录；默认会从这里重新生成天气标签
  --label-dir /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels                               # 天气标签 CSV 输出目录，会更新 latest_weather_labels*.csv
  --vision-dir /home/wdz/BT/Stage1/vision_weather                                                 # 视觉分类模型工程目录
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt # 视觉分类模型权重
  --vision-batch-size 64                                                                          # 生成照片天气标签时的 batch size
  --vision-num-workers 8                                                                          # 生成照片天气标签时的 DataLoader worker 数
  --dataset-dir /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets                           # NPZ pass 数据集保存目录
  # --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260623_1344.npz # 可选：指定新 NPZ 输出路径；若 reuse-dataset=1，则表示复用这个已有 NPZ
  # --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/20260623_1344_weather_labels_slim.csv             # 可选：指定已有照片标签 CSV；不指定时 reuse-dataset=0 会重新生成，reuse-dataset=1 默认用 latest
  --reuse-dataset 0                                                                               # 0=使用新数据生成标签/NPZ；1=复用已有 NPZ 和 latest 标签
  --incremental-npz 0                                                                             # 0=全量重建并重新匹配所有历史照片；1=只增量合并新增 pass
  # --incremental-source-npz /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260623_1344.npz # 可选：incremental-npz=1 时显式指定增量构建的源 NPZ；不指定则自动找 dataset-dir 中最新 NPZ
  --incremental-lookback-minutes 20                                                               # 增量构建时回看已有 NPZ 末尾多少分钟，避免 pass 边界断裂
  --checkpoint-base /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints                         # 模型 checkpoint 根目录
  --result-base /home/wdz/BT/Stage1/rain_retrieval/analysis/satellite_weather_diff/runs          # 预测 CSV 和指标输出根目录
  --log-dir /home/wdz/BT/Stage1/rain_retrieval/model/logs                                        # workflow/train 日志目录
  --feature-groups link,position,ground_weather,image_weather,dry_delta                           # 输入特征组，决定 input_dim 和 feature_group_dims
  --position-mode raw6_geo2                                                                            # 位置特征模式：raw6/raw6_geo2/raw6_geo4/geo4；新增几何特征需重建 NPZ
  --val-strategy stratified_all                                                                  # 数据划分策略，见脚本顶部说明
  --iterations 3                                                                                 # 独立训练重复次数
  --epochs 100                                                                                   # 每次训练最大 epoch
  --batch-size 64                                                                                # 训练 batch size
  --patience 15                                                                                  # early stopping 容忍 epoch 数
  --lr 0.0001                                                                                    # 初始学习率
  --image-tolerance 10min                                                                        # 相机天气标签和卫星过境的时间匹配窗口
  --dry-baseline-image-rain-prob-threshold 0.2                                                    # dry baseline 候选中排除视觉雨天的概率阈值
  --dry-baseline-require-image-available 0                                                        # 0=无图像也可参与 dry baseline；1=只用有相机标签的 pass
  --dry-baseline-min-sunny-prob 0.0                                                               # >0 时要求 prob_sunny 不低于该阈值；0=不启用 sunny 阈值
  --auxiliary-loss-weight 0.3                                                                    # 辅助目标损失权重
  --eval-batch-size 128                                                                          # 训练后评估 batch size
)

"${python_cmd[@]}" "$@"
