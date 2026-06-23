#!/usr/bin/env bash
# Stage1 input-feature ablation workflow.
# This workflow reuses an existing pass_dataset_*.npz and image-label CSV so
# all ablation variants are trained/evaluated on exactly the same samples.
# It does not regenerate camera labels or rebuild the NPZ dataset. Run
# run_rain_retrieval_workflow.sh first if new camera images or DB rows need to
# be included.
# Default variants:
#   full_a       = link + position + ground_weather + image_weather + dry_delta
#   core_e       = link + position + dry_delta
#   no_position  = link + ground_weather + image_weather + dry_delta
#   no_image     = link + position + ground_weather + dry_delta
#   no_ground    = link + position + image_weather + dry_delta
#
# 固定数据建议：
#   消融实验必须尽量固定同一个 NPZ，否则不同特征组合之间的数据样本不同，指标不可直接比较。
#   因此这里默认 reuse-dataset=1，并显式写 --pass-dataset-path。
#   如果想复用最新 NPZ 但不关心具体文件，可以删除 --pass-dataset-path；代码会从
#   --dataset-dir 下按文件修改时间选择最新的 pass_dataset_*.npz。
#   --image-label-csv 只用于记录/配置一致性；当 NPZ 已经构建好时，图像天气特征已经写入 NPZ。

set -euo pipefail

cd /home/wdz/BT/Stage1/rain_retrieval/model

# --val-strategy 可选：
#   stratified_all         雨/无雨样本分别随机打乱后按 70/20/10 切分，train/val/test 都尽量有雨样本，适合当前雨样本稀少时做诊断。
#   stratified_before_test 先按时间保留最后 10% 作为 test，再把前 90% 的雨/无雨样本分层切成 train/val，更接近在线验证。
#   time                   完全按 pass 起始时间顺序切分，前 70% train、中间 20% val、最后 10% test。

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=7                                                                        # 训练使用的 GPU 编号；如需沿用当前环境，删除本行
  python3 run_workflow.py feature-ablation
  --experiment feature_ablation                                                                  # 实验名称，参与默认 dataset_name 命名
  --variants full_a,core_e,no_position,no_image,no_ground                                         # 要运行的特征消融组合，逗号分隔
  --reuse-dataset 1                                                                               # 1=复用老数据；消融固定同一个 NPZ，保证不同特征组合公平对比
  --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260618_1536.npz # 消融复用的 NPZ pass 数据集；不会重新切片
  # --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260623_1344.npz # 示例：改成最近一次 workflow 生成的 NPZ
  --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv # 已生成的相机天气标签 CSV；不会重新预测照片标签
  # --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/20260623_1344_weather_labels_slim.csv             # 示例：显式绑定某一次 run 的照片标签 CSV，便于复现实验
  --dataset-dir /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets                           # 默认查找 NPZ pass 数据集的目录
  --checkpoint-base /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints                         # 模型 checkpoint 根目录
  --result-base /home/wdz/BT/Stage1/rain_retrieval/analysis/satellite_weather_diff/runs          # 预测 CSV 和指标输出根目录
  --log-dir /home/wdz/BT/Stage1/rain_retrieval/model/logs                                        # workflow/train 日志目录
  --val-strategy stratified_all                                                                  # 数据划分策略，见脚本顶部说明
  --iterations 1                                                                                 # 每个消融组合的独立训练重复次数
  --epochs 100                                                                                   # 每次训练最大 epoch
  --batch-size 64                                                                                # 训练 batch size
  --patience 15                                                                                  # early stopping 容忍 epoch 数
  --lr 0.0001                                                                                    # 初始学习率
  --dry-baseline-image-rain-prob-threshold 0.2                                                    # dry baseline 候选中排除视觉雨天的概率阈值
  --auxiliary-loss-weight 0.3                                                                    # 辅助目标损失权重
  --eval-batch-size 128                                                                          # 训练后评估 batch size
)

"${python_cmd[@]}" "$@"
