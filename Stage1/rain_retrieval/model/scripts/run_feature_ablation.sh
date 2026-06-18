#!/usr/bin/env bash
# Stage1 input-feature ablation workflow.
# Default variants:
#   core_e      = link + position + dry_delta
#   no_position = link + ground_weather + image_weather + dry_delta

set -euo pipefail

cd /home/wdz/BT/Stage1/rain_retrieval/model

# --val-strategy 可选：
#   stratified_all         雨/无雨样本分别随机打乱后按 70/20/10 切分，train/val/test 都尽量有雨样本，适合当前雨样本稀少时做诊断。
#   stratified_before_test 先按时间保留最后 10% 作为 test，再把前 90% 的雨/无雨样本分层切成 train/val，更接近在线验证。
#   time                   完全按 pass 起始时间顺序切分，前 70% train、中间 20% val、最后 10% test。

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0                                                                        # 训练使用的 GPU 编号；如需沿用当前环境，删除本行
  python3 run_workflow.py feature-ablation
  --experiment feature_ablation                                                                  # 实验名称，参与默认 dataset_name 命名
  --variants core_e,no_position                                                                  # 要运行的特征消融组合，逗号分隔
  --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv # 已生成的相机天气标签 CSV
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
