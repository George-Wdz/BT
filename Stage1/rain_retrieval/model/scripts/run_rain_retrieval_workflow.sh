#!/usr/bin/env bash
# End-to-end Stage1 rainfall retrieval workflow.
# The orchestration lives in scripts/run_workflow.py; this file keeps the
# experiment knobs visible, similar to Stage2/GPT4TS scripts.

set -euo pipefail

cd /home/wdz/BT/Stage1/rain_retrieval/model

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0                                                                        # 训练使用的 GPU 编号；如需沿用当前环境，删除本行
  python3 run_workflow.py rain
  --experiment rain_retrieval                                                                    # 实验名称，参与默认 dataset_name 命名
  --db-path /home/wdz/satellite_data/satellite_data.db                                           # 卫星链路和气象数据库路径
  --dataset-dir /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets                           # NPZ pass 数据集保存目录
  --checkpoint-base /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints                         # 模型 checkpoint 根目录
  --result-base /home/wdz/BT/Stage1/rain_retrieval/analysis/satellite_weather_diff/runs          # 预测 CSV 和指标输出根目录
  --log-dir /home/wdz/BT/Stage1/rain_retrieval/model/logs                                        # workflow/train 日志目录
  --feature-groups link,position,ground_weather,image_weather,dry_delta                           # 输入特征组，决定 input_dim 和 feature_group_dims
  --val-strategy stratified_all                                                                  # train/val/test 划分策略
  --iterations 1                                                                                 # 独立训练重复次数
  --epochs 100                                                                                   # 每次训练最大 epoch
  --batch-size 64                                                                                # 训练 batch size
  --patience 15                                                                                  # early stopping 容忍 epoch 数
  --lr 0.0001                                                                                    # 初始学习率
  --image-tolerance 10min                                                                        # 相机天气标签和卫星过境的时间匹配窗口
  --dry-baseline-image-rain-prob-threshold 0.2                                                    # dry baseline 候选中排除视觉雨天的概率阈值
  --auxiliary-loss-weight 0.3                                                                    # 辅助目标损失权重
  --eval-batch-size 128                                                                          # 训练后评估 batch size
)

"${python_cmd[@]}" "$@"
