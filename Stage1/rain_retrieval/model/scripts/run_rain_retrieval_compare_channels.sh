#!/usr/bin/env bash
# Compare channel-mixing (cm) and two-stage channel attention (cw) on the same
# Stage1 rainfall retrieval dataset.

set -euo pipefail

cd /home/wdz/BT/Stage1/rain_retrieval/model

# --val-strategy 可选：
#   stratified_all         雨/无雨样本分别随机打乱后按 70/20/10 切分，train/val/test 都尽量有雨样本，适合当前雨样本稀少时做诊断。
#   stratified_before_test 先按时间保留最后 10% 作为 test，再把前 90% 的雨/无雨样本分层切成 train/val，更接近在线验证。
#   time                   完全按 pass 起始时间顺序切分，前 70% train、中间 20% val、最后 10% test。

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0                                                                        # 训练使用的 GPU 编号；如需沿用当前环境，删除本行
  python3 run_workflow.py compare-channels
  --experiment rain_retrieval_compare_channels                                                   # 实验名称，参与默认 dataset_name 命名
  --db-path /home/wdz/satellite_data/satellite_data.db                                           # 卫星链路和气象数据库路径
  --dataset-dir /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets                           # NPZ pass 数据集保存目录
  --checkpoint-base /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints                         # 模型 checkpoint 根目录
  --result-base /home/wdz/BT/Stage1/rain_retrieval/analysis/satellite_weather_diff/runs          # 预测 CSV 和指标输出根目录
  --log-dir /home/wdz/BT/Stage1/rain_retrieval/model/logs                                        # workflow/train 日志目录
  --feature-groups link,position,ground_weather,image_weather,dry_delta                           # 输入特征组，cm/cw 两个变体共用
  --val-strategy stratified_all                                                                  # 数据划分策略，见脚本顶部说明
  --iterations 1                                                                                 # 每个通道注意力变体的独立训练重复次数
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
