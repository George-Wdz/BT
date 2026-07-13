#!/usr/bin/env bash
# Stage1 input-feature ablation workflow.
# This workflow reuses an existing pass_dataset_*.npz and image-label CSV so
# all ablation variants are trained/evaluated on exactly the same samples.
# It does not regenerate camera labels or rebuild the NPZ dataset. Run
# run_rain_retrieval_workflow.sh first if new camera images or DB rows need to
# be included.
# Default variants:
#   full_a       = link + position + ground_weather + image_weather + dry_delta
#   no_image     = link + position + ground_weather + dry_delta
#   no_ground    = link + position + image_weather + dry_delta
#   no_dry_delta = link + position + ground_weather + image_weather
#
# 固定数据建议：
#   消融实验必须尽量固定同一个 NPZ，否则不同特征组合之间的数据样本不同，指标不可直接比较。
#   因此这里默认 reuse-dataset=1，并显式写 --pass-dataset-path。
#   如果想复用最新 NPZ 但不关心具体文件，可以删除 --pass-dataset-path；代码会从
#   --dataset-dir 下按文件修改时间选择最新的 pass_dataset_*.npz。
#   --image-label-csv 只用于记录/配置一致性；当 NPZ 已经构建好时，图像天气特征已经写入 NPZ。
#
# dry baseline 默认逻辑：
#   先要求气象站累计雨量/瞬时雨强为 0；若有相机标签，则排除视觉模型认为下雨的 pass。
#   默认不强制必须有相机标签，也不强制 sunny 概率阈值。

set -euo pipefail

cd /home/wdz/BT/Stage1/rain_retrieval/model

# --val-strategy 可选：
#   stratified_all         雨/无雨样本分别随机打乱后按 70/20/10 切分，train/val/test 都尽量有雨样本，适合当前雨样本稀少时做诊断。
#   stratified_before_test 先按时间保留最后 10% 作为 test，再把前 90% 的雨/无雨样本分层切成 train/val，更接近在线验证。
#   time                   完全按 pass 起始时间顺序切分，前 70% train、中间 20% val、最后 10% test。

python_cmd=(
  python3 run_workflow.py feature-ablation
  --cuda-visible-devices 7                                                                       # 训练/评估使用的 GPU；多卡可写 0,1,2,3
  --experiment feature_ablation                                                                  # 实验名称，参与默认 dataset_name 命名
  --variants full_a,no_image,no_ground,no_dry_delta                                               # 只消融可选信息源；geo4位置作为物理必需量固定保留
  --reuse-dataset 1                                                                               # 1=复用老数据；消融固定同一个 NPZ，保证不同特征组合公平对比
  --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260618_1536.npz # 消融复用的 NPZ pass 数据集；不会重新切片
  # --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260623_1344.npz # 示例：改成最近一次 workflow 生成的 NPZ
  --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv # 已生成的相机天气标签 CSV；不会重新预测照片标签
  # --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/20260623_1344_weather_labels_slim.csv             # 示例：显式绑定某一次 run 的照片标签 CSV，便于复现实验
  --dataset-dir /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets                           # 默认查找 NPZ pass 数据集的目录
  --checkpoint-base /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints                         # 模型 checkpoint 根目录
  --result-base /home/wdz/BT/Stage1/rain_retrieval/analysis/satellite_weather_diff/runs          # 预测 CSV 和指标输出根目录
  --log-dir /home/wdz/BT/Stage1/rain_retrieval/model/logs                                        # workflow/train 日志目录
  --position-mode geo4                                                                            # 位置统一使用传播几何四维；复用NPZ必须包含geo4
  --fusion-mode cw                                                                               # 使用新版多模态独立编码与时间/通道两阶段融合
  --group-hidden-dim 128                                                                         # ga 模式下每个物理特征组 token 的隐藏维度
  --group-attention-heads 4                                                                      # ga 模式下组间 self-attention 头数
  --group-attention-layers 1                                                                     # ga 模式下组间 self-attention 层数；可试 1/2
  --group-attention-dropout 0.1                                                                  # ga 模式下组间 attention dropout
  --val-strategy stratified_all                                                                  # 数据划分策略，见脚本顶部说明
  --iterations 1                                                                                 # 每个消融组合的独立训练重复次数
  --epochs 100                                                                                   # 每次训练最大 epoch
  --batch-size 64                                                                                # 训练 batch size
  --patience 15                                                                                  # early stopping 容忍 epoch 数
  --lr 0.0001                                                                                    # 初始学习率
  --dry-baseline-method geo_weighted                                                              # 所有特征消融固定使用几何加权无雨参考
  --dry-baseline-image-rain-prob-threshold 0.2                                                    # dry baseline 候选中排除视觉雨天的概率阈值
  --dry-baseline-require-image-available 0                                                        # 0=无图像也可参与 dry baseline；1=只用有相机标签的 pass
  --dry-baseline-min-sunny-prob 0.0                                                               # >0 时要求 prob_sunny 不低于该阈值；0=不启用 sunny 阈值
  --dry-baseline-geo-top-k 5                                                                       # geo_weighted 中参与加权的相近几何 dry pass 数
  --dry-baseline-geo-min-candidates 2                                                             # 同卫星几何候选少于该值时退回同卫星均值
  --dry-baseline-geo-blend-alpha 0.5                                                              # final=alpha*几何 baseline+(1-alpha)*同卫星均值
  --dry-baseline-geo-slant-scale-km 500                                                           # slant range 差异归一化尺度，单位 km
  --dry-baseline-geo-elevation-scale-deg 10                                                       # elevation 差异归一化尺度，单位 degree
  --dry-baseline-geo-azimuth-scale-deg 45                                                         # azimuth 差异归一化尺度，单位 degree
  --auxiliary-loss-weight 0.3                                                                    # 辅助目标损失权重
  --adaptive-task-weighting 0                                                                    # 固定任务权重，避免与特征消融混杂
  --task-log-var-bound 1.5                                                                       # 自适应任务log方差绝对边界
  --task-weight-regularization 0.01                                                              # 防止小数据下任务权重漂移
  --eval-batch-size 128                                                                          # 训练后评估 batch size
)

"${python_cmd[@]}" "$@"
