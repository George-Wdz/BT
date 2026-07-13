#!/usr/bin/env bash
# Compare channel-mixing (cm), channel-wise/two-stage (cw), and group-attention
# (ga) on the same Stage1 rainfall retrieval dataset.
#
# 默认使用新数据并全量重建 NPZ，保证 cm/cw/ga 三个模型在同一个最新数据集上比较。
# 如果只想复用已有 NPZ，把 --reuse-dataset 改成 1，并指定 --pass-dataset-path。
#
# 常用路径参数：
#   --dataset-dir       NPZ 数据集目录。未指定 --pass-dataset-path 时，新 NPZ 默认保存为：
#                       {dataset-dir}/pass_dataset_{experiment}_{run_ts}.npz
#   --pass-dataset-path 指定某一个 NPZ 文件。reuse-dataset=0 时表示“新 NPZ 输出路径”；
#                       reuse-dataset=1 时表示“复用这个已有 NPZ”。
#   --image-label-csv   指定已有照片天气标签 CSV。复用已有 NPZ 做 cm/cw 对比时建议显式指定，
#                       便于 run_manifest 记录对应标签版本。
#   --incremental-source-npz
#                       incremental-npz=1 时显式指定增量构建的源 NPZ；不指定则自动找最新 NPZ。
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
  python3 run_workflow.py compare-channels
  --cuda-visible-devices 0                                                                       # 训练/评估使用的 GPU；多卡可写 0,1,2,3
  --experiment rain_retrieval_compare_channels                                                   # 实验名称，参与默认 dataset_name 命名
  --db-path /home/wdz/satellite_data/satellite_data.db                                           # 卫星链路和气象数据库路径
  --camera-input-dir /home/wdz/BT/Stage1/rain_retrieval/data/camera                              # 新照片目录；默认会从这里重新生成天气标签
  --label-dir /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels                               # 天气标签 CSV 输出目录，会更新 latest_weather_labels*.csv
  --vision-dir /home/wdz/BT/Stage1/vision_weather                                                 # 视觉分类模型工程目录
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260623_133102_weather_cls_rain_station_balanced_100ep_best_model.pt # 视觉分类模型权重
  --vision-batch-size 64                                                                          # 生成照片天气标签时的 batch size
  --vision-num-workers 8                                                                          # 生成照片天气标签时的 DataLoader worker 数
  --dataset-dir /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets                           # NPZ pass 数据集保存目录
  # --pass-dataset-path /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_compare_channels_20260623_1344.npz # 可选：指定新 NPZ 输出路径；若 reuse-dataset=1，则表示复用这个已有 NPZ
  # --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/20260623_1344_weather_labels_slim.csv                              # 可选：指定已有照片标签 CSV；不指定时 reuse-dataset=0 会重新生成，reuse-dataset=1 默认用 latest
  --reuse-dataset 0                                                                               # 0=使用新数据生成标签/NPZ；1=复用已有 NPZ 和 latest 标签
  --incremental-npz 0                                                                             # 0=全量重建并重新匹配所有历史照片；1=只增量合并新增 pass
  # --incremental-source-npz /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260623_1344.npz # 可选：incremental-npz=1 时显式指定增量构建的源 NPZ；不指定则自动找 dataset-dir 中最新 NPZ
  --checkpoint-base /home/wdz/BT/Stage1/rain_retrieval/model/checkpoints                         # 模型 checkpoint 根目录
  --result-base /home/wdz/BT/Stage1/rain_retrieval/analysis/satellite_weather_diff/runs          # 预测 CSV 和指标输出根目录
  --log-dir /home/wdz/BT/Stage1/rain_retrieval/model/logs                                        # workflow/train 日志目录
  --feature-groups link,position,ground_weather,image_weather,dry_delta                           # 输入特征组，cm/cw/ga 三个变体共用
  --position-mode raw6_geo4                                                                       # 位置特征模式：raw6/raw6_geo2/raw6_geo4/geo4；新增几何特征需重建 NPZ
  --fusion-variants cm,cw,ga                                                                      # 要比较的融合模式：cm=混合投影；cw=通道两阶段；ga=物理组 attention
  --use-modal-encoders 1                                                                         # cw分支使用模态专用轻量编码器
  --use-conditioning 1                                                                           # 所有变体启用相同条件化，保证比较口径一致
  --use-quality-gating 1                                                                         # cw分支启用模态质量门控
  --group-hidden-dim 128                                                                         # ga 模式下每个物理特征组 token 的隐藏维度
  --group-attention-heads 4                                                                      # ga 模式下组间 self-attention 头数
  --group-attention-layers 1                                                                     # ga 模式下组间 self-attention 层数；可试 1/2
  --group-attention-dropout 0.1                                                                  # ga 模式下组间 attention dropout
  --val-strategy stratified_all                                                                  # 数据划分策略，见脚本顶部说明
  --iterations 3                                                                                 # 每个通道注意力变体的独立训练重复次数
  --epochs 100                                                                                   # 每次训练最大 epoch
  --batch-size 64                                                                                # 训练 batch size
  --patience 15                                                                                  # early stopping 容忍 epoch 数
  --lr 0.0001                                                                                    # 初始学习率
  --image-tolerance 6min                                                                        # 相机天气标签和卫星过境的时间匹配窗口
  --dry-baseline-method geo_weighted                                                              # 融合结构对比统一使用几何加权无雨参考
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
  --adaptive-task-weighting 1                                                                    # 所有变体使用相同的有界多任务权重
  --task-log-var-bound 1.5                                                                       # 自适应任务log方差绝对边界
  --task-weight-regularization 0.01                                                              # 防止小数据下任务权重漂移
  --eval-batch-size 128                                                                          # 训练后评估 batch size
)

"${python_cmd[@]}" "$@"
