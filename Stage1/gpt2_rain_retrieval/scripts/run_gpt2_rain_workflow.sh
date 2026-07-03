#!/usr/bin/env bash
# GPT2 baseline workflow for Stage1 rainfall retrieval.

set -euo pipefail

cd /home/wdz/BT/Stage1/gpt2_rain_retrieval

python_cmd=(
  python3 run_workflow.py
  --cuda-visible-devices 0                                                                      # 训练可见 GPU；多卡写 0,1,2,3
  --eval-cuda-visible-devices 0                                                                 # 训练后评估使用的 GPU，单卡即可
  --omp-num-threads 1                                                                           # 每个 DDP 进程使用的 CPU OpenMP 线程数；默认 1 可避免 torchrun warning
  # --ddp                                                                                        # 启用多卡 DDP；nproc_per_node 会按 --cuda-visible-devices 自动推断
  --dataset-npz /home/wdz/BT/Stage1/rain_retrieval/model/data/datasets/pass_dataset_rain_retrieval_20260626_1804.npz # 复用的 Stage1 NPZ 数据集
  --image-label-csv /home/wdz/BT/Stage1/rain_retrieval/data/camera_labels/latest_weather_labels_slim.csv # 图像天气标签 CSV，用于配置和复现记录
  --gpt2-model-dir /home/wdz/BT/Stage2/GPT4TS/Long-term_Forecasting/gpt2                         # 本地 GPT2 权重目录
  --val-strategy stratified_all                                                         # stratified_all / stratified_before_test / time
  --position-columns "[longitude,latitude,satAltitude,posLongitude,posLatitude,altitude,slant_range_km,elevation_deg,azimuth_sin,azimuth_cos]" # raw6_geo4 位置特征列
  --input-dim 25                                                                                # link4 + position10 + ground3 + image4 + dry_delta4
  --feature-group-dims "[4,10,3,4,4]"                                                           # 与 feature groups 对齐的维度
  --gpt2-layers 6                                                                               # 使用 GPT2 前 N 层；可试 3/6/12
  --freeze-gpt2 all                                                                             # all=冻结GPT2；ln_wpe=只训练LayerNorm和位置嵌入；none=训练GPT2主干但冻结未使用的文字词表
  --use-group-attention 1                                                                       # 1=先按特征组做 attention 数值编码；0=旧版直接展平 patch
  --group-hidden-dim 128                                                                        # 每个特征组 token 的隐藏维度
  --group-attention-heads 4                                                                     # 特征组 self-attention 头数
  --group-attention-layers 1                                                                    # 特征组 self-attention 层数；可试 1/2
  --group-attention-dropout 0.1                                                                 # 特征组编码器 dropout
  --iterations 3                                                                                # 独立重复训练次数
  --batch-size 32                                                                               # 单卡 batch；DDP 时是每张 GPU 的 batch
  --epochs 100                                                                                  # 每次训练最大 epoch
  --patience 15                                                                                 # early stopping 容忍 epoch 数
  --lr 0.0001                                                                                   # AdamW 初始学习率
  --weight-decay 0.00001                                                                        # AdamW 权重衰减
  --use-cosine 1                                                                                # 1=CosineAnnealingLR；0=使用配置中的阶梯衰减 lradj
  --tmax 100                                                                                     # 余弦退火周期；use-cosine=1 时生效，eta_min 固定为 1e-8
)

"${python_cmd[@]}" "$@"
