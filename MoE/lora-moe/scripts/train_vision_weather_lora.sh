#!/usr/bin/env bash
# Train vision-weather LoRA-MoE expert.

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

python_cmd=(
  env
  CUDA_VISIBLE_DEVICES=0,1                                                                 # 训练使用的 GPU 编号；如需沿用当前环境，删除本行
  python -m lora_moe.train.vision_weather_lora
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --split-root /home/wdz/BT/Stage1/vision_weather/data/split                                # 视觉天气分类数据 split 根目录
  --vision-weights /home/wdz/BT/Stage1/vision_weather/weights/20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt # 冻结视觉分类器权重
  --output-dir /home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_smoke                  # LoRA adapter、projector 和训练参数输出目录
  --device-map auto                                                                         # Qwen 多卡切分方式
  --dtype bfloat16                                                                          # Qwen 权重精度
  --batch-size 1                                                                            # 每步样本数
  --grad-accum-steps 8                                                                      # 梯度累积步数，有效 batch=batch_size*grad_accum_steps
  --epochs 1                                                                                # 训练轮数
  --max-steps 20                                                                            # 最大优化步数，0 表示按 epochs 跑完
  --max-train-samples 64                                                                    # 最大训练样本数，0 表示使用完整训练集
  --max-val-samples 128                                                                     # 最大验证样本数，0 表示使用完整验证集
  --save-steps 10                                                                           # 每隔多少优化步保存 checkpoint，0 表示只保存最终产物
  --eval-steps 0                                                                            # 每隔多少优化步跑验证，0 表示不中途验证
  --early-stopping-patience 0                                                               # 早停容忍次数，0 表示关闭
  --early-stopping-min-delta 1e-4                                                           # 早停最小改进阈值
  --learning-rate 2e-4                                                                      # LoRA + projector 学习率
  --num-visual-tokens 8                                                                     # 图像特征映射成的 soft token 数
  --projector-hidden-dim 1024                                                               # projector MLP 隐层维度
  --lora-r 8                                                                                # LoRA rank
  --lora-alpha 16                                                                           # LoRA alpha，通常为 2*r
  --lora-dropout 0.05                                                                       # LoRA dropout
  --lora-target-modules q_proj,v_proj                                                       # 注入 LoRA 的 Qwen 模块
)

"${python_cmd[@]}" "$@"
