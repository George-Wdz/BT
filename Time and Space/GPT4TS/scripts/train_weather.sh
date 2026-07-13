#!/bin/bash

# # Setup NVIDIA compatibility for GPU training
# source ~/nvidia_compat/setup_env.sh

# # Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gpt4ts

export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS
gpt_layers=6
freq_list=(10min h)

# 创建日志目录
log_dir=./logs/custom_weather
mkdir -p $log_dir

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/training_${timestamp}.log"

echo "Training started at $(date)" | tee $log_file
echo "Log file: $log_file" | tee -a $log_file
echo "========================================" | tee -a $log_file

for percent in 100
do
for pred_len in 96 192 336
do
for freq in "${freq_list[@]}"
do

echo "Training with pred_len=$pred_len, freq=$freq" | tee -a $log_file

python main.py \
    --root_path ./datasets/weather/ \
    --data_path weather_512.csv \
    --model_id weather_custom_$model'_'$gpt_layers'_'$seq_len'_'$pred_len'_'$freq'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 6 \
    --c_out 6 \
    --freq $freq \
    --target '雨量(mm)' \
    --lradj type3 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --is_gpt 1 2>&1 | tee -a $log_file

done
done
done

echo "========================================" | tee -a $log_file
echo "Training completed at $(date)" | tee -a $log_file
echo "Results saved in ./results/" | tee -a $log_file
echo "Log saved to: $log_file"
