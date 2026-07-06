#!/usr/bin/env bash
# Serve two plain Qwen instances and a Gateway that lets them talk to each other.
#
# 对外前端只访问 Gateway：
#   http://服务器IP:8030
#
# Ollama 兼容接口：
#   GET  /api/tags
#   POST /api/generate
#   POST /api/chat
#
# 内部普通 Qwen 服务：
#   Qwen-A -> http://127.0.0.1:8131
#   Qwen-B -> http://127.0.0.1:8132

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

mkdir -p logs/dual_qwen

qwen_a_cmd=(
  python -m lora_moe.serve.general_qwen_fastapi
  --cuda-visible-devices 0,1                                                              # Qwen-A 使用的 GPU 编号；14B bf16 通常建议 2 张 4090
  --role-name qwen_a                                                                       # 服务标识，便于日志和 health 区分
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --host 127.0.0.1                                                                         # 内部服务只监听本机
  --port 8131                                                                              # Qwen-A 内部端口
  --device-map auto                                                                        # Transformers 自动切分到可见 GPU
  --dtype bfloat16                                                                         # 4090 可用 bfloat16；显存紧张可考虑量化方案
)

qwen_b_cmd=(
  python -m lora_moe.serve.general_qwen_fastapi
  --cuda-visible-devices 2,3                                                              # Qwen-B 使用的 GPU 编号；避免和 Qwen-A 抢显存
  --role-name qwen_b                                                                       # 服务标识，便于日志和 health 区分
  --model-dir /home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct                                 # Qwen2.5-14B-Instruct 基座路径
  --host 127.0.0.1                                                                         # 内部服务只监听本机
  --port 8132                                                                              # Qwen-B 内部端口
  --device-map auto                                                                        # Transformers 自动切分到可见 GPU
  --dtype bfloat16                                                                         # 4090 可用 bfloat16；显存紧张可考虑量化方案
)

gateway_cmd=(
  python -m lora_moe.serve.dual_qwen_gateway_fastapi
  --host 0.0.0.0                                                                           # Gateway 对外监听地址
  --port 8030                                                                              # Gateway 对外端口；前端/Ollama 客户端连接这个端口
  --qwen-a-url http://127.0.0.1:8131                                                       # Qwen-A 内部地址
  --qwen-b-url http://127.0.0.1:8132                                                       # Qwen-B 内部地址
  --request-timeout-s 600                                                                  # 双模型串行调用可能较慢，超时设置长一些
)

pids=()

"${qwen_a_cmd[@]}" > logs/dual_qwen/qwen_a.log 2>&1 &
pids+=("$!")

"${qwen_b_cmd[@]}" > logs/dual_qwen/qwen_b.log 2>&1 &
pids+=("$!")

"${gateway_cmd[@]}" > logs/dual_qwen/gateway.log 2>&1 &
pids+=("$!")

echo "Dual Qwen demo started."
echo "Gateway: http://0.0.0.0:8030"
echo "Qwen-A:  http://127.0.0.1:8131"
echo "Qwen-B:  http://127.0.0.1:8132"
echo "Ollama-compatible model name: dual-qwen-demo"
echo "Logs:    /home/wdz/BT/MoE/lora-moe/logs/dual_qwen"
echo "PIDs:    ${pids[*]}"
echo "Press Ctrl+C to stop all services."

trap 'kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true' EXIT INT TERM
wait
