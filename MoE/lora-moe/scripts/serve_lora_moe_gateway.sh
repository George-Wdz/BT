#!/usr/bin/env bash
# Serve LoRA-MoE gateway with one frontend/API for multiple expert services.

set -euo pipefail

export PYTHONPATH=/home/wdz/BT/MoE/lora-moe/src
export TRITON_CACHE_DIR=/tmp/triton-cache-wdz
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd /home/wdz/BT/MoE/lora-moe

python_cmd=(
  python -m lora_moe.serve.lora_moe_gateway_fastapi
  --host 0.0.0.0                                                                            # Gateway 对外监听地址，前端只访问这个服务
  --port 8020                                                                               # Gateway 对外端口
  --vision-url http://127.0.0.1:8121                                                        # 视觉天气专家内部地址
  --stage1-url http://127.0.0.1:8122                                                        # Stage1 链路反演专家内部地址
  --request-timeout-s 300                                                                   # 等待后端专家推理的超时时间
)

"${python_cmd[@]}" "$@"
