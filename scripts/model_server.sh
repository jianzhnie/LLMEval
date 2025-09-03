#!/bin/bash/

source set_env.sh

model_path="Qwen/QwQ-32B"
model_name="Qwen/QwQ-32B"

num_gpus=8
max_model_len=32768  # ✅ 支持 32k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len  \
    --enforce-eager \
    --port 8090
