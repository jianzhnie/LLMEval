#!/bin/bash/

source set_env.sh

hfhub="/home/jianzhnie/llmtuner/hfhub/models"

model_path="${hfhub}/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

num_gpus=2
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
