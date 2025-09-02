#!/bin/bash/

source /root/llmtuner/llm/LLMEval/set_env.sh

model_path="/root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B"
model_name="Qwen/Qwen2.5-7B"

num_gpus=4
max_model_len=2048  # ✅ 支持 2k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len  \
    --chat-template /root/llmtuner/llm/LLMEval/chat_template/qwen25_template.jinja \
    --enforce-eager \
    --port 8090


