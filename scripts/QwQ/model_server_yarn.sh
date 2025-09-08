#!/bin/bash/

source set_env.sh

hfhub="/home/jianzhnie/llmtuner/hfhub/models"

model_path="${hfhub}/Qwen/QwQ-32B"
model_name="Qwen/QwQ-32B"

num_gpus=8
gpu_memory_utilization=0.9
rope_factor=4
original_max_position_embeddings=32768
max_model_len=$((original_max_position_embeddings * rope_factor))

python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --trust-remote-code \
    --served-model-name "$model_name" \
    --tensor-parallel-size "$num_gpus" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len "$max_model_len" \
    --enforce-eager \
    --port 8090
