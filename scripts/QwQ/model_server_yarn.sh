#!/bin/bash

set -e

if [[ ! -f "set_env.sh" ]]; then
    echo "Error: set_env.sh not found." >&2
    exit 1
fi
source set_env.sh

hfhub="/home/jianzhnie/llmtuner/hfhub/models"
model_name="Qwen/QwQ-32B"
model_path="${hfhub}/${model_name}"

if [[ ! -d "$model_path" ]]; then
    echo "Error: Model path not found: $model_path" >&2
    exit 1
fi

num_gpus=8
gpu_memory_utilization=0.9
max_model_len=32768
rope_factor=2.0

# 构造 rope-scaling 参数，引用 max_model_len 变量
rope_scaling="{\"rope_type\":\"yarn\",\"factor\":${rope_factor},\"original_max_position_embeddings\":${max_model_len}}"
expanded_max_len=$(awk "BEGIN {print int((${max_model_len}) * (${rope_factor}))}")

vllm_args=(
    --model "$model_path"
    --trust-remote-code
    --served-model-name "$model_name"
    --tensor-parallel-size "$num_gpus"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --rope-scaling "$rope_scaling"
    --max-model-len "$expanded_max_len"
    --enforce-eager
    --port 8090
)

echo "Starting vLLM server for model: $model_name on port 8090..."
python -m vllm.entrypoints.openai.api_server "${vllm_args[@]}"
