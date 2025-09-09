#!/bin/bash

set -euo pipefail

# --- Configuration ---
output_dir="./output/deekseek-ai/Distill-Qwen-1.5B"
model_name_or_path="/home/jianzhnie/llmtuner/hfhub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

n_samples=64  # Default sample size for aime24 and aime25

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

rope_scaling={"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}

# --- Run Inference Tasks ---

# aime24 (repeated sample 64 times)
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime24.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --batch_size 64 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --max_model_len 32768 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --enforce_eager \
    --n_samples "${n_samples}" \
    --temperature 0.6  \
    --top_p 0.95 \
    --top_k 40 \
    --system_prompt_type deepseek_r1

# aime25 (repeated sample 64 times)
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime25.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --batch_size 64 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --max_model_len 32768 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --enforce_eager \
    --n_samples "${n_samples}" \
    --temperature 0.6  \
    --top_p 0.95 \
    --top_k 40 \
    --system_prompt_type deepseek_r1

echo "🎉 All inference tasks completed successfully!"
