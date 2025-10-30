#!/bin/bash

set -euo pipefail

# --- Configuration ---
output_dir="./output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

base_url="http://127.0.0.1:8090/v1"
n_samples=64  # Default sample size for aime24 and aime25

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# --- Run Inference Tasks ---

# aime24 (repeated sample 64 times)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime24.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --temperature 0.6  \
    --system_prompt_type empty \
    --max_workers 64

# aime25 (repeated sample 64 times)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime25.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --temperature 0.6  \
    --system_prompt_type empty \
    --max_workers 64

echo "🎉 All inference tasks completed successfully!"
