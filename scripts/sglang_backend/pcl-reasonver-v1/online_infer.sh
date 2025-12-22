#!/bin/bash
source set_env_sglang.sh

set -euo pipefail

# --- Configuration ---
output_dir="./output/PCL-Reasoner-v1-sglangv2"
model_name="PCL-Reasoner-v1-sglang"

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
    --system_prompt_type amthinking \
    --max_workers 128 \
    --temperature 0.6  \
    --top_p 0.95 \
    --top_k 40

# aime25 (repeated sample 64 times)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime25.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --system_prompt_type amthinking \
    --max_workers 128 \
    --temperature 0.6  \
    --top_p 0.95 \
    --top_k 40

echo "🎉 All inference tasks completed successfully!"
