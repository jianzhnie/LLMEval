#!/bin/bash

set -euo pipefail

# --- Configuration ---
output_dir="./output/Qwen/Qwen2.5-7B"
model_name="Qwen/Qwen2.5-7B"

base_url="http://127.0.0.1:8090/v1"
n_samples=1  # Default sample size for aime24 and aime25

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# --- Run Inference Tasks ---

# aime24 (repeated sample 64 times)
python ./llmeval/vllm_utils/infer_multithread.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --max_workers 8 

# aime25 (repeated sample 64 times)
python ./llmeval/vllm_utils/infer_multithread.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --max_workers 8 

# math500
python ./llmeval/vllm_utils/infer_multithread.py \
    --input_file "./data/math500.jsonl" \
    --output_file "${output_dir}/math500_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --max_workers 8 


echo "🎉 All inference tasks completed successfully!"
