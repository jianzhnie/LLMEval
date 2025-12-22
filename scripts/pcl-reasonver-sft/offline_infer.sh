#!/bin/bash

source set_env.sh

set -euo pipefail

# --- Configuration ---
output_dir="./output/PCL-Reasoner-v1"
model_name_or_path="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476"

n_samples=64  # Default sample size for aime24 and aime25

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# --- Run Inference Tasks ---
# aime24 (repeated sample 64 times)
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime24.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --batch_size 64 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --max_model_len 131072 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 8 \
    --n_samples "${n_samples}" \
    --system_prompt_type amthinking \
    --temperature 0.6  \
    --top_p 0.95 \
    --top_k 40


# aime25 (repeated sample 64 times)
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime25.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --batch_size 64 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --max_model_len 131072 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 8 \
    --enforce_eager \
    --n_samples "${n_samples}" \
    --system_prompt_type amthinking \
    --temperature 0.6  \
    --top_p 0.95 \
    --top_k 40


echo "🎉 All inference tasks completed successfully!"
