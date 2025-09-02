#!/bin/bash
# Description: Evaluate inference outputs for multiple datasets
# Author: (your name)
# Date: 2025-04-27

set -euo pipefail

# --- Configuration ---
# output_dir="./output/Qwen-32B-Distill"
output_dir="./output/Qwen/Qwen2.5-7B"
n_samples=16  # Default sample size for aime24 and aime25

# Evaluation output directory
reval_dir="${output_dir}/eval_score/"

# Create evaluation directory if it doesn't exist
mkdir -p "${reval_dir}"

# --- Evaluate Each Task ---
# Evaluate math500
python ./eval/eval.py \
    --input_path "${output_dir}/math500_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/math500_bz${n_samples}.jsonl" \
    --task_name "math_opensource/math500" \
    > "${reval_dir}/math500_bz${n_samples}_res_result.txt"


# Evaluate aime24
python ./eval/eval.py \
    --input_path "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime24_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime24" \
    > "${reval_dir}/aime24_bz${n_samples}_res_result.txt"

# Evaluate aime25
python ./eval/eval.py \
    --input_path "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime25_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime25" \
    > "${reval_dir}/aime25_bz${n_samples}_res_result.txt"

echo "🎯 Evaluation completed successfully!"
