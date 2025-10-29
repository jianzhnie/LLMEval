#!/bin/bash
# Description: Automatically evaluate inference outputs for multiple datasets from a list of files (Parallelized).
# Author: (your name)
# Date: 2025-08-12

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
set -euo pipefail

# --- Configuration ---
# Input directory containing the inference outputs
input_dir="/home/jianzhnie/llmtuner/llm/LLMEval/output/PCL-Reasoner"
output_dir="/home/jianzhnie/llmtuner/llm/LLMEval/output/omni-data"
# Evaluation output directory
reval_dir="${output_dir}/eval_score/"
# Python evaluation script path
eval_script="/home/jianzhnie/llmtuner/llm/LLMEval/llmeval/tasks/math_eval/eval.py"
# The task name for the evaluation script
task_name="math_opensource/aime24"

# --- Variables ---
input_key="problem"
label_key="answer"
max_workers=32       # Max workers for the *Python script* internal parallelism
timeout=20
max_parallel_jobs=32 # Maximum number of *Bash parallel processes*

# Create evaluation directory if it doesn't exist
mkdir -p "${reval_dir}"

# Function to evaluate a single file
evaluate_file() {
    local input_file="$1"
    local filename=$(basename -- "${input_file}")
    local cache_path="${reval_dir}/${filename}"
    # Log file for stdout and stderr (unified)
    # 移除 .jsonl 后缀，然后添加 .log
    local filename_base=${filename%.jsonl}  # 结果：infer_PCL-Reasoner_omin-math_shard_113_bz8
    
    local log_path="${reval_dir}/${filename_base}.log"
    
    echo "▶️ [$(date +%H:%M:%S)] Starting evaluation for: ${filename} (PID: $$)..."
    
    # Run the evaluation script with the dynamic paths
    # Redirect both stdout and stderr to the log file
    python "${eval_script}" \
        --input_path "${input_file}" \
        --cache_path "${cache_path}" \
        --task_name "${task_name}" \
        --input_key "${input_key}" \
        --label_key "${label_key}" \
        --max_workers "${max_workers}" \
        --timeout "${timeout}" \
        &> "${log_path}"  # Unified redirection to log file
    
    # Check the exit status of the python script
    if [[ $? -eq 0 ]]; then
        echo "✅ [$(date +%H:%M:%S)] Completed successfully: ${filename}. Output in ${log_path}"
    else
        echo "❌ [$(date +%H:%M:%S)] Failed evaluation for: ${filename}. Check ${log_path} for errors."
    fi
}
# Export the function so background processes can access it
export -f evaluate_file
export input_dir reval_dir eval_script task_name input_key label_key max_workers timeout

# --- Evaluate Each Task ---
echo "⚙️ Starting parallel evaluation with max ${max_parallel_jobs} jobs..."

# Use 'shopt -s nullglob' to ensure the loop doesn't run if no files are found.
shopt -s nullglob
file_list=("${input_dir}/infer_PCL-Reasoner_omin-math_shard_"*.jsonl)

if [ ${#file_list[@]} -eq 0 ]; then
    echo "⚠️ No files found matching the pattern: ${input_dir}/infer_PCL-Reasoner_omin-math_shard_*.jsonl"
    exit 0
fi

# Loop through all files
for input_file in "${file_list[@]}"; do
    
    # Check the current number of background jobs
    while true; do
        # Count the number of currently running background processes started by this script.
        # This is more robust than a simple counter.
        current_jobs=$(jobs -p | wc -l)
        
        if [[ $current_jobs -lt $max_parallel_jobs ]]; then
            break # Less than max allowed, proceed
        fi
        
        # Wait for any background job to complete (non-blocking wait)
        echo "⏳ [$(date +%H:%M:%S)] Max jobs (${max_parallel_jobs}) reached. Waiting for a slot..."
        wait -n || true # Use 'wait -n' (available in Bash 4.3+)
    done
    
    # Start evaluation in background
    # We pass the file path to the function
    evaluate_file "${input_file}" &
done

# Wait for all remaining jobs to complete
echo "✨ All files queued. Waiting for final jobs to finish..."
wait
echo "🎯 Evaluation completed successfully for all files!"