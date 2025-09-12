#!/bin/bash

set -euo pipefail

input_file=/home/jianzhnie/llmtuner/llm/QwQ/eval/output/am-thinking/eval_score/infer_qwen25_32B_math_top_30K_rl_verify_part_000_bz8.jsonl
output_file=output/CompassVerifier-3B/infer_qwen25_32B_math_top_30K_rl_verify_part_000_bz8.jsonl
model_name_or_path=/pcl_shared_dpc/hfhub/models/opencompass/CompassVerifier-3B

python /home/jianzhnie/llmtuner/llm/LLMEval/llmeval/vllm/compassverifier_offline_infer.py \
    --input_file $input_file \
    --output_file $output_file \
    --model_name_or_path $model_name_or_path \
    --trust_remote_code \
    --input_key prompt \
    --label_key answer \
    --response_key gen \
    --batch_size 8 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.9 \
    --temperature 0.0
