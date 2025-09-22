#!/bin/bash

set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

input_file=/home/jianzhnie/llmtuner/llm/LLMEval/output/am-thinking-0528/infer_qwen25_32B_math_top_30K_rl_merged.json
output_file=./output/CompassVerifier-3B/infer_qwen25_32B_math_top_30K_rl_merged_comass-verifier3b.jsonl
model_name_or_path=/home/jianzhnie/llmtuner/hfhub/models/OpenCompass/CompassVerifier-3B

python /home/jianzhnie/llmtuner/llm/LLMEval/llmeval/vllm/verifier_offline_infer.py \
    --input_file $input_file \
    --output_file $output_file \
    --model_name_or_path $model_name_or_path \
    --trust_remote_code \
    --verifier_prompt_type compassverify_prompt \
    --input_key prompt \
    --label_key answer \
    --response_key gen \
    --keep_origin_data \
    --batch_size 128 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 4 \
    --max_model_len 32768 \
    --max_tokens 2048 \
    --temperature 0.0
