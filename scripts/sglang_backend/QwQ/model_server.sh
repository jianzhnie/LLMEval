#!/bin/bash/

source set_env.sh

export DISABLE_L2_CACHE=1

hfhub="/home/jianzhnie/llmtuner/hfhub/models"
model_path="${hfhub}/Qwen/QwQ-32B"
model_name="Qwen/QwQ-32B"

num_gpus=8
max_model_len=32768
mem_fraction_static=0.8

python -m sglang.launch_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --mem-fraction-static $mem_fraction_static \
    --context-length $max_model_len  \
    --schedule-conservativeness 1.3 \
    --chunked-prefill-size 1024 \
    --cuda-graph-max-bs 128 \
    --max-prefill-tokens 2048 \
    --attention-backend ascend \
    --sampling-backend ascend \
    --device npu \
    --port 8090 \