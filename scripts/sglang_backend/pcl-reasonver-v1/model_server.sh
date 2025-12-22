#!/bin/bash/

source set_env_sglang.sh

model_path="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476"
model_name="PCL-Reasoner-v1-sglang"

num_gpus=8
max_model_len=131072
mem_fraction_static=0.8

python3 -m sglang.launch_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --mem-fraction-static $mem_fraction_static \
    --context-length $max_model_len  \
    --schedule-conservativeness 2 \
    --cuda-graph-max-bs 64 \
    --disable-radix-cache \
    --attention-backend ascend \
    --sampling-backend vllm_ascend \
    --device npu \
    --port 8090
