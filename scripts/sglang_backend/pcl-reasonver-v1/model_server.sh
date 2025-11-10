#!/bin/bash/

source set_env.sh

export DISABLE_L2_CACHE=1

model_path="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476"
model_name="PCL-Reasoner-V1"

num_gpus=8
max_model_len=32768
mem_fraction_static=0.7

python -m sglang.launch_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --mem-fraction-static $mem_fraction_static \
    --context-length $max_model_len  \
    --schedule-conservativeness 1.5 \
    --chunked-prefill-size 1024 \
    --cuda-graph-max-bs 96 \
    --max-prefill-tokens 2048 \
    --attention-backend ascend \
    --sampling-backend ascend \
    --tp-size 8 \
    --device npu \
    --port 8090

# --enable-fp32-lm-head
# dtype float16