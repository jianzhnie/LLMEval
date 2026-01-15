#!/bin/bash/

source set_env_lmcache.sh

model_path="/home/jianzhnie/llmtuner/llm/LLMEval/work_dir/opg_32b_step_800"
model_name="PCL-Reasoner-v1"

num_gpus=8
max_model_len=131072  # ✅ 支持 128k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len  \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
    --disable-log-requests \
    --port 8090
