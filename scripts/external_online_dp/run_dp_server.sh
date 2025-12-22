export HCCL_IF_IP=10.16.201.224
export GLOO_SOCKET_IFNAME=enp66s0f5
export TP_SOCKET_IFNAME=enp66s0f5
export HCCL_SOCKET_IFNAME=enp66s0f5
export VLLM_LOGGING_LEVEL="info"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_DETERMINISTIC=True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-start-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --seed 1024 \
    --served-model-name qwen3 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
