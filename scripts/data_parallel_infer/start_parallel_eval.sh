#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
AUTO_INFER_SH="${AUTO_INFER_SH:-${SCRIPT_DIR}/auto_model_infer.sh}"
DEFAULT_NODE_LIST="${DEFAULT_NODE_LIST:-${PROJECT_ROOT}/available_nodes.txt}"
PROFILE="${PROFILE:-tp2}"

case "$PROFILE" in
    tp2)
        export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
        export INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-4}"
        export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
        export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
        export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-51200}"
        export N_SAMPLES="${N_SAMPLES:-8}"
        ;;
    tp8)
        export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
        export INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-1}"
        export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-32}"
        export SWAP_SPACE="${SWAP_SPACE:-0}"
        export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
        export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
        export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
        export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512000}"
        export N_SAMPLES="${N_SAMPLES:-4}"
        export EXTRA_ENGINE_ARGS="--cpu-offload-gb ${CPU_OFFLOAD_GB} --swap-space ${SWAP_SPACE} --dtype ${VLLM_DTYPE} ${EXTRA_ENGINE_ARGS:-}"
        ;;
    *)
        printf 'Unsupported PROFILE=%s; expected tp2 or tp8\n' "$PROFILE" >&2
        exit 1
        ;;
esac

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.9}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-${PROFILE}}"

export PROJECT_DIR="${PROJECT_DIR:-${PROJECT_ROOT}}"
export INFER_SCRIPT="${INFER_SCRIPT:-${PROJECT_DIR}/llmeval/inference/online.py}"
export SET_ENV_SCRIPT="${SET_ENV_SCRIPT:-${PROJECT_DIR}/set_env.sh}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${SERVED_MODEL_NAME}}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/data_parallel_logs/${SERVED_MODEL_NAME}}"
export DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/clone_datasets}"
export DATASET_GLOB="${DATASET_GLOB:-aime*}"
export INPUT_KEY="${INPUT_KEY:-prompt}"

export SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-amthinking}"
export MAX_WORKERS="${MAX_WORKERS:-128}"
export DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
export MAX_WAIT_TIME="${MAX_WAIT_TIME:-600}"
export HEALTH_PATH="${HEALTH_PATH:-/health}"
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3}"

NODE_LIST_FILE="${1:-$DEFAULT_NODE_LIST}"
if [[ ! -f "$AUTO_INFER_SH" ]]; then
    printf 'Auto inference script not found: %s\n' "$AUTO_INFER_SH" >&2
    exit 1
fi
if [[ ! -f "$NODE_LIST_FILE" ]]; then
    printf 'Node list not found: %s\n' "$NODE_LIST_FILE" >&2
    exit 1
fi

printf 'Starting distributed inference\n'
printf '  Profile: %s\n' "$PROFILE"
printf '  Nodes: %s\n' "$NODE_LIST_FILE"
printf '  Model: %s\n' "$MODEL_PATH"
printf '  Served name: %s\n' "$SERVED_MODEL_NAME"
printf '  TP/instances: %s/%s\n' "$TENSOR_PARALLEL_SIZE" "$INSTANCES_PER_NODE"
printf '  Output: %s\n' "$OUTPUT_DIR"

exec bash "$AUTO_INFER_SH" "$NODE_LIST_FILE"
