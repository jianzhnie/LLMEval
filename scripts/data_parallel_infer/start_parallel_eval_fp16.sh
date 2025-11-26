#!/usr/bin/env bash
set -euo pipefail

# Absolute paths
AUTO_INFER_SH="/home/jianzhnie/llmtuner/llm/LLMEval/scripts/data_parallel_infer/auto_model_infer_tp8_fp16.sh"
DEFAULT_NODE_LIST="/home/jianzhnie/llmtuner/llm/LLMEval/node_list.txt"

# Override below as needed (or export before running)
export SSH_USER="${SSH_USER:-jianzhnie}"

# Model/engine
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.9}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PCL-Reasoner-FP16}"
export N_SAMPLES="${N_SAMPLES:-8}"

# Project
export PROJECT_DIR="${PROJECT_DIR:-/home/jianzhnie/llmtuner/llm/LLMEval}"
export INFER_SCRIPT="${INFER_SCRIPT:-${PROJECT_DIR}/llmeval/vllm/online_server.py}"
export SET_ENV_SCRIPT="${SET_ENV_SCRIPT:-${PROJECT_DIR}/set_env.sh}"

# IO
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jianzhnie/llmtuner/llm/LLMEval/output}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${SERVED_MODEL_NAME}}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/data_paprallel_logs/${SERVED_MODEL_NAME}}"

# Dataset
export DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/clone_datasets}"
export DATASET_GLOB="${DATASET_GLOB:-aime*}"
export INPUT_KEY="${INPUT_KEY:-prompt}"                            # 输入字段键名

# Client concurrency
export SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-amthinking}"
export MAX_WORKERS="${MAX_WORKERS:-64}"
export MAX_JOBS="${MAX_JOBS:-128}"

# Server
export DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
export DISABLE_STATE_DUMP="${DISABLE_STATE_DUMP:-1}"
export API_WORKERS="${API_WORKERS:-1}"
export MAX_WAIT_TIME="${MAX_WAIT_TIME:-600}"
export HEALTH_PATH="${HEALTH_PATH:-/health}"
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3}"

# Args: optional node list file
NODE_LIST_FILE="${1:-$DEFAULT_NODE_LIST}"

# Validations
if [[ ! -f "$AUTO_INFER_SH" ]]; then
  echo "Error: auto infer script not found: $AUTO_INFER_SH" >&2
  exit 1
fi
if [[ ! -f "$NODE_LIST_FILE" ]]; then
  echo "Error: node list file not found: $NODE_LIST_FILE" >&2
  exit 1
fi

# Show key settings
echo "Starting distributed inference:"
echo "  Nodes file: $NODE_LIST_FILE"
echo "  Model: ${MODEL_PATH}"
echo "  Served name: ${SERVED_MODEL_NAME}"
echo "  GPUs per node (TP): ${NUM_GPUS}"
echo "  Concurrency: MAX_NUM_SEQS=${MAX_NUM_SEQS}, MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Logs dir: ${LOG_DIR}"

# Launch
bash "$AUTO_INFER_SH" "$NODE_LIST_FILE"
