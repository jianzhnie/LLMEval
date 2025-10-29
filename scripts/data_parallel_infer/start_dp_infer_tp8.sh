#!/usr/bin/env bash
set -euo pipefail

# =======================================================
# 分布式 vLLM 推理启动脚本
# =======================================================
#
# 功能描述：
#   1. 配置并启动分布式 vLLM 推理任务
#   2. 设置必要的环境变量
#   3. 验证配置并调用 auto_model_infer_tp8.sh
#
# 使用方法：
#   ./start_dp_infer_tp8.sh [NODE_LIST_FILE]
#
# 作者：LLM Eval Team
# 版本：1.1
# 更新日期：2025
# =======================================================

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# 统一的错误处理函数
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg"
    exit "$exit_code"
}

# 显示使用帮助
usage() {
    cat << EOF
用法: $0 [NODE_LIST_FILE]

启动分布式 vLLM 推理任务。

参数:
  NODE_LIST_FILE    节点列表文件 (默认: $DEFAULT_NODE_LIST)

环境变量:
  SSH_USER                    SSH 用户名 (默认: jianzhnie)
  MODEL_PATH                  模型路径
  NUM_GPUS                    每节点GPU数量 (默认: 8)
  MEMORY_UTILIZATION          GPU内存利用率 (默认: 0.9)
  MAX_MODEL_LEN               最大模型长度 (默认: 65536)
  MAX_NUM_SEQS                最大并发序列数 (默认: 1024)
  MAX_NUM_BATCHED_TOKENS      最大批处理token数 (默认: 512000)
  SERVED_MODEL_NAME           服务模型名称 (默认: PCL-Reasoner)
  N_SAMPLES                   采样数 (默认: 8)
  PROJECT_DIR                 项目目录
  OUTPUT_ROOT                 输出根目录
  OUTPUT_DIR                  输出目录
  LOG_DIR                     日志目录
  DATASET_DIR                 数据集目录
  DATASET_GLOB                数据集文件匹配模式
  INPUT_KEY                   输入字段键名 (默认: problem)
  SYSTEM_PROMPT_TYPE          系统提示类型 (默认: amthinking)
  MAX_WORKERS                 最大工作线程数 (默认: 128)
  MAX_WAIT_TIME               最大等待时间 (默认: 600)
  HEALTH_PATH                 健康检查路径 (默认: /health)
  HEALTH_TIMEOUT              健康检查超时 (默认: 3)

示例:
  $0
  $0 ./custom_nodes.txt
  SSH_USER=root MODEL_PATH=/path/to/model $0
EOF
    exit 1
}

# 检查依赖文件是否存在
check_dependencies() {
    local deps=("$AUTO_INFER_SH" "$NODE_LIST_FILE")
    local missing=()

    for dep in "${deps[@]}"; do
        if [[ ! -f "$dep" ]]; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "缺少以下依赖文件:"
        for file in "${missing[@]}"; do
            log_error "  - $file"
        done
        exit 1
    fi
}

# 验证关键目录是否存在
validate_directories() {
    local dirs=("$PROJECT_DIR" "$DATASET_DIR")
    local missing=()

    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            missing+=("$dir")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "缺少以下目录:"
        for dir in "${missing[@]}"; do
            log_error "  - $dir"
        done
        exit 1
    fi
}

# Absolute paths
AUTO_INFER_SH="/home/jianzhnie/llmtuner/llm/LLMEval/scripts/data_parallel_infer/auto_model_infer_tp8.sh"
DEFAULT_NODE_LIST="/home/jianzhnie/llmtuner/llm/LLMEval/available_nodes.txt"

# Override below as needed (or export before running)
export SSH_USER="${SSH_USER:-jianzhnie}"

# Model/engine
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-0.5B}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.9}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PCL-Reasoner}"
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
export DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/Omni-MATH}"
export DATASET_GLOB="${DATASET_GLOB:-omin-math_shard*}"
export INPUT_KEY="${INPUT_KEY:-problem}"                            # 输入字段键名

# Client concurrency
export SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-amthinking}"
export MAX_WORKERS="${MAX_WORKERS:-128}"

# Server
export MAX_WAIT_TIME="${MAX_WAIT_TIME:-600}"
export HEALTH_PATH="${HEALTH_PATH:-/health}"
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3}"

# 处理命令行参数
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
fi

# Args: optional node list file
NODE_LIST_FILE="${1:-$DEFAULT_NODE_LIST}"

# Validations
log_info "正在验证依赖文件..."
check_dependencies

log_info "正在验证目录..."
validate_directories

# Show key settings
log_info "启动分布式推理任务:"
log_info "  节点文件: $NODE_LIST_FILE"
log_info "  模型路径: ${MODEL_PATH}"
log_info "  推理脚本: ${INFER_SCRIPT}"
log_info "  服务名称: ${SERVED_MODEL_NAME}"
log_info "  每节点GPU/NPU数(TP): ${NUM_GPUS}"
log_info "  并发设置: MAX_NUM_SEQS=${MAX_NUM_SEQS}, MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
log_info "  输出目录: ${OUTPUT_DIR}"
log_info "  日志目录: ${LOG_DIR}"
log_info "  数据集目录: ${DATASET_DIR}"
log_info "  数据集匹配模式: ${DATASET_GLOB}"
log_info "  输入字段键名: ${INPUT_KEY}"
log_info "  System Prompt Type: ${SYSTEM_PROMPT_TYPE}"
log_info "  最大工作线程数: ${MAX_WORKERS}"

# Launch
log_info "正在启动推理任务..."
log_info "执行命令: bash $AUTO_INFER_SH $NODE_LIST_FILE"
bash "$AUTO_INFER_SH" "$NODE_LIST_FILE"