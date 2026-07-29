#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Online Inference 评测脚本
# =============================================================================
# 对接 run_vllm_long-context.sh 启动的 vLLM 服务，运行数学推理评测。
#
# Usage:
#   # 默认 (aime24 + aime25, 各 64 样本)
#   bash scripts/longcat-flash/online_infer.sh
#
#   # 自定义服务地址和模型
#   BASE_URL=http://10.42.11.130:8200/v1 MODEL_NAME=longcat-flash bash scripts/longcat-flash/online_infer.sh
#
#   # 只跑单个 benchmark
#   BENCHMARKS=aime24 bash scripts/longcat-flash/online_infer.sh
#
#   # 自定义采样数和并发
#   N_SAMPLES=128 MAX_WORKERS=128 bash scripts/longcat-flash/online_infer.sh
#
#   # 续跑 (断点续传): 直接重跑同一条命令, 已完成的 prompt 会自动跳过
#   bash scripts/longcat-flash/online_infer.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# =============================================================================
# 服务配置 (可通过环境变量覆盖)
# =============================================================================
BASE_URL="${BASE_URL:-http://127.0.0.1:8200/v1}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
API_KEY="${API_KEY:-EMPTY}"

# =============================================================================
# 推理参数
# =============================================================================
N_SAMPLES="${N_SAMPLES:-32}"            # 每题重复采样数 (aime pass@N 需要大样本)
MAX_WORKERS="${MAX_WORKERS:-32}"         # 客户端并发数
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-32768}"        # 长上下文推理上限 (给 chain-of-thought 留足空间)
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60000}" # 单请求超时 (秒), 长 CoT 可能很久
MAX_RETRIES="${MAX_RETRIES:-3}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-empty}"

# =============================================================================
# 输出目录 (按模型名隔离)
# =============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}}"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Benchmarks: "name|input_file|n_samples"
# name:      数据集标识 (用于输出文件名和 task 参数)
# input_file: JSONL 数据路径
# n_samples: 每题采样数 (空 = 用全局 N_SAMPLES)
# =============================================================================
BENCHMARKS="${BENCHMARKS:-aime25 aime24}"

declare -A BM_INPUT=(
    [aime25]="./data/aime25.jsonl"
    [aime24]="./data/aime24.jsonl"
)
declare -A BM_SAMPLES=(
    [aime25]="$N_SAMPLES"
    [aime24]="$N_SAMPLES"
)

# =============================================================================
# 启动前摘要
# =============================================================================
echo "============================================"
echo "[INFO] LongCat-Flash-Chat — Online Inference"
echo "[INFO] Server:   $BASE_URL"
echo "[INFO] Model:    $MODEL_NAME"
echo "[INFO] Benchmarks: $BENCHMARKS"
echo "[INFO] N_SAMPLES:  $N_SAMPLES"
echo "[INFO] MAX_WORKERS: $MAX_WORKERS"
echo "[INFO] TEMPERATURE: $TEMPERATURE"
echo "[INFO] MAX_TOKENS:  $MAX_TOKENS"
echo "[INFO] Output:   $OUTPUT_DIR"
echo "============================================"

# =============================================================================
# 前置检查
# =============================================================================
# 服务可达性
if ! curl -sf --max-time 5 "${BASE_URL}/models" > /dev/null 2>&1; then
    echo "[WARN] 服务 $BASE_URL 不可达, 稍后推理可能失败" >&2
fi

export OPENAI_API_KEY="$API_KEY"

# =============================================================================
# 运行推理: PARALLEL=1 并行跑所有 benchmark, 默认并行
# =============================================================================
PARALLEL="${PARALLEL:-1}"

run_infer() {
    local bm="$1"
    local input_file="$2"
    local output_file="$3"
    local n_samples="$4"

    echo ""
    echo "----------------------------------------"
    echo "[INFO] 开始推理: $bm"
    echo "[INFO] 输入:   $input_file"
    echo "[INFO] 输出:   $output_file"
    echo "[INFO] 采样数: $n_samples"
    echo "----------------------------------------"

    if python llmeval/vllm/online_server.py \
        --input_file "$input_file" \
        --input_key "prompt" \
        --output_file "$output_file" \
        --base_url "$BASE_URL" \
        --model_name "$MODEL_NAME" \
        --n_samples "$n_samples" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --max_workers "$MAX_WORKERS" \
        --max_retries "$MAX_RETRIES" \
        --request_timeout "$REQUEST_TIMEOUT" \
        --system_prompt_type "$SYSTEM_PROMPT_TYPE"; then
        echo "[OK] $bm 推理完成: $output_file"
        return 0
    else
        echo "[FAIL] $bm 推理失败 (exit=$?)" >&2
        return 1
    fi
}

# 收集任务列表
declare -a TASK_NAMES=()
declare -a TASK_PIDS=()
declare -A TASK_STATUS=()

for bm in $BENCHMARKS; do
    input_file="${BM_INPUT[$bm]:-}"
    if [[ -z "$input_file" ]]; then
        echo "[ERROR] 未知 benchmark: $bm (可用: ${!BM_INPUT[*]})" >&2
        continue
    fi

    n_samples="${BM_SAMPLES[$bm]:-$N_SAMPLES}"
    output_file="${OUTPUT_DIR}/${bm}_bz${n_samples}.jsonl"
    TASK_NAMES+=("$bm")

    if [[ "$PARALLEL" == "1" ]]; then
        # 并行: 所有 benchmark 同时启动，共享服务端并发
        run_infer "$bm" "$input_file" "$output_file" "$n_samples" &
        TASK_PIDS+=($!)
    else
        # 串行: 顺序执行
        run_infer "$bm" "$input_file" "$output_file" "$n_samples" || true
    fi
done

# 并行模式下等待所有后台任务
if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[INFO] 等待 ${#TASK_PIDS[@]} 个并行任务完成..."
    for i in "${!TASK_PIDS[@]}"; do
        pid="${TASK_PIDS[$i]}"
        bm="${TASK_NAMES[$i]}"
        if wait "$pid"; then
            TASK_STATUS[$bm]="OK"
        else
            TASK_STATUS[$bm]="FAIL"
        fi
    done
fi

# =============================================================================
# 结果摘要
# =============================================================================
echo ""
echo "============================================"
FAILED=()
for bm in "${TASK_NAMES[@]}"; do
    if [[ "${TASK_STATUS[$bm]:-}" == "FAIL" ]]; then
        FAILED+=("$bm")
    fi
done

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "🎉 全部推理任务完成!"
else
    echo "⚠️  失败: ${FAILED[*]}"
fi
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
