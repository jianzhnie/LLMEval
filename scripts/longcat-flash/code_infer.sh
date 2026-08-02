#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Code Generation Inference (HumanEval / MBPP)
# =============================================================================
# 对接 vLLM 服务, 对代码生成 benchmark 运行推理.
#
# Usage:
#   # 默认 (humaneval, pass@1)
#   bash scripts/longcat-flash/code_infer.sh
#
#   # 只跑 humaneval
#   BENCHMARKS=humaneval bash scripts/longcat-flash/code_infer.sh
#
#   # 自定义 pass@N 采样数
#   N_SAMPLES=64 bash scripts/longcat-flash/code_infer.sh
#
#   # 自定义服务地址
#   BASE_URL=http://10.42.11.130:8200/v1 MODEL_NAME=longcat-flash bash scripts/longcat-flash/code_infer.sh
#
# Data preparation:
#   python scripts/data_process/prepare_code_benchmarks.py --benchmarks humaneval mbpp --output_dir ./data
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# =============================================================================
# 服务配置
# =============================================================================
BASE_URL="${BASE_URL:-http://127.0.0.1:8200/v1}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
API_KEY="${API_KEY:-EMPTY}"

# =============================================================================
# 推理参数
# =============================================================================
N_SAMPLES="${N_SAMPLES:-1}"               # pass@1 默认 1 次采样
MAX_WORKERS="${MAX_WORKERS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"         # pass@1 用贪婪解码
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-1024}"          # 代码生成一般较短
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-empty}"
TOOL_CHOICE="${TOOL_CHOICE:-none}"

# =============================================================================
# 输出目录
# =============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}}"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Benchmarks 注册
# =============================================================================
# 可用: humaneval mbpp humaneval_plus mbpp_plus
# 预设组:
#   ALL     = humaneval mbpp
#   HUMEVAL = humaneval humaneval_plus
BENCHMARKS="${BENCHMARKS:-ALL}"

case "$BENCHMARKS" in
    ALL)     BENCHMARKS="humaneval mbpp" ;;
    HUMEVAL) BENCHMARKS="humaneval humaneval_plus" ;;
esac

declare -A BENCHMARK_INPUT=(
    [humaneval]="./data/humaneval.jsonl"
    [mbpp]="./data/mbpp.jsonl"
    [humaneval_plus]="./data/humaneval_plus.jsonl"
    [mbpp_plus]="./data/mbpp_plus.jsonl"
)

# pass@1 为默认, 温度 0 (贪婪解码); pass@N 温度 0.2
declare -A BENCHMARK_SAMPLES=(
    [humaneval]="$N_SAMPLES"
    [mbpp]="$N_SAMPLES"
    [humaneval_plus]="$N_SAMPLES"
    [mbpp_plus]="$N_SAMPLES"
)

# =============================================================================
# 启动前摘要
# =============================================================================
echo "============================================"
echo "[INFO] LongCat-Flash-Chat — Code Inference"
echo "[INFO] Server:    $BASE_URL"
echo "[INFO] Model:     $MODEL_NAME"
echo "[INFO] Output:    $OUTPUT_DIR"
echo "[INFO] Workers:   $MAX_WORKERS | MaxTokens: $MAX_TOKENS"
echo "[INFO] Benchmarks:"
for task in $BENCHMARKS; do
    n="${BENCHMARK_SAMPLES[$task]:-$N_SAMPLES}"
    printf "        %-16s  n_sample=%-3s  temp=%-4s\n" "$task" "$n" "$TEMPERATURE"
done
echo "============================================"

# =============================================================================
# 前置检查
# =============================================================================
if ! curl -sf --max-time 5 "${BASE_URL}/models" > /dev/null 2>&1; then
    echo "[WARN] 服务 $BASE_URL 不可达, 稍后推理可能失败" >&2
fi

export OPENAI_API_KEY="$API_KEY"

# =============================================================================
# 推理函数
# =============================================================================
run_infer() {
    local task="$1"
    local input_file="$2"
    local output_file="$3"
    local n_samples="$4"
    local temperature="$5"

    echo "[START] $task (temp=$temperature, n=$n_samples) -> $output_file"

    python llmeval/vllm/online_server.py \
        --input_file "$input_file" \
        --input_key "prompt" \
        --task "$task" \
        --output_file "$output_file" \
        --base_url "$BASE_URL" \
        --model_name "$MODEL_NAME" \
        --n_samples "$n_samples" \
        --temperature "$temperature" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --max_workers "$MAX_WORKERS" \
        --max_retries "$MAX_RETRIES" \
        --request_timeout "$REQUEST_TIMEOUT" \
        --system_prompt_type "$SYSTEM_PROMPT_TYPE" \
        --tool_choice "$TOOL_CHOICE" 2>&1

    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[OK] $task 推理完成: $output_file"
    else
        echo "[FAIL] $task 推理失败 (exit=$rc)" >&2
    fi
    return $rc
}

# =============================================================================
# 执行推理
# =============================================================================
PARALLEL="${PARALLEL:-1}"
declare -a TASK_NAMES=()
declare -a TASK_PIDS=()

for task in $BENCHMARKS; do
    input_file="${BENCHMARK_INPUT[$task]:-}"
    if [[ -z "$input_file" ]]; then
        echo "[ERROR] 未知 benchmark: $task (可用: ${!BENCHMARK_INPUT[*]})" >&2
        continue
    fi
    if [[ ! -f "$input_file" ]]; then
        echo "[ERROR] 数据文件不存在: $input_file" >&2
        echo "[HINT]  运行数据准备: python scripts/data_process/prepare_code_benchmarks.py --benchmarks $task --output_dir ./data" >&2
        continue
    fi

    n_samples="${BENCHMARK_SAMPLES[$task]:-$N_SAMPLES}"
    output_file="${OUTPUT_DIR}/${task}_bz${n_samples}.jsonl"
    TASK_NAMES+=("$task")

    if [[ "$PARALLEL" == "1" ]]; then
        log_file="${OUTPUT_DIR}/${task}_infer.log"
        {
            echo "[$task] temp=$TEMPERATURE n=$n_samples"
            run_infer "$task" "$input_file" "$output_file" "$n_samples" "$TEMPERATURE" >> "$log_file" 2>&1
        } &
        TASK_PIDS+=($!)
    else
        run_infer "$task" "$input_file" "$output_file" "$n_samples" "$TEMPERATURE" || true
    fi
done

# 等待并行任务
if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[INFO] 等待 ${#TASK_PIDS[@]} 个并行任务完成..."
    for pid in "${TASK_PIDS[@]}"; do
        wait "$pid" || true
    done
fi

echo ""
echo "============================================"
echo "[INFO] Code inference 完成!"
echo "[INFO] 输出目录: $OUTPUT_DIR"
echo "============================================"
