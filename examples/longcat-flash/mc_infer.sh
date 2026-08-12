#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Multiple-Choice Inference
# =============================================================================
# 支持两种模式 (aligned with lm-evaluation-harness):
#   loglikelihood: 比较每个选项的对数似然, 选最高分 (更准确, 默认)
#   generate:      生成文本 → 提取答案字母 (更简单, 兼容性好)
#
# Usage:
#   bash scripts/longcat-flash/mc_infer.sh                    # loglikelihood
#   LOGLIKELIHOOD_MODE=continuation bash scripts/longcat-flash/mc_infer.sh
#   MC_MODE=generate bash scripts/longcat-flash/mc_infer.sh   # generate
#   BENCHMARKS=ceval bash scripts/longcat-flash/mc_infer.sh   # single benchmark
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROJECT_ROOT
cd "$PROJECT_ROOT"

# =============================================================================
# 配置
# =============================================================================
BASE_URL="${BASE_URL:-http://127.0.0.1:8200/v1}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
API_KEY="${API_KEY:-EMPTY}"
MC_MODE="${MC_MODE:-loglikelihood}"       # loglikelihood | generate
LOGLIKELIHOOD_MODE="${LOGLIKELIHOOD_MODE:-first_token}" # first_token | continuation
MAX_WORKERS="${MAX_WORKERS:-64}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-empty}"
N_SHOT="${N_SHOT:-0}"                    # 0=zero-shot, 5=five-shot
FEW_SHOT_FILE="${FEW_SHOT_FILE:-}"        # dev file for few-shot
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}}"
mkdir -p "$OUTPUT_DIR"

BENCHMARKS="${BENCHMARKS:-ALL}"
case "$BENCHMARKS" in
    ALL)  BENCHMARKS="mmlu mmlu_pro ceval" ;;
esac

declare -A BENCHMARK_INPUT=(
    [mmlu]="./data/mmlu.jsonl"
    [mmlu_pro]="./data/mmlu_pro.jsonl"
    [ceval]="./data/ceval.jsonl"
)

export OPENAI_API_KEY="$API_KEY"

echo "============================================"
echo "[INFO] MC Inference — Mode: $MC_MODE"
echo "[INFO] Loglikelihood scoring: $LOGLIKELIHOOD_MODE"
echo "[INFO] Server: $BASE_URL ($MODEL_NAME)"
echo "[INFO] Benchmarks: $BENCHMARKS"
echo "============================================"

if ! curl -sf --max-time 5 "${BASE_URL}/models" > /dev/null 2>&1; then
    echo "[WARN] 服务 $BASE_URL 不可达" >&2
fi

# =============================================================================
# 推理
# =============================================================================
PARALLEL="${PARALLEL:-1}"
declare -a TASK_NAMES=()
declare -a TASK_PIDS=()
declare -A TASK_STATUS=()
task_failed=0

for task in $BENCHMARKS; do
    input_file="${BENCHMARK_INPUT[$task]:-}"
    if [[ -z "$input_file" ]]; then
        echo "[ERROR] 未知 benchmark: $task (可用: ${!BENCHMARK_INPUT[*]})" >&2
        task_failed=1
        continue
    fi
    if [[ ! -f "$input_file" ]]; then
        echo "[ERROR] 数据文件不存在: $input_file" >&2
        echo "[INFO] 下载: python scripts/data_process/prepare_mc_benchmarks.py --benchmarks $task --output_dir ./data"
        task_failed=1
        continue
    fi

    output_file="${OUTPUT_DIR}/${task}_bz1.jsonl"
    log_file="${OUTPUT_DIR}/${task}_mc_infer.log"
    TASK_NAMES+=("$task")

    echo "[START] $task (mode=$MC_MODE) → $output_file"

    if [[ "$PARALLEL" == "1" ]]; then
        {
            python -m llmeval.inference.mc \
                --input_file "$input_file" \
                --output_file "$output_file" \
                --base_url "$BASE_URL" \
                --model_name "$MODEL_NAME" \
                --mode "$MC_MODE" \
                --loglikelihood_mode "$LOGLIKELIHOOD_MODE" \
                --max_workers "$MAX_WORKERS" \
                --request_timeout "$REQUEST_TIMEOUT" \
                --max_retries "$MAX_RETRIES" \
                --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
                --temperature "$TEMPERATURE" \
                --system_prompt_type "$SYSTEM_PROMPT_TYPE" \
                --n_shot "$N_SHOT" \
                --few_shot_file "$FEW_SHOT_FILE" 2>&1
        } >> "$log_file" &
        TASK_PIDS+=($!)
    else
        if python -m llmeval.inference.mc \
            --input_file "$input_file" \
            --output_file "$output_file" \
            --base_url "$BASE_URL" \
            --model_name "$MODEL_NAME" \
            --mode "$MC_MODE" \
            --loglikelihood_mode "$LOGLIKELIHOOD_MODE" \
            --max_workers "$MAX_WORKERS" \
            --request_timeout "$REQUEST_TIMEOUT" \
            --max_retries "$MAX_RETRIES" \
            --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
            --temperature "$TEMPERATURE" \
            --system_prompt_type "$SYSTEM_PROMPT_TYPE" \
            --n_shot "$N_SHOT" \
            --few_shot_file "$FEW_SHOT_FILE"; then
            TASK_STATUS[$task]="OK"
            echo "[OK] $task"
        else
            TASK_STATUS[$task]="FAIL"
            task_failed=1
            echo "[FAIL] $task" >&2
        fi
    fi
done

if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    for i in "${!TASK_PIDS[@]}"; do
        if wait "${TASK_PIDS[$i]}"; then TASK_STATUS[${TASK_NAMES[$i]}]="OK"
        else TASK_STATUS[${TASK_NAMES[$i]}]="FAIL"; task_failed=1; fi
    done
fi

echo ""
for task in "${TASK_NAMES[@]}"; do
    echo "[${TASK_STATUS[$task]:-UNKNOWN}] $task"
done
if [[ $task_failed -eq 0 ]]; then
    echo "🎉 MC 推理完成!"
else
    echo "[ERROR] MC 推理存在失败任务" >&2
fi
exit "$task_failed"
