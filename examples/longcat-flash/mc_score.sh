#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Multiple-Choice Scoring (MMLU, C-Eval, etc.)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROJECT_ROOT
cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source set_env.sh 2>/dev/null || true

INPUT_DIR="${INPUT_DIR:-./output/longcat-flash}"
EVAL_DIR="${EVAL_DIR:-${INPUT_DIR}/eval_score}"
mkdir -p "$EVAL_DIR"

BENCHMARKS="${BENCHMARKS:-ALL}"
MAX_WORKERS="${MAX_WORKERS:-32}"
TIMEOUT="${TIMEOUT:-10}"

case "$BENCHMARKS" in
    ALL)  BENCHMARKS="mmlu mmlu_pro ceval" ;;
esac

declare -A TASK_NAME=(
    [mmlu]="mc_opensource/mmlu"
    [mmlu_pro]="mc_opensource/mmlu_pro"
    [ceval]="mc_opensource/ceval"
)

echo "============================================"
echo "[INFO] MC Scoring — Benchmarks: $BENCHMARKS"
echo "============================================"

PARALLEL="${PARALLEL:-1}"
declare -a TASK_NAMES=()
declare -a TASK_PIDS=()
task_failed=0

for task in $BENCHMARKS; do
    task_name="${TASK_NAME[$task]:-}"
    if [[ -z "$task_name" ]]; then
        echo "[ERROR] 未知: $task" >&2; task_failed=1; continue
    fi

    input_file="${INPUT_DIR}/${task}_bz1.jsonl"
    result_file="${EVAL_DIR}/${task}_bz1.json"
    log_file="${EVAL_DIR}/${task}_bz1_score.txt"

    if [[ ! -f "$input_file" ]]; then
        echo "[ERROR] 输入文件不存在: $input_file" >&2; task_failed=1; continue
    fi

    TASK_NAMES+=("$task")
    echo "[START] $task ($task_name)"

    if [[ "$PARALLEL" == "1" ]]; then
        {
            python -m llmeval.evaluator \
                --input_path "$input_file" \
                --result_path "$result_file" \
                --task_name "$task_name" \
                --max_workers "$MAX_WORKERS" \
                --timeout "$TIMEOUT" > "$log_file" 2>&1
        } &
        TASK_PIDS+=($!)
    else
        if python -m llmeval.evaluator \
            --input_path "$input_file" \
            --result_path "$result_file" \
            --task_name "$task_name" \
            --max_workers "$MAX_WORKERS" \
            --timeout "$TIMEOUT" > "$log_file" 2>&1; then
            echo "[OK] $task"
        else
            echo "[FAIL] $task" >&2
            task_failed=1
        fi
    fi
done

if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    for i in "${!TASK_PIDS[@]}"; do
        if wait "${TASK_PIDS[$i]}"; then
            echo "[OK] ${TASK_NAMES[$i]}"
        else
            echo "[FAIL] ${TASK_NAMES[$i]}" >&2
            task_failed=1
        fi
    done
fi

# 结果汇总
echo ""
echo "============================================"
for task in "${TASK_NAMES[@]}"; do
    rf="${EVAL_DIR}/${task}_bz1_score.txt"
    if [[ -f "$rf" ]]; then
        score=$(grep -oE '[0-9]+\.[0-9]+%?' "$rf" | tail -1 || echo "N/A")
        printf "  %-12s  %s\n" "$task" "$score"
    fi
done
if [[ $task_failed -eq 0 ]]; then
    echo "🎯 MC 评分完成!"
else
    echo "[ERROR] MC 评分存在失败任务" >&2
fi
exit "$task_failed"
