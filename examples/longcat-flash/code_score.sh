#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Code Scoring (pass@k via code execution sandbox)
# =============================================================================
# 对 code_infer.sh 产出的推理结果运行代码执行 + pass@k 评分.
#
# Usage:
#   # 默认 (humaneval + mbpp, 使用 inference 输出目录)
#   bash scripts/longcat-flash/code_score.sh
#
#   # 自定义输入目录
#   INPUT_DIR=./output/longcat-flash bash scripts/longcat-flash/code_score.sh
#
#   # 只评 humaneval
#   BENCHMARKS=humaneval bash scripts/longcat-flash/code_score.sh
#
#   # 自定义执行超时和并发
#   EXEC_TIMEOUT=10.0 MAX_WORKERS=16 bash scripts/longcat-flash/code_score.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source set_env.sh 2>/dev/null || true

# =============================================================================
# 配置
# =============================================================================
INPUT_DIR="${INPUT_DIR:-./output/longcat-flash}"
N_SAMPLES="${N_SAMPLES:-1}"
MAX_WORKERS="${MAX_WORKERS:-32}"     # 代码执行并行度 (受 CPU 限制)
TIMEOUT="${TIMEOUT:-30}"             # 池级超时 (秒)
EXEC_TIMEOUT="${EXEC_TIMEOUT:-5.0}"  # 单条代码执行超时 (秒)

EVAL_DIR="${EVAL_DIR:-${INPUT_DIR}/eval_score}"
mkdir -p "$EVAL_DIR"

# =============================================================================
# Benchmarks 注册
# =============================================================================
BENCHMARKS="${BENCHMARKS:-ALL}"
PARALLEL="${PARALLEL:-1}"

case "$BENCHMARKS" in
    ALL)     BENCHMARKS="humaneval mbpp" ;;
    HUMEVAL) BENCHMARKS="humaneval humaneval_plus" ;;
esac

# task_name 映射 (对应 CodeEvalArguments)
declare -A TASK_NAME=(
    [humaneval]="code_opensource/humaneval"
    [mbpp]="code_opensource/mbpp"
    [humaneval_plus]="code_opensource/humaneval_plus"
    [mbpp_plus]="code_opensource/mbpp_plus"
)

declare -A BENCHMARK_SAMPLES=(
    [humaneval]="$N_SAMPLES"
    [mbpp]="$N_SAMPLES"
    [humaneval_plus]="$N_SAMPLES"
    [mbpp_plus]="$N_SAMPLES"
)

# =============================================================================
# 评分函数
# =============================================================================
run_eval() {
    local task="$1"
    local input_file="$2"
    local result_file="$3"
    local log_file="$4"
    local task_name="$5"

    echo "[INFO] 开始评分: $task ($task_name)"
    echo "[INFO] 输入:     $input_file"
    echo "[INFO] 结果:     $result_file"
    echo "[INFO] 日志:     $log_file"
    echo "[INFO] exec_timeout: ${EXEC_TIMEOUT}s"

    if python -m llmeval.evaluator \
        --input_path "$input_file" \
        --result_path "$result_file" \
        --task_name "$task_name" \
        --max_workers "$MAX_WORKERS" \
        --timeout "$TIMEOUT" \
        --exec_timeout "$EXEC_TIMEOUT" \
        --allow_unsafe_code \
        > "$log_file" 2>&1; then
        echo "[OK] $task 评分完成"
        # 从结果文件中提取 Pass@1
        if grep -q "Pass@1" "$log_file" 2>/dev/null; then
            local score
            score=$(grep "Pass@1" "$log_file" | tail -1)
            echo "[SCORE] $score"
        fi
        return 0
    else
        echo "[FAIL] $task 评分失败 (exit=$?)" >&2
        return 1
    fi
}

# =============================================================================
# 启动摘要
# =============================================================================
echo "============================================"
echo "[INFO] LongCat-Flash-Chat — Code Scoring (pass@k)"
echo "[INFO] INPUT_DIR:    $INPUT_DIR"
echo "[INFO] EVAL_DIR:     $EVAL_DIR"
echo "[INFO] Benchmarks:   $BENCHMARKS"
echo "[INFO] N_SAMPLES:    $N_SAMPLES"
echo "[INFO] MAX_WORKERS:  $MAX_WORKERS"
echo "[INFO] EXEC_TIMEOUT: ${EXEC_TIMEOUT}s"
echo "============================================"

# =============================================================================
# 收集并执行任务
# =============================================================================
declare -a TASK_NAMES=()
declare -a TASK_PIDS=()
declare -A TASK_STATUS=()

for task in $BENCHMARKS; do
    task_name="${TASK_NAME[$task]:-}"
    if [[ -z "$task_name" ]]; then
        echo "[ERROR] 未知 benchmark: $task (可用: ${!TASK_NAME[*]})" >&2
        continue
    fi

    n_samples="${BENCHMARK_SAMPLES[$task]:-$N_SAMPLES}"
    input_file="${INPUT_DIR}/${task}_bz${n_samples}.jsonl"
    result_file="${EVAL_DIR}/${task}_bz${n_samples}.json"
    log_file="${EVAL_DIR}/${task}_bz${n_samples}_score.txt"

    if [[ ! -f "$input_file" ]]; then
        echo "[ERROR] 输入文件不存在: $input_file" >&2
        echo "[HINT]  请先运行 code_infer.sh 生成推理结果" >&2
        continue
    fi

    TASK_NAMES+=("$task")

    if [[ "$PARALLEL" == "1" ]]; then
        run_eval "$task" "$input_file" "$result_file" "$log_file" "$task_name" &
        TASK_PIDS+=($!)
    else
        run_eval "$task" "$input_file" "$result_file" "$log_file" "$task_name" || true
    fi
done

# 并行等待
if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[INFO] 等待 ${#TASK_PIDS[@]} 个评分任务完成..."
    for i in "${!TASK_PIDS[@]}"; do
        pid="${TASK_PIDS[$i]}"
        task="${TASK_NAMES[$i]}"
        if wait "$pid"; then
            TASK_STATUS[$task]="OK"
        else
            TASK_STATUS[$task]="FAIL"
        fi
    done
fi

# =============================================================================
# 结果摘要
# =============================================================================
echo ""
echo "============================================"
echo "[SUMMARY] Code Scoring Results"
echo "============================================"
FAILED=()
for task in "${TASK_NAMES[@]}"; do
    if [[ "${TASK_STATUS[$task]:-}" == "FAIL" ]]; then
        FAILED+=("$task")
    else
        n_samples="${BENCHMARK_SAMPLES[$task]:-$N_SAMPLES}"
        result_file="${EVAL_DIR}/${task}_bz${n_samples}_score.txt"
        if [[ -f "$result_file" ]]; then
            score=$(grep -oE 'Pass@1: [0-9]+\.[0-9]+%' "$result_file" | tail -1 || echo "N/A")
            echo "[RESULT] $task: $score"
        fi
    fi
done

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "🎯 全部评分完成!"
else
    echo "⚠️  失败: ${FAILED[*]}"
fi
echo "评分结果: $EVAL_DIR/"
echo "============================================"
