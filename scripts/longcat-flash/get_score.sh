#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — 评分脚本 (math-verify)
# =============================================================================
# 对 online_infer.sh 产出的推理结果运行 math-verify 打分。
#
# Usage:
#   # 默认 (aime24 + aime25, 使用 inference 输出目录)
#   bash scripts/longcat-flash/get_score.sh
#
#   # 自定义输入目录和采样数
#   INPUT_DIR=./output/longcat-flash N_SAMPLES=32 bash scripts/longcat-flash/get_score.sh
#
#   # 只评单个 benchmark
#   BENCHMARKS=aime24 bash scripts/longcat-flash/get_score.sh
#
#   # 并行评分 (默认开启)
#   PARALLEL=1 MAX_WORKERS=32 bash scripts/longcat-flash/get_score.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source set_env.sh 2>/dev/null || true

# =============================================================================
# 配置 (可通过环境变量覆盖)
# =============================================================================
INPUT_DIR="${INPUT_DIR:-./output/longcat-flash}"
N_SAMPLES="${N_SAMPLES:-32}"
MAX_WORKERS="${MAX_WORKERS:-32}"    # math-verify 并行度 (CPU 密集, 不宜过高)
TIMEOUT="${TIMEOUT:-30}"            # 单题验证超时 (秒)

# 评分结果输出到 INPUT_DIR 下的 eval_score 子目录
EVAL_DIR="${EVAL_DIR:-${INPUT_DIR}/eval_score}"
mkdir -p "$EVAL_DIR"

# =============================================================================
# Benchmarks 注册
# =============================================================================
BENCHMARKS="${BENCHMARKS:-aime25 aime24}"
PARALLEL="${PARALLEL:-1}"           # 默认并行评分

declare -A TASK_NAME=(
    [aime25]="math_opensource/aime25"
    [aime24]="math_opensource/aime24"
)

# =============================================================================
# 评分函数
# =============================================================================
run_eval() {
    local bm="$1"
    local input_file="$2"
    local cache_file="$3"
    local result_file="$4"
    local task_name="$5"

    echo "[INFO] 开始评分: $bm ($task_name)"
    echo "[INFO] 输入:   $input_file"
    echo "[INFO] 缓存:   $cache_file"
    echo "[INFO] 结果:   $result_file"

    if python llmeval/tasks/math_eval/eval.py \
        --input_path "$input_file" \
        --cache_path "$cache_file" \
        --task_name "$task_name" \
        --max_workers "$MAX_WORKERS" \
        --timeout "$TIMEOUT" \
        > "$result_file" 2>&1; then
        echo "[OK] $bm 评分完成"
        # 从结果文件中提取准确率
        if grep -q "accuracy\|Accuracy" "$result_file" 2>/dev/null; then
            echo "[SCORE] $(grep -E 'accuracy|Accuracy|acc' "$result_file" | tail -1)"
        fi
        return 0
    else
        echo "[FAIL] $bm 评分失败 (exit=$?)" >&2
        return 1
    fi
}

# =============================================================================
# 启动摘要
# =============================================================================
echo "============================================"
echo "[INFO] LongCat-Flash-Chat — 评分 (math-verify)"
echo "[INFO] INPUT_DIR:   $INPUT_DIR"
echo "[INFO] EVAL_DIR:    $EVAL_DIR"
echo "[INFO] Benchmarks:  $BENCHMARKS"
echo "[INFO] N_SAMPLES:   $N_SAMPLES"
echo "[INFO] MAX_WORKERS: $MAX_WORKERS"
echo "[INFO] PARALLEL:    $PARALLEL"
echo "============================================"

# =============================================================================
# 收集并执行任务
# =============================================================================
declare -a TASK_NAMES=()
declare -a TASK_PIDS=()
declare -A TASK_STATUS=()

for bm in $BENCHMARKS; do
    task_name="${TASK_NAME[$bm]:-}"
    if [[ -z "$task_name" ]]; then
        echo "[ERROR] 未知 benchmark: $bm (可用: ${!TASK_NAME[*]})" >&2
        continue
    fi

    input_file="${INPUT_DIR}/${bm}_bz${N_SAMPLES}.jsonl"
    cache_file="${EVAL_DIR}/${bm}_bz${N_SAMPLES}.jsonl"
    result_file="${EVAL_DIR}/${bm}_bz${N_SAMPLES}_score.txt"

    if [[ ! -f "$input_file" ]]; then
        echo "[ERROR] 输入文件不存在: $input_file" >&2
        continue
    fi

    TASK_NAMES+=("$bm")

    if [[ "$PARALLEL" == "1" ]]; then
        run_eval "$bm" "$input_file" "$cache_file" "$result_file" "$task_name" &
        TASK_PIDS+=($!)
    else
        run_eval "$bm" "$input_file" "$cache_file" "$result_file" "$task_name" || true
    fi
done

# 并行等待
if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[INFO] 等待 ${#TASK_PIDS[@]} 个评分任务完成..."
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
    else
        result_file="${EVAL_DIR}/${bm}_bz${N_SAMPLES}_score.txt"
        if [[ -f "$result_file" ]]; then
            score=$(grep -oE '[0-9]+\.[0-9]+%?' "$result_file" | tail -1 || echo "N/A")
            echo "[RESULT] $bm: $score"
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
