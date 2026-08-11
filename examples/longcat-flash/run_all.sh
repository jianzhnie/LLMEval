#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — 一键推理 + 评分
# =============================================================================
# 先跑推理 (online_infer.sh), 再跑评分 (get_score.sh), 最后汇总结果。
#
# Usage:
#   # 一键全流程 (准备数据 → 推理 → 评分)
#   bash scripts/longcat-flash/run_all.sh
#
#   # 只推理
#   STAGE=infer bash scripts/longcat-flash/run_all.sh
#
#   # 只评分 (已有推理结果)
#   STAGE=score bash scripts/longcat-flash/run_all.sh
#
#   # 快速验证 (gsm8k + math500, 样本少)
#   BENCHMARKS=QUICK bash scripts/longcat-flash/run_all.sh
#
#   # 自定义采样数
#   N_SAMPLES=64 bash scripts/longcat-flash/run_all.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROJECT_ROOT

cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source set_env.sh 2>/dev/null || true

# =============================================================================
# 阶段控制: prepare | infer | score | all
# =============================================================================
STAGE="${STAGE:-all}"

# =============================================================================
# 参数透传 (在线推理 + 评分子脚本共用)
# =============================================================================
export BENCHMARKS="${BENCHMARKS:-ALL}"
export N_SAMPLES="${N_SAMPLES:-32}"
export MAX_WORKERS="${MAX_WORKERS:-32}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-empty}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:8200/v1}"
export MODEL_NAME="${MODEL_NAME:-longcat-flash}"
export INPUT_DIR="${INPUT_DIR:-./output/${MODEL_NAME}}"
export OUTPUT_DIR="$INPUT_DIR"

echo "============================================"
echo "[INFO] LongCat-Flash-Chat — 一键评测"
echo "[INFO] STAGE:   $STAGE"
echo "[INFO] Bench:   $BENCHMARKS"
echo "[INFO] Server:  $BASE_URL ($MODEL_NAME)"
echo "[INFO] Output:  $OUTPUT_DIR"
echo "============================================"

# =============================================================================
# Step 0: 数据准备 (按需)
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "prepare" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "[STEP 1/3] 准备数据..."
    echo "----------------------------------------"

    # 展开预设组获取实际 benchmark 列表
    MISSING=()
    for task in $BENCHMARKS; do
        case "$task" in gsm8k|math500|hmmt25|gpqa_diamond|aime24|aime25|aime26) ;; *) continue ;; esac
        if [[ ! -f "./data/${task}.jsonl" ]]; then
            MISSING+=("$task")
        fi
    done

    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "[INFO] 缺少数据文件: ${MISSING[*]} — 自动下载..."
        python scripts/data_process/prepare_math_benchmarks.py \
            --benchmarks "${MISSING[@]}" \
            --output_dir ./data
    else
        echo "[INFO] 数据文件已就绪, 跳过下载"
    fi
fi

# =============================================================================
# Step 1: 推理
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "infer" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "[STEP 2/3] 推理..."
    echo "----------------------------------------"
    bash "${SCRIPT_DIR}/online_infer.sh"
fi

# =============================================================================
# Step 2: 评分
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "score" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "[STEP 3/3] 评分..."
    echo "----------------------------------------"
    bash "${SCRIPT_DIR}/get_score.sh"
fi

# =============================================================================
# 结果汇总
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "score" ]]; then
    echo ""
    echo "============================================"
    echo "[SUMMARY] 评分汇总"
    echo "============================================"

    EVAL_DIR="${INPUT_DIR}/eval_score"
    if [[ -d "$EVAL_DIR" ]]; then
        for f in "$EVAL_DIR"/*_score.txt; do
            if [[ -f "$f" ]]; then
                task=$(basename "$f" | sed 's/_bz.*//')
                score=$(grep -oE '[0-9]+\.[0-9]+%?' "$f" | tail -1 || echo "N/A")
                printf "  %-12s  %s\n" "$task" "$score"
            fi
        done
    fi
    echo ""
    echo "详细结果: $EVAL_DIR/"
    echo "🎉 一键评测完成!"
fi
