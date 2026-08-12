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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROJECT_ROOT

cd "$PROJECT_ROOT"

# =============================================================================
# 服务配置 (可通过环境变量覆盖)
# =============================================================================
BASE_URL="${BASE_URL:-http://127.0.0.1:8200/v1}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
API_KEY="${API_KEY:-EMPTY}"

# =============================================================================
# 推理参数 (全局默认值, 可被 benchmark 级覆盖)
# =============================================================================
N_SAMPLES="${N_SAMPLES:-32}"              # pass@N 默认采样数
MAX_WORKERS="${MAX_WORKERS:-32}"
TEMPERATURE="${TEMPERATURE:-0.6}"          # pass@N 默认温度
TOP_P="${TOP_P:-0.95}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-32768}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-empty}"

# =============================================================================
# 输出目录 (按模型名隔离)
# =============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}}"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Benchmarks 注册
# =============================================================================
# 可用 benchmark: gsm8k math500 hmmt25 gpqa_diamond aime24 aime25 aime26
# 预设组:
#   ALL     = 全部 7 个数据集
#   HARD    = aime24 aime25 aime26 hmmt25 gpqa_diamond (高难度)
#   QUICK   = gsm8k math500 (快速验证, 样本少)
# 自定义: BENCHMARKS="gsm8k gpqa_diamond" bash scripts/longcat-flash/online_infer.sh
# =============================================================================
BENCHMARKS="${BENCHMARKS:-ALL}"

# 预设组展开
case "$BENCHMARKS" in
    ALL)   BENCHMARKS="gsm8k math500 hmmt25 gpqa_diamond aime24 aime25 aime26" ;;
    HARD)  BENCHMARKS="aime24 aime25 aime26 hmmt25 gpqa_diamond" ;;
    QUICK) BENCHMARKS="gsm8k math500" ;;
esac

declare -A BENCHMARK_INPUT=(
    [gsm8k]="./data/gsm8k.jsonl"
    [math500]="./data/math500.jsonl"
    [hmmt25]="./data/hmmt25.jsonl"
    [gpqa_diamond]="./data/gpqa_diamond.jsonl"
    [aime24]="./data/aime24.jsonl"
    [aime25]="./data/aime25.jsonl"
    [aime26]="./data/aime26.jsonl"
)
# 每 benchmark 采样数 + 温度 (空 = 用全局默认)
# pass@1: temp=0 (贪婪解码, 确定性); pass@N: temp=0.6 (多样性采样)
declare -A BENCHMARK_SAMPLES=(
    [gsm8k]="1"
    [math500]="1"
    [hmmt25]="$N_SAMPLES"
    [gpqa_diamond]="$N_SAMPLES"
    [aime24]="$N_SAMPLES"
    [aime25]="$N_SAMPLES"
    [aime26]="$N_SAMPLES"
)
declare -A BENCHMARK_TEMP=(
    [gsm8k]="0"
    [math500]="0"
)

# =============================================================================
# 启动前摘要
# =============================================================================
echo "============================================"
echo "[INFO] LongCat-Flash-Chat — Online Inference"
echo "[INFO] Server:    $BASE_URL"
echo "[INFO] Model:     $MODEL_NAME"
echo "[INFO] Output:    $OUTPUT_DIR"
echo "[INFO] Workers:   $MAX_WORKERS | MaxCompletionTokens: $MAX_COMPLETION_TOKENS"
echo "[INFO] Benchmarks:"
for task in $BENCHMARKS; do
    n="${BENCHMARK_SAMPLES[$task]:-$N_SAMPLES}"
    t="${BENCHMARK_TEMP[$task]:-$TEMPERATURE}"
    printf "        %-14s  n_samples=%-3s  temp=%-4s\n" "$task" "$n" "$t"
done
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
    local task="$1"
    local input_file="$2"
    local output_file="$3"
    local n_samples="$4"
    local temperature="$5"
    local log_file="$6"  # 可选: 独立日志文件

    # 摘要始终输出到主日志, 详细进度输出到独立日志
    echo "[START] $task (temp=$temperature, n=$n_samples) → $output_file"

    if [[ -n "$log_file" ]]; then
        # 并行模式: 独立日志
        {
            echo "[$task] temp=$temperature n=$n_samples"
            python -m llmeval.inference.online \
                --input_file "$input_file" \
                --input_key "prompt" \
                --output_file "$output_file" \
                --base_url "$BASE_URL" \
                --model_name "$MODEL_NAME" \
                --n_samples "$n_samples" \
                --temperature "$temperature" \
                --top_p "$TOP_P" \
                --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
                --max_workers "$MAX_WORKERS" \
                --max_retries "$MAX_RETRIES" \
                --request_timeout "$REQUEST_TIMEOUT" \
                --system_prompt_type "$SYSTEM_PROMPT_TYPE" \
                2>&1
        } >> "$log_file"
        local rc=$?
    else
        # 串行模式: 直接输出
        python -m llmeval.inference.online \
            --input_file "$input_file" \
            --input_key "prompt" \
            --output_file "$output_file" \
            --base_url "$BASE_URL" \
            --model_name "$MODEL_NAME" \
            --n_samples "$n_samples" \
            --temperature "$temperature" \
            --top_p "$TOP_P" \
            --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
            --max_workers "$MAX_WORKERS" \
            --max_retries "$MAX_RETRIES" \
            --request_timeout "$REQUEST_TIMEOUT" \
            --system_prompt_type "$SYSTEM_PROMPT_TYPE"
        local rc=$?
    fi

    if [[ $rc -eq 0 ]]; then
        echo "[OK] $task 推理完成: $output_file"
        return 0
    else
        echo "[FAIL] $task 推理失败 (exit=$rc)" >&2
        return 1
    fi
}

# 收集任务列表
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

    n_samples="${BENCHMARK_SAMPLES[$task]:-$N_SAMPLES}"
    temperature="${BENCHMARK_TEMP[$task]:-$TEMPERATURE}"
    output_file="${OUTPUT_DIR}/${task}_bz${n_samples}.jsonl"
    TASK_NAMES+=("$task")

    if [[ "$PARALLEL" == "1" ]]; then
        # 并行: 独立日志避免串扰
        log_file="${OUTPUT_DIR}/${task}_infer.log"
        run_infer "$task" "$input_file" "$output_file" "$n_samples" "$temperature" "$log_file" &
        TASK_PIDS+=($!)
    else
        # 串行: 直接输出
        if run_infer "$task" "$input_file" "$output_file" "$n_samples" "$temperature" ""; then
            TASK_STATUS[$task]="OK"
        else
            TASK_STATUS[$task]="FAIL"
            task_failed=1
        fi
    fi
done

# 并行模式下等待所有后台任务
if [[ "$PARALLEL" == "1" && ${#TASK_PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "[INFO] 等待 ${#TASK_PIDS[@]} 个并行任务完成..."
    for i in "${!TASK_PIDS[@]}"; do
        pid="${TASK_PIDS[$i]}"
        task="${TASK_NAMES[$i]}"
        if wait "$pid"; then
            TASK_STATUS[$task]="OK"
        else
            TASK_STATUS[$task]="FAIL"
            task_failed=1
        fi
    done
fi

# =============================================================================
# 结果摘要
# =============================================================================
echo ""
echo "============================================"
FAILED=()
for task in "${TASK_NAMES[@]}"; do
    if [[ "${TASK_STATUS[$task]:-}" == "FAIL" ]]; then
        FAILED+=("$task")
    fi
done

if [[ $task_failed -eq 0 ]]; then
    echo "🎉 全部推理任务完成!"
else
    echo "⚠️  失败: ${FAILED[*]}"
fi
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
exit "$task_failed"
