#!/bin/bash
set -euo pipefail


# === 数据集相关 ===
readonly PROJECT_DIR="/home/jianzhnie/llmtuner/llm/LLMEval"
readonly DATASET_DIR="${PROJECT_DIR}/data/"
readonly TEMP_DATASET_DIR="${DATASET_DIR}/aime"

# 要复制的本地数据集文件
readonly DATASET_FILES=(
    "${DATASET_DIR}/aime24.jsonl"
    "${DATASET_DIR}/aime25.jsonl"
)

readonly NUM_DUPLICATES=16      # 数据集复制份数

# 复制数据集多份
duplicate_datasets() {
    echo "🔁 正在准备数据集副本..."
    for dataset in "${DATASET_FILES[@]}"; do
        [[ ! -f "$dataset" ]] && { echo "❌ 错误: 数据集文件未找到：$dataset" >&2; exit 1; }
    done

    rm -rf "${TEMP_DATASET_DIR}"
    mkdir -p "${TEMP_DATASET_DIR}"

    for dataset in "${DATASET_FILES[@]}"; do
        local filename
        filename=$(basename "$dataset" .jsonl)
        local extension=".jsonl"

        echo "📁 正在复制 ${filename} x${NUM_DUPLICATES} 份"
        for ((i=1; i<=NUM_DUPLICATES; i++)); do
            cp "${dataset}" "${TEMP_DATASET_DIR}/${filename}_duplicate_${i}${extension}"
        done
    done
    echo "✅ 数据集副本创建完成。"
}

# =======================================================
#                  主流程
# =======================================================
main() {
    duplicate_datasets
}

# 运行主函数
main "$@"
