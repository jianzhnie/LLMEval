#!/bin/bash

# 设置默认参数
home_dir="${HOME_DIR:-/home/jianzhnie/llmtuner/llm/LLMEval}"
data_dir="${DATA_DIR:-/home/jianzhnie/llmtuner/llm/LLMEval/output/opg_32b_step_800_bz32}"
n_samples="${N_SAMPLES:-2}"
num_prompts="${NUM_PROMPTS:-30}"
num_clone_files="${NUM_CLONE_FILES:-16}"

# 计算总行数
lines_per_file=$((num_prompts * n_samples ))
samples_per_prompt=$((n_samples * num_clone_files))
total_lines=$((num_prompts * samples_per_prompt))

# 输出配置信息
echo "----------------------------------------"
echo "数据目录: $data_dir"
echo "每个Prompt采样次数: $n_samples"
echo "AIME测试集Prompt数: $num_prompts"
echo "每个文件复制次数: $num_clone_files"
echo "实际每个Prompt采样次数: $n_samples * $num_clone_files = $samples_per_prompt"
echo "每个输出文件的打满数据: $n_samples * $num_prompts = $lines_per_file"
echo "合并后AIME数据文件大小: $num_prompts * $samples_per_prompt = $total_lines"
echo "----------------------------------------"

# 检查目录是否存在
if [[ ! -d "$data_dir" ]]; then
    echo "错误: 数据目录 $data_dir 不存在"
    exit 1
fi

cd "$data_dir" || exit 1

# 清理失败文件
rm -f *failed*

# 合并函数
merge_files() {
    local pattern=$1
    local output_file=$2
    local expected_lines=$3

    echo "正在合并 $pattern 文件到 $output_file (期望行数: $expected_lines)"

    # 收集匹配的文件
    local matching_files=()
    local match_count=0

    while IFS= read -r -d '' file; do
        local file_lines=$(wc -l < "$file")
        if [[ $file_lines -eq $expected_lines ]]; then
            matching_files+=("$file")
            ((match_count++))
            echo "  ✓ $file ($file_lines 行)"
        else
            echo "  ✗ $file ($file_lines 行, 期望 $expected_lines 行)"
        fi
    done < <(find . -maxdepth 1 -name "$pattern" -type f -print0)

    echo "找到 $match_count 个匹配文件"

    if [[ ${#matching_files[@]} -eq 0 ]]; then
        echo "警告: 没有找到符合条件的 $pattern 文件"
        return 1
    fi

    # 合并文件（只输出文件路径给 xargs）
    printf '%s\0' "${matching_files[@]}" | xargs -0 cat > "$output_file"

    # 检查输出文件
    if [[ -s "$output_file" ]]; then
        local output_lines=$(wc -l < "$output_file")
        echo "成功创建 $output_file ($output_lines 行)"
        return 0
    else
        echo "警告: $output_file 为空或未创建"
        return 1
    fi
}

# 合并 aime24 和 aime25 文件
merge_files "*aime24*" "aime24_bz${samples_per_prompt}.jsonl" "$total_lines"
merge_files "*aime25*" "aime25_bz${samples_per_prompt}.jsonl" "$total_lines"

# 返回原目录
cd "$home_dir" || exit 1

echo "合并完成!"
