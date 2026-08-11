# LLMEval 支持的 Benchmark

## 数学推理任务

数学推理是评估大模型高级推理能力的核心维度。业内三大基础评测方向：**数学推理**、**竞赛编程**、**科学知识**。

AIME（美国数学邀请赛）和 HMMT（哈佛-MIT 数学锦标赛）作为极具挑战性的人类数学考试，已成为评估大模型推理能力的"试金石"。

### GSM8K
小学数学应用题，包含四则运算和简单逻辑推理。标准评测为 1-shot greedy decoding。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| 样本数 | 1,319 (test) |
| 答案格式 | `#### 数字` |
| 评分方式 | math-verify |

### MATH-500
竞赛数学精选 500 题，覆盖代数、几何、数论、组合等。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| 样本数 | 500 (test) |
| 答案格式 | LaTeX 表达式 |
| 评分方式 | math-verify |

### AIME 2024 / 2025 / 2026
美国数学邀请赛，每年 30 题，整数答案 0-999。需 pass@N 多次采样降方差。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [math-ai/aime24](https://huggingface.co/datasets/math-ai/aime24) / [aime25](https://huggingface.co/datasets/math-ai/aime25) / [aime26](https://huggingface.co/datasets/math-ai/aime26) |
| 样本数 | 各 30 (test) |
| 答案格式 | 整数 0-999 |
| 评分方式 | math-verify, pass@32 |

### HMMT 2025 (Feb)
哈佛-MIT 数学竞赛，难度大于 AIME。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |
| 样本数 | 30 |
| 答案格式 | 整数 / LaTeX |
| 评分方式 | math-verify |

### GPQA Diamond
Google-Proof Q&A，博士级科学推理（物理/化学/生物），diamond 为最难子集。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [lightonai/gpqa_diamond_multilingual](https://huggingface.co/datasets/lightonai/gpqa_diamond_multilingual) (en 子集) |
| 样本数 | 198 |
| 答案格式 | `\boxed{字母}` |
| 评分方式 | math-verify |

---

## 通用领域 Multiple-Choice

支持两种评测模式：
- **loglikelihood**（默认）：比较候选答案的对数似然，支持两种评分方式：
  - `first_token`（默认）：通过 Chat Completions 的首 token `top_logprobs`
    比较答案标记（A-J）。
  - `continuation`：计算完整候选答案的 token loglikelihood 总和；要求后端兼容
    `/completions`、`echo`、输入 token logprobs 和 `text_offset`。
- **generate**：生成文本 → 提取答案字母。兼容性好。

两种 loglikelihood 评分方法的原理、限制及适用场景见
[MC Loglikelihood 评分机制](mc_scoring.md)。

### MMLU
Massive Multitask Language Understanding，57 学科多选题。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 样本数 | ~14,000 (test, 57 subjects) |
| 选项 | 4 (A/B/C/D) |
| 推荐 n-shot | 5 |

### MMLU-Pro
MMLU 增强版，10 选项，去除了简单题。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| 样本数 | ~12,000 (test, 14 subjects) |
| 选项 | 10 (A-J) |
| 推荐 n-shot | 5 |

### C-Eval
中文综合学科评测，52 科目。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) |
| 样本数 | ~5,500 (test, 52 subjects) |
| 选项 | 4 (A/B/C/D)，中文题干 |
| 推荐 n-shot | 5 |

---

## 代码生成任务

代码生成评估使用沙箱执行 + pass@k 评分。模型生成代码后，在隔离子进程中执行测试用例。

### HumanEval
OpenAI 发布的 164 道 Python 函数生成题。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| 样本数 | 164 (test) |
| 答案格式 | assert + check() |
| 评分方式 | pass@1 沙箱执行 |

### HumanEval+
HumanEval 增强版，每个题目增加更多测试用例。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus) |
| 样本数 | 164 (test) |
| 答案格式 | assert + check() |
| 评分方式 | pass@1 沙箱执行 |

### MBPP
Google 发布的 500 道 Python 编程题（test split）。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (full) |
| 样本数 | 500 (test) |
| 答案格式 | assert 语句 |
| 评分方式 | pass@1 沙箱执行 |

### MBPP+
MBPP 增强版。

| 属性 | 值 |
|------|-----|
| HF 数据集 | [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) |
| 样本数 | 500 (test) |
| 答案格式 | assert 语句 |
| 评分方式 | pass@1 沙箱执行 |


---

## 汇总表

| Benchmark | 类型 | 样本数 | 评分 | HF |
|-----------|------|--------|------|-----|
| gsm8k | 数学 | 1,319 | math-verify | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| math500 | 数学 | 500 | math-verify | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| aime24 | 数学 | 30 | math-verify pass@32 | [math-ai/aime24](https://huggingface.co/datasets/math-ai/aime24) |
| aime25 | 数学 | 30 | math-verify pass@32 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| aime26 | 数学 | 30 | math-verify pass@32 | [math-ai/aime26](https://huggingface.co/datasets/math-ai/aime26) |
| hmmt25 | 数学 | 30 | math-verify | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |
| gpqa_diamond | 科学 | 198 | math-verify | [lightonai/gpqa_diamond_multilingual](https://huggingface.co/datasets/lightonai/gpqa_diamond_multilingual) |
| hle_full | 科学 | ~2,700 | math-verify | [cais/hle](https://huggingface.co/datasets/cais/hle) (gated, 需 HF_TOKEN) |
| mmlu | MC | ~14,000 | loglikelihood | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| mmlu_pro | MC | ~12,000 | loglikelihood | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| ceval | MC | ~5,500 | loglikelihood | [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) |
| humaneval | 代码 | 164 | pass@1 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| humaneval_plus | 代码 | 164 | pass@1 | [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus) |
| mbpp | 代码 | 500 | pass@1 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| mbpp_plus | 代码 | 500 | pass@1 | [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) |

## 数据准备

```bash
# 数学 benchmark
python scripts/data_process/prepare_math_benchmarks.py \
    --benchmarks gsm8k math500 aime24 aime25 aime26 hmmt25 gpqa_diamond

# 代码 benchmark
python scripts/data_process/prepare_code_benchmarks.py \
    --benchmarks humaneval mbpp

# MC benchmark
python scripts/data_process/prepare_mc_benchmarks.py \
    --benchmarks mmlu mmlu_pro ceval

# Few-shot dev 数据放在 data/few_shot/，格式同推理数据
```
