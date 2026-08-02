# LLM 推理模型评测系统

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

</div>

## 概述

LLMEval 是一个用于评测大型语言模型（LLM）的综合评估系统，覆盖数学推理和通用知识 benchmark。支持在线（API）、离线（本地推理）和 MC（loglikelihood/generate）三种模式。

### 主要特性

- **多推理后端**：vLLM (GPU/NPU)、SGLang、OpenAI API
- **三种评测模式**：在线生成、离线本地推理、MC loglikelihood 对比
- **10 个 Benchmark**：AIME 2024/2025/2026、MATH-500、GSM8K、GPQA-Diamond、HMMT-25、MMLU、MMLU-Pro、C-Eval
- **lm-eval 对齐**：acc/acc_norm/exact_match、few-shot 去重
- **一键评测**：Shell 脚本端到端推理 → 评分
- **断点续评**：自动恢复中断的评测任务

## 评测结果

我们成功在 AIME 2024 和 AIME 2025 基准测试上复现了多个开源模型的结果。

对于 AIME24 这类仅包含 30 道题的基准测试，多次采样至关重要，因为随机采样会引入较大方差。以下结果均采用每题 64 次采样取平均值，以确保评估稳定性。

### DeepSeek-R1-Distill-Qwen-32B

| 数据集   | LLMEval | 官方报告 |
|:-------:|:-------:|:-------:|
| AIME24  | 70.625  |  72.6   |
| AIME25  | 55.052  |  59.0   |
| MATH-500|  93.2   |  94.3   |

### DeepSeek-R1-Distill-Qwen-7B

| 数据集   | LLMEval | 官方报告 |
|:-------:|:-------:|:-------:|
| AIME24  |  51.77  |  55.5   |
| AIME25  |  36.77  |  39.2   |

### DeepSeek-R1-Distill-Qwen-1.5B

| 数据集   | LLMEval | 官方报告 |
|:-------:|:-------:|:-------:|
| AIME24  |  27.92  |  28.9   |
| AIME25  |  23.44  |  21.4   |

### QwQ-32B

| 数据集   | LLMEval | 官方报告 |
|:-------:|:-------:|:-------:|
| AIME24  |  78.65  |  79.5   |
| AIME25  |  67.22  |  69.5   |

### Skywork-OR1-32B

| 数据集   | LLMEval | 官方报告 |
|:-------:|:-------:|:-------:|
| AIME24  |  81.25  |  82.2   |
| AIME25  |  72.66  |  73.3   |

### OpenThinker3-7B

| 数据集   | LLMEval | 官方报告 |
|:-------:|:-------:|:-------:|
| AIME24  |  70.41  |  69.0   |
| AIME25  |  59.16  |  53.3   |

## 安装

### 环境要求

| 软件     | 版本      |
|---------|----------|
| Python  | >= 3.10  |
| torch   | >= 2.0   |

华为昇腾 NPU 用户额外需要：
- CANN == 8.1.RC1
- torch_npu == 2.5.1

### 从源码安装

```bash
git clone https://github.com/jianzhnie/LLMEval.git
# 或使用 gitee 镜像：git clone https://gitee.com/jianzhnie/LLMEval.git
cd LLMEval
pip install -e .
```

### 安装 vLLM（可选）

GPU 用户：
```bash
pip install vllm>=0.7.0
```

华为昇腾 NPU 用户：
```bash
# 安装 vllm
git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install -e .

# 安装 vllm-ascend
git clone -b v0.7.3.post1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
```

## 快速开始

### 1. 启动推理服务器（在线模式）

使用 vLLM：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/QwQ-32B \
    --served-model-name QwQ-32B \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --port 8090
```

使用 SGLang（支持数据并行）：
```bash
python -m sglang_router.launch_server \
    --model-path Qwen/QwQ-32B \
    --dp-size 4 \
    --port 30000
```

### 2. 一键评测（推荐）

**数学 benchmark** — 生成式推理 + math-verify 评分：

```bash
bash scripts/longcat-flash/run_all.sh           # 全流程
BENCHMARKS=QUICK bash scripts/longcat-flash/run_all.sh   # 快速验证
BENCHMARKS=HARD bash scripts/longcat-flash/run_all.sh    # 高难度
```

**MC benchmark** — loglikelihood 选项对比 + acc/acc_norm 评分：

```bash
N_SHOT=5 bash scripts/longcat-flash/mc_infer.sh   # 5-shot 推理
bash scripts/longcat-flash/mc_score.sh              # 评分
```

或分步执行：
```bash
bash scripts/longcat-flash/online_infer.sh   # 数学推理
bash scripts/longcat-flash/get_score.sh      # 数学评分
```

### 3. 手动推理（CLI 模式）

在线模式：
```bash
python ./llmeval/inference/online.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "./output/aime24.jsonl" \
    --base_url "http://127.0.0.1:8090/v1" \
    --model_name "QwQ-32B" \
    --n_samples 64 \
    --max_workers 8
```

离线模式（无需启动服务器）：
```bash
python llmeval/inference/offline.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "./output/aime24.jsonl" \
    --model_name_or_path "Qwen/QwQ-32B" \
    --tensor_parallel_size 8 \
    --n_samples 64
```

### 4. 手动评分

```bash
python ./llmeval/evaluator.py \
    --input_path "./output/aime24.jsonl" \
    --cache_path "./output/aime24_scores.jsonl" \
    --task_name "math_opensource/aime24" \
    --max_workers 16
```

## 详细使用说明

### 推理参数

在线和离线模式通用参数：

| 参数             | 默认值   | 说明                     |
|----------------|---------|------------------------|
| `--n_samples`   | 1       | 每道题采样次数              |
| `--temperature` | 0.6     | 采样温度                   |
| `--top_p`       | 0.95    | 核采样参数                  |
| `--top_k`       | 40      | Top-k 采样参数             |
| `--max_tokens`  | 32768   | 最大生成 token 数          |

在线模式特有参数：

| 参数             | 默认值   | 说明                     |
|----------------|---------|------------------------|
| `--base_url`    | 必填     | API 服务器地址             |
| `--model_name`  | 必填     | API 使用的模型名称          |
| `--max_workers` | 128     | 并发请求线程数              |

离线模式特有参数：

| 参数                      | 默认值  | 说明                      |
|--------------------------|--------|--------------------------|
| `--model_name_or_path`   | 必填    | 本地模型路径或 HuggingFace ID |
| `--tensor_parallel_size` | 1      | 张量并行 GPU 数量           |
| `--gpu_memory_utilization`| 0.9   | GPU 显存使用比例            |
| `--batch_size`           | 128    | 推理批次大小                |

### 支持的评测任务

| 类别 | 任务名 | 评分方式 |
|------|--------|----------|
| 数学 | `math_opensource/aime24` `aime25` `aime26` `gsm8k` `math500` `hmmt25` | math-verify |
| 科学 | `math_opensource/gpqa_diamond` | math-verify |
| MC | `mc_opensource/mmlu` `mmlu_pro` `ceval` | loglikelihood |

详见 [docs/benchmark.md](docs/benchmark.md)。

### 断点续评

如果推理过程中断，直接重新运行相同命令即可。脚本会自动：
1. 读取已有输出文件
2. 统计每道题已完成采样数
3. 从断点继续评测

### 上下文长度扩展（YaRN）

处理超过 32K token 的长文本时，使用 RoPE 缩放技术：

**vLLM：**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --rope-scaling '{"rope_type":"yarn","factor":4.0}' \
    --max-model-len 131072
```

**SGLang：**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0}}' \
    --context-length 131072
```

## 项目结构

```
LLMEval/
├── llmeval/
│   ├── vllm/              # 推理引擎
│   ├── tasks/
│   │   ├── math_eval/     # 数学评分 (math-verify)
│   │   └── mc_eval/       # MC 评分 (loglikelihood/generate)
│   └── utils/             # 工具函数
├── scripts/
│   ├── longcat-flash/     # 一键评测脚本
│   │   ├── run_all.sh     #   数学全流程
│   │   ├── online_infer.sh #  数学推理
│   │   ├── get_score.sh   #  数学评分
│   │   ├── mc_infer.sh    #   MC 推理
│   │   └── mc_score.sh    #   MC 评分
│   └── data_process/      # 数据准备
├── tests/                 # 130 测试用例
├── data/                  # 10 个 benchmark, 7000+ 题目
└── docs/benchmark.md      # Benchmark 参考文档
```

## 许可证

本项目采用 MIT 许可证。
