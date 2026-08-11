# LLM 推理模型评测系统

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

</div>

## 概述

LLMEval 是一个用于评测大型语言模型（LLM）的综合评估系统，覆盖数学推理、代码生成和通用知识 benchmark。支持在线（API）、离线（本地推理）和 MC（loglikelihood/generate）三种模式。

### 主要特性

- **多推理后端**：vLLM (GPU/NPU)、SGLang、OpenAI API
- **三种评测模式**：在线生成、离线本地推理、MC loglikelihood 对比
- **14 个 Benchmark**：AIME 2024/2025/2026、MATH-500、GSM8K、GPQA-Diamond、HMMT-25、MMLU、MMLU-Pro、C-Eval、HumanEval、MBPP
- **代码评估**: HumanEval / MBPP 沙箱执行 + pass@k 评分
- **MC 评分**：答案 token loglikelihood + acc/acc_norm/exact_match、few-shot 去重
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
vllm serve \
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
bash examples/longcat-flash/run_all.sh           # 全流程
BENCHMARKS=QUICK bash examples/longcat-flash/run_all.sh   # 快速验证
BENCHMARKS=HARD bash examples/longcat-flash/run_all.sh    # 高难度
```

**MC benchmark** — 答案 token loglikelihood 对比 + acc/acc_norm 评分：

```bash
N_SHOT=5 bash examples/longcat-flash/mc_infer.sh   # 5-shot 推理
bash examples/longcat-flash/mc_score.sh              # 评分
```

**代码 benchmark** — 代码生成 + 沙箱执行 pass@k summary：

```bash
bash examples/longcat-flash/code_infer.sh            # HumanEval + MBPP 推理
bash examples/longcat-flash/code_score.sh            # 评分

# Pass@64（64 次采样，temperature 0.2）
N_SAMPLES=64 TEMPERATURE=0.2 bash examples/longcat-flash/code_infer.sh
```

或分步执行：
```bash
bash examples/longcat-flash/online_infer.sh   # 数学推理
bash examples/longcat-flash/get_score.sh      # 数学评分
```

### 3. 手动推理（CLI 模式）

在线模式：
```bash
python -m llmeval.inference.online \
    --input_file "./data/aime24.jsonl" \
    --output_file "./output/aime24.jsonl" \
    --base_url "http://127.0.0.1:8090/v1" \
    --model_name "QwQ-32B" \
    --n_samples 64 \
    --max_workers 8
```

离线模式（无需启动服务器）：
```bash
python -m llmeval.inference.offline \
    --input_file "./data/aime24.jsonl" \
    --output_file "./output/aime24.jsonl" \
    --model_name_or_path "Qwen/QwQ-32B" \
    --tensor_parallel_size 8 \
    --n_samples 64
```

### 4. 手动评分

```bash
# 数学评分
python -m llmeval.evaluator \
    --input_path "./output/aime24.jsonl" \
    --result_path "./output/aime24_scores.json" \
    --task_name "math_opensource/aime24" \
    --max_workers 16

# 代码评分（pass@k summary，执行超时 5 秒）
python -m llmeval.evaluator \
    --input_path "./output/humaneval.jsonl" \
    --result_path "./output/humaneval_scores.json" \
    --task_name "code_opensource/humaneval" \
    --max_workers 32 \
    --exec_timeout 5.0 \
    --allow_unsafe_code
```

## 详细使用说明

### 推理参数

在线和离线模式通用参数：

| 参数             | 默认值   | 说明                     |
|----------------|---------|------------------------|
| `--n_samples`   | 1       | 每道题采样次数              |
| `--temperature` | 0.6     | 采样温度                   |
| `--top_p`       | 0.95    | 核采样参数                  |
| `--max_completion_tokens` | 32768 | 最大生成 completion token 数 |
| `--seed`        | 0       | 生成随机种子                |

在线模式特有参数：

| 参数             | 默认值   | 说明                     |
|----------------|---------|------------------------|
| `--base_url`    | `https://api.openai.com/v1` | API 服务器地址 |
| `--model_name`  | `gpt-4o` | API 使用的模型名称          |
| `--max_workers` | 128     | 并发请求线程数              |
| `--extra_body`  | `{}`     | 服务商扩展请求字段的 JSON 对象 |

参数含义及 Python、cURL、结构化输出、工具调用和流式输出示例，详见
[OpenAI API 常用参数](docs/openai_api.md)。

离线模式特有参数：

| 参数                      | 默认值  | 说明                      |
|--------------------------|--------|--------------------------|
| `--model_name_or_path`   | 必填    | 本地模型路径或 HuggingFace ID |
| `--tensor_parallel_size` | 1      | 张量并行 GPU 数量           |
| `--gpu_memory_utilization`| 0.9   | GPU 显存使用比例            |
| `--batch_size`           | 128    | 推理批次大小                |
| `--top_k`                | 40     | vLLM Top-k 采样参数         |
| `--enable_thinking`      | false  | vLLM chat template 思考选项 |
| `--skip_special_tokens`  | true   | 删除生成结果中的特殊 token  |
| `--repetition_penalty`   | 1.0    | vLLM 重复惩罚               |

### 支持的评测任务

| 类别 | 任务名 | 评分方式 |
|------|--------|----------|
| 数学 | `math_opensource/aime24` `aime25` `aime26` `gsm8k` `math500` `math` `hmmt25` | math-verify |
| 科学 | `math_opensource/gpqa_diamond` `hle_full` | math-verify |
| 代码 | `code_opensource/humaneval` `mbpp` `humaneval_plus` `mbpp_plus` | pass@k 沙箱执行 |
| MC | `mc_opensource/mmlu` `mmlu_pro` `ceval` | 答案 token loglikelihood |

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
vllm serve \
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
│   ├── evaluator.py       # 评分编排器 (math / mc / code)
│   ├── inference/         # 推理引擎 (online / offline / mc)
│   ├── tasks/
│   │   ├── math_eval/     # 数学评分 (math-verify)
│   │   ├── code_eval/     # 代码评分 (沙箱执行 pass@k)
│   │   └── mc_eval/       # MC 评分 (loglikelihood/generate)
│   └── utils/             # 工具函数
├── examples/
│   ├── longcat-flash/     # 一键评测脚本
│   │   ├── run_all.sh     #   数学全流程
│   │   ├── online_infer.sh #  数学推理
│   │   ├── get_score.sh   #  数学评分
│   │   ├── code_infer.sh  #  代码推理
│   │   ├── code_score.sh  #  代码评分
│   │   ├── mc_infer.sh    #  MC 推理
│   │   └── mc_score.sh    #  MC 评分
│   └── QwQ/               # QwQ-32B 脚本
├── scripts/
│   ├── data_parallel_infer/ # 多机数据并行推理
│   └── data_process/      # 数据准备
├── tests/                 # 231 个测试用例
├── data/                  # benchmark 数据集
└── docs/                  # 文档
```

## 许可证

本项目采用 Apache License 2.0 许可证。
