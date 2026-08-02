# LLM Evaluation

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

</div>

## Overview

LLMEval is a comprehensive evaluation system for assessing Large Language Models (LLMs) on mathematical reasoning and general knowledge benchmarks. It supports both online (API-based) and offline (local inference) modes with built-in answer verification and multiple-choice evaluation.

### Key Features

- **Multiple Inference Backends**: Support for vLLM (GPU/NPU) and SGLang
- **Flexible Evaluation Modes**: Online (API), offline (local), and MC (loglikelihood/generate)
- **Rich Benchmark Coverage**: 14 benchmarks — AIME 2024/2025/2026, MATH-500, GSM8K, GPQA-Diamond, HMMT-25, HumanEval, MBPP, MMLU, MMLU-Pro, C-Eval
- **lm-eval Aligned**: Loglikelihood MC scoring with acc/acc_norm/exact_match, few-shot dedup
- **One-Click Pipeline**: Shell scripts for end-to-end inference → scoring
- **Resume Capability**: Automatically resume interrupted evaluations
- **Verification Support**: Built-in answer extraction and correctness verification

## Results

We have successfully reproduced various open-source model results on the AIME 2024 & AIME 2025 benchmarks.

For benchmarks like AIME24, which contains only 30 problems, it is crucial to sample multiple responses as this can introduce high variance across repeated runs. All results below use 64 samples per problem to ensure stability.

### DeepSeek-R1-Distill-Qwen-32B

| Dataset | LLMEval | Official Report |
|:-------:|:-------:|:---------------:|
| AIME24  | 70.625  |      72.6       |
| AIME25  | 55.052  |      59.0       |
| MATH-500|  93.2   |      94.3       |

### QwQ-32B

| Dataset | LLMEval | Official Report |
|:-------:|:-------:|:---------------:|
| AIME24  |  78.65  |       79.5      |
| AIME25  |  67.22  |       69.5      |

### Skywork-OR1-32B

| Dataset | LLMEval | Official Report |
|:-------:|:-------:|:---------------:|
| AIME24  |  81.25  |       82.2      |
| AIME25  |  72.66  |       73.3      |

### OpenThinker3-7B

| Dataset | LLMEval | Official Report |
|:-------:|:-------:|:---------------:|
| AIME24  |  70.41  |       69.0      |
| AIME25  |  59.16  |       53.3      |

## Installation

### Requirements

| Software | Version |
|----------|---------|
| Python   | >= 3.10 |
| torch    | >= 2.0  |

For Huawei Ascend NPU users:
- CANN >= 8.1.RC1
- torch_npu >= 2.5.1
- vllm-ascend >= 0.7.3.post1

### Install from Source

```bash
git clone https://github.com/jianzhnie/LLMEval.git
# Or use gitee mirror: git clone https://gitee.com/jianzhnie/LLMEval.git
cd LLMEval
pip install -e .
```

### Install vLLM (Optional)

For GPU users:
```bash
pip install vllm>=0.7.0
```

For Huawei Ascend NPU users:
```bash
# Install vllm
git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install -e .

# Install vllm-ascend
git clone -b v0.7.3.post1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
```

## Quick Start

### Step 1: Start vLLM Server (Online Mode)

First, start the vLLM server with the following command:

```bash
model_path="Qwen/QwQ-32B"  # or model to the path where the model is located
model_name="Qwen/QwQ-32B"

num_gpus=8
max_model_len=32768  # ✅ 支持 32k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len  \
    --enforce-eager \
    --port 8090
```

Adjust the `tensor_parallel_size` parameter based on your available devices. Please refer to the [script](./scripts/model_server.sh) for more details.

Optional, Start SGLang server/router.Since the evaluation could takes days, we also suggest using SGLang with data parallelism to accelerate the evaluation. Refer to [SGLang documentation](https://docs.sglang.ai/router/router.html) for more details.

```bash
# Use router to support better data parallelism
python -m sglang_router.launch_server --model-path Qwen/QwQ-32B --dp-size 4 --host=0.0.0.0 --port=30000
```

Adjust the `dp_size` parameter based on your available devices. Also adjust the port in following commands.


### Step 2: Run Inference

After starting the vLLM service, run the inference script to generate responses.

```bash
output_dir="./output/Qwen/QwQ-32B"
model_name="Qwen/QwQ-32B"

base_url="http://127.0.0.1:8090/v1"
n_samples=64  # Default sample size for aime24 and aime25

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# --- Run Inference Tasks ---
# aime24 (repeated sample 64 times)
python ./llmeval/inference/online.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --system_prompt_type empty \
    --max_workers 8

# aime25 (repeated sample 64 times)
python ./llmeval/inference/online.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --system_prompt_type empty \
    --max_workers 8
```

Please refer to the [script](./scripts/longcat-flash/online_infer.sh) for more details.

**Note:** We apply repeated sampling to reduce evaluation variance, but it may take a long time to complete (more than 8 hours depending on your device).

### One-Click Pipeline (Recommended)

**Math benchmarks** — generation with math-verify scoring:

```bash
# Full pipeline (data → inference → scoring)
bash scripts/longcat-flash/run_all.sh

# Quick test
BENCHMARKS=QUICK bash scripts/longcat-flash/run_all.sh

# Hard problems
BENCHMARKS=HARD bash scripts/longcat-flash/run_all.sh
```

**Multiple-choice benchmarks** — loglikelihood comparison with acc/acc_norm:

```bash
# 5-shot MMLU / C-Eval
N_SHOT=5 bash scripts/longcat-flash/mc_infer.sh
bash scripts/longcat-flash/mc_score.sh

# Generate mode (alternative)
MC_MODE=generate bash scripts/longcat-flash/mc_infer.sh
```

**Code benchmarks** — generation with sandbox execution + pass@k:

```bash
# HumanEval + MBPP
bash scripts/longcat-flash/code_infer.sh
bash scripts/longcat-flash/code_score.sh

# Pass@64 (64 samples, temperature 0.2)
N_SAMPLES=64 TEMPERATURE=0.2 bash scripts/longcat-flash/code_infer.sh
```

Or run each math step separately:

```bash
bash scripts/longcat-flash/online_infer.sh   # inference
bash scripts/longcat-flash/get_score.sh      # scoring
```

### Step 3: Scoring (CLI)

Alternatively, use the CLI directly:

```bash
python ./llmeval/evaluator.py \
    --input_path "./output/longcat-flash/aime24_bz32.jsonl" \
    --cache_path "./output/longcat-flash/eval_score/aime24_bz32.jsonl" \
    --task_name "math_opensource/aime24" \
    --max_workers 16
```

## Detailed Usage

### Inference Parameters

Common parameters for both online and offline modes:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_samples` | 1 | Number of samples per problem |
| `--temperature` | 0.6 | Sampling temperature |
| `--top_p` | 0.95 | Nucleus sampling parameter |
| `--top_k` | 40 | Top-k sampling parameter |
| `--max_tokens` | 32768 | Maximum tokens to generate |

Online mode specific:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_url` | Required | API server URL |
| `--model_name` | Required | Model name for API |
| `--max_workers` | 128 | Concurrent request threads |

Offline mode specific:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name_or_path` | Required | Local model path or HuggingFace ID |
| `--tensor_parallel_size` | 1 | GPU count for tensor parallelism |
| `--gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `--batch_size` | 128 | Inference batch size |

### Supported Tasks

| Category | Task Name | Scoring |
|----------|-----------|---------|
| Math | `math_opensource/aime24` `math_opensource/aime25` `math_opensource/aime26` | math-verify |
| Math | `math_opensource/gsm8k` `math_opensource/math500` `math_opensource/hmmt25` | math-verify |
| Science | `math_opensource/gpqa_diamond` | math-verify |
| Code | `code_opensource/humaneval` `code_opensource/mbpp` | pass@1 sandbox |
| Code | `code_opensource/humaneval_plus` `code_opensource/mbpp_plus` | pass@1 sandbox |
| MC | `mc_opensource/mmlu` `mc_opensource/mmlu_pro` `mc_opensource/ceval` | loglikelihood |

See [docs/benchmark.md](docs/benchmark.md) for dataset details and HuggingFace links.

### Resume Interrupted Evaluation

If inference is interrupted, simply re-run the same command. The script automatically:
1. Reads existing output file
2. Counts completed samples per problem
3. Continues from where it left off

### Context Length Extension (YaRN)

For contexts exceeding 32K tokens, use RoPE scaling:

**vLLM:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --rope-scaling '{"rope_type":"yarn","factor":4.0}' \
    --max-model-len 131072
```

**SGLang:**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0}}' \
    --context-length 131072
```

## Project Structure

```
LLMEval/
├── llmeval/
│   ├── evaluator.py        # Main orchestrator (math / mc / code)
│   ├── inference/           # Inference backends
│   │   ├── online.py        #   OpenAI-compatible API
│   │   ├── offline.py       #   vLLM local engine
│   │   ├── verifier.py      #   Verifier inference
│   │   └── mc.py            #   MC inference
│   ├── tasks/               # Task-specific scoring
│   │   ├── code_eval/       #   Code (HumanEval / MBPP / pass@k)
│   │   │   ├── execute.py   #     Sandbox execution
│   │   │   └── code_score.py #    pass@k driver
│   │   ├── math_eval/       #   Math (math-verify)
│   │   │   ├── math_score.py
│   │   │   └── utils_parser.py
│   │   └── mc_eval/         #   MC (loglikelihood/generate)
│   │       └── mc_score.py
│   └── utils/               # Shared utilities
│       ├── config.py        #   Configuration dataclasses
│       ├── log.py           #   Logging
│       ├── retry.py         #   API retry logic
│       ├── prompts.py       #   System prompts
│       └── verifier_prompts.py # Verifier prompts
├── scripts/
│   ├── longcat-flash/       # One-click evaluation
│   │   ├── run_all.sh
│   │   ├── online_infer.sh  #   Math inference
│   │   ├── get_score.sh     #   Math scoring
│   │   ├── code_infer.sh    #   Code inference
│   │   ├── code_score.sh    #   Code scoring
│   │   ├── mc_infer.sh      #   MC inference
│   │   └── mc_score.sh      #   MC scoring
│   └── data_process/
│       ├── prepare_math_benchmarks.py
│       └── prepare_code_benchmarks.py
├── tests/                   # Unit tests
├── data/                    # Benchmark datasets
└── docs/                    # Documentation
```

## License

This project is licensed under the MIT License.
