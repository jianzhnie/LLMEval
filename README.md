# LLM Evaluation

## Overview
We have successfully reproduced various open-source model results on the AIME 2024 & AIME 2025 benchmarks.

For benchmarks like AIME24, which contains only 30 problems, it is crucial to sample multiple responses as this can introduce high variance across repeated runs. The number of responses sampled per prompt likely accounts for the slight differences between our evaluation results and those reported by DeepSeek.

### DeepSeek-R1-Distill-Qwen-32B

| Datasets | (🤗 LLMEval) | DeepSeek-R1-Distill-Qwen-32B（Reported） |
| :------: | :---------: | :--------------------------------------: |
|  AIME24  |   70.625    |                   72.6                   |
|  AIME25  |   55.052    |                   59.0                   |
| MATH-500 |    93.2     |                   94.3                   |


### QwQ-32B

| Datasets | (🤗 LLMEval) | QwQ-32B（Reported） |
| :------: | :---------: | :-----------------: |
|  AIME24  |    78.65    |        79.5         |
|  AIME25  |    67.22    |        69.5         |



### Skywork-OR1-32B

| Datasets | (🤗 LLMEval) | Skywork-OR1-32B（Reported） |
| :------: | :---------: | :-------------------------: |
|  AIME24  |    81.25    |            82.2             |
|  AIME25  |    72.66    |            73.3             |



### OpenThinker3-7B

| Datasets | (🤗 LLMEval) | OpenThinker3-7B（Reported） |
| :------: | :---------: | :-------------------------: |
| AIME24   | 70.41      | 69.0                    |
| AIME25   | 59.16      | 53.3                    |


## Installation

### Basic Environment Setup

| software  | version    |
| --------- | ---------- |
| Python    | == 3.10    |
| CANN      | == 8.1.RC1 |
| torch     | == 2.5.1   |
| torch_npu | == 2.5.1   |

For basic environment setup, please refer to this [documentation](https://gitee.com/ascend/pytorch).

### vllm & vllm-ascend

To properly use vllm in verl, you need to compile and install vllm and vllm-ascend using the following commands. Please note the installation method varies depending on your machine type.

```bash
# vllm
git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt

# for Atlas 200T A2 Box16
VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

# for Atlas 900 A2 PODc
VLLM_TARGET_DEVICE=empty pip install -e .
# vllm-ascend
git clone -b v0.7.3.post1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
```

### llmeval

Install the `llmeval` package by cloning the repository and then using `pip` to install it in editable mode. This will also install all the necessary dependencies.

```bash
git clone https://gitee.com/jianzhnie/LLMEval.git
cd LLMEval
pip install -e .
```

## Evaluation

### Step 1: Start vLLM Server

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
Adjust the `tensor_parallel_size` parameter based on your available devices.

Please refer to the [script](./scripts/model_server.sh) for more details.


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
python ./llmeval/vllm_utils/infer_multithread.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --max_workers 8

# aime25 (repeated sample 64 times)
python ./llmeval/vllm_utils/infer_multithread.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --max_workers 8
```
Please refer to the [script](./scripts/run_infer.sh) for more details.


**Note:** We apply repeated sampling to reduce evaluation variance, but it may take a long time to complete (more than 8 hours depending on your device).

#### Parameter Description

- `--base_url`: Base URL of the vLLM service
- `--model_name`: Must match the model name used in Step 1
- `--n_samples`: Number of samples per prompt
  - AIME24 / AIME 25: Recommended 64 samples
- `--input_file`: Input data file path
- `--output_file`: Output result file path, model responses will be stored in the `gen` field
- `--max_workers`: Maximum number of concurrent threads to control inference speed and resource usage

#### Sampling Parameters

We use ``top_p=0.95``, ``temperature=0.6``, ``top_k=40``, ``max_tokens=32768`` for sampling.

#### Resuming Interrupted Inference

If the inference process is interrupted, simply rerun the same command to resume. The script will automatically read the previous output file and process any prompts that haven't completed the required number of samples.

### Step 3: Scoring

After completing the inference, use the following commands for scoring:

```bash
output_dir="./output/Qwen/QwQ-32B"
n_samples=64  # Default sample size for aime24 and aime25

# Evaluation output directory
reval_dir="${output_dir}/eval_score"
# Create evaluation directory if it doesn't exist
mkdir -p "${reval_dir}"

# --- Evaluate Each Task ---
# Evaluate aime24
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime24_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime24" \
    --max_workers 16 \
    > "${reval_dir}/aime24_bz${n_samples}_res_result.txt"

# Evaluate aime25
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime25_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime25" \
    --max_workers 16 \
    > "${reval_dir}/aime25_bz${n_samples}_res_result.txt"
```
Please refer to the [script](./scripts/get_scores.sh) for more details.

#### Parameter Description

- `--input_path`: Input file path, can directly use the output file from multi-threaded inference or other files with consistent format. Requirements:
  - JSONL format
  - Contains `prompt` and corresponding fields
  - Model responses stored in the `gen` field
- `--cache_path`: Cache directory for storing temporary files during evaluation
- `--task_name`: Evaluation task name, must be one of the following options:
  - `math_opensource/aime24`
  - `math_opensource/aime25`
- `max_workers`: Maximum number of concurrent threads to control evaluation speed and resource usage.
