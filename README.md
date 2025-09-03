# LLM Evaluation

## Benchmark

Currently, this repository contains the code to reproduce the following scores.

| Datasets                 | QwQ-32B |
|--------------------------|---------|
| AIME24                   | 79.5    |
| AIME25                   | 69.5    |






## Evaluation

### Step 0: Requirements

Ensure you have Python >= 3.9 installed along with GPU devices totaling at least 100GB of memory (for bf16 inference with enough context length).

Install the dependencies using:

```bash
pip install -f requirements.txt
```

### Step 1: Start vLLM Server

First, start the vLLM server with the following command:

```bash
model_path="Qwen/QwQ-32B" # or path to your local checkpoint
model_name="Qwen/QwQ-32B"
num_gpus=4

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --enforce-eager \
    --port 8030
```
Adjust the `tensor_parallel_size` parameter based on your available devices.
### Optional : Start SGLang server/router

Since the evaluation could takes days, we also suggest using SGLang with data parallelism to accelerate the evaluation. Refer to [SGLang documentation](https://docs.sglang.ai/router/router.html) for more details.
```bash
# Use router to support better data parallelism
python -m sglang_router.launch_server --model-path Qwen/QwQ-32B --dp-size 4 --host=0.0.0.0 --port=30000

```
Adjust the `dp_size` parameter based on your available devices. Also adjust the port in following commands.

### Step 2: Run Inference

After starting the vLLM service, run the inference script to generate responses.

```bash
mkdir -p output

# aime24 (repeated sample 64 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/aime24.jsonl" --output_file "./output/aime24_bz64.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name "Qwen/QwQ-32B"

# aime25 (repeated sample 64 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/aime25.jsonl" --output_file "./output/aime25_bz64.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name "Qwen/QwQ-32B"
```

**Note:** We apply repeated sampling to reduce evaluation variance, but it may take a long time to complete (more than 8 hours depending on your device).

#### Parameter Description

- `--base_url`: Base URL of the vLLM service
- `--model_name`: Must match the model name used in Step 1
- `--n_samples`: Number of samples per prompt
  - AIME24 / AIME 25: Recommended 64 samples
  - LiveCodeBench: Recommended 8 samples
  - IFEval: Recommended 1 sample
- `--input_file`: Input data file path
- `--output_file`: Output result file path, model responses will be stored in the `gen` field
- `--max_workers`: Maximum number of concurrent threads to control inference speed and resource usage

#### Sampling Parameters

We use ``top_p=0.95``, ``temperature=0.6``, ``top_k=40``, ``max_tokens=32768`` for sampling.

#### Resuming Interrupted Inference

If the inference process is interrupted, simply rerun the same command to resume. The script will automatically read the previous output file and process any prompts that haven't completed the required number of samples.

## Scoring

After completing the inference, use the following commands for scoring:

```bash
mkdir -p eval_res

python  ./eval/eval.py --input_path ./output/aime24_bz64.jsonl --cache_path ./eval_res/aime24_bz64.jsonl  --task_name "math_opensource/aime24" > ./eval_res/aime24_bz64_res_result.txt

python  ./eval/eval.py --input_path ./output/aime25_bz64.jsonl --cache_path ./eval_res/aime25_bz64.jsonl  --task_name "math_opensource/aime25" > ./eval_res/aime25_bz64_res_result.txt

# download all test cases
python ./data/process_data.py
# Note: running all code test cases can be very slow (more than 4 hours)
python  ./eval/eval.py --input_path ./output/livecodebench_v5_bz8.jsonl --cache_path ./eval_res/livecodebench_v5_bz8.jsonl  --task_name "livecodebench" > ./eval_res/livecodebench_v5_bz8_res_result.txt

python  ./eval/eval.py --input_path ./output/ifeval_bz1.jsonl --cache_path ./eval_res/ifeval_bz1.jsonl  --task_name "ifeval" > ./eval_res/ifeval_bz1_res_result.txt
```

### Parameter Description

- `--input_path`: Input file path, can directly use the output file from multi-threaded inference or other files with consistent format. Requirements:
  - JSONL format
  - Contains `prompt` and corresponding fields
  - Model responses stored in the `gen` field
- `--cache_path`: Cache directory for storing temporary files during evaluation
- `--task_name`: Evaluation task name, must be one of the following options:
  - `math_opensource/aime24`
  - `math_opensource/aime25`
  - `livecodebench`
  - `ifeval`
