# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMEval is a comprehensive evaluation system for assessing Large Language Models (LLMs) on math reasoning (AIME 2024/2025/2026, MATH-500, GSM8K, HMMT), multiple-choice (MMLU, MMLU-Pro, C-Eval), and code generation (HumanEval, MBPP) benchmarks. It supports both online (OpenAI-compatible API) and offline (local vLLM engine) inference modes with built-in answer verification using the `math-verify` library.

## Build / Lint / Test

```bash
# Install in editable mode
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_evaluator.py

# Run a single test function
pytest tests/test_evaluator.py::TestEvaluateTask::test_selects_task_specific_arguments -v

# Lint (ruff format + ruff --fix via pre-commit)
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

## Architecture

The system follows a three-stage evaluation pipeline: **Inference → Scoring → Verification**.

### Configuration System (`llmeval/utils/config.py`)

All CLI arguments use `HfArgumentParser` from `transformers` to parse into strongly-typed dataclasses. The inheritance hierarchy is:

- `DataArguments` — input/output paths and resume repair
- `PromptArguments` — input/label/response keys, system prompt selection from `SYSTEM_PROMPT_FACTORY`
- `GenerationArguments` — temperature, top_p, n_samples, max_completion_tokens, seed
- `VLLMGenerationArguments` — top_k, thinking, special-token, and repetition controls
- `VLLMEngineArguments` — model path, tensor parallelism, GPU memory, RoPE scaling
- `ServerArguments` — base URL, model name, max_workers, retries/timeout

These are composed via multiple inheritance into mode-specific argument classes:
- `OnlineInferArguments` → `DataArguments + PromptArguments + GenerationArguments + ServerArguments`
- `OfflineInferArguments` → `DataArguments + PromptArguments + GenerationArguments + VLLMGenerationArguments + VLLMEngineArguments`
- `MathEvalArguments`, `MCEvalArguments`, `CodeEvalArguments` — task-family-specific scoring arguments

### Prompt Templates (`llmeval/utils/prompts.py`)

System prompts are stored in `SYSTEM_PROMPT_FACTORY`, mapping string names to prompt text. The `is_chat_template_applied()` function detects pre-formatted queries and raises an error to prevent double-application of chat templates.

### Inference Layer (`llmeval/inference/`)

Three entry points, each a standalone script with its own `main()`:

- **`online.py`** — `InferenceClient` wraps `openai.OpenAI` with exponential backoff retry logic. `InferenceRunner` orchestrates concurrent requests via `ThreadPoolExecutor`, with resume support based on stable `doc_id`. Entry: `python -m llmeval.inference.online --input_file ... --output_file ... --base_url ...`

- **`offline.py`** — `OfflineInferenceRunner` uses vLLM's native `LLM` class for local batched inference. Converts data to chat message format, handles resume, writes results incrementally. Entry: `python -m llmeval.inference.offline --model_name_or_path ... --input_file ...`

- **`mc.py`** — Multiple-choice inference. Loglikelihood mode scores each choice via the completions API (writes `choices` / `gold` / `logprobs`); generate mode produces free-form text (writes `gen`) for letter extraction at scoring time.

### Evaluation/Scoring Layer (`llmeval/evaluator.py`, `llmeval/tasks/`)

- **`llmeval/evaluator.py`** — Entry point for scoring. Selects `MathEvalArguments`, `MCEvalArguments`, or `CodeEvalArguments` from the task prefix, validates input JSONL, then dispatches to the registered scorer. Entry: `python -m llmeval.evaluator --input_path ... --task_name math_opensource/aime24 --result_path ...`

- **`tasks/math_eval/math_score.py`** — Core math scoring using the `math-verify` library. Uses `ProcessPool` (pebble) for parallel answer verification with timeout support. Implements `math_metric` with both `ExprExtractionConfig` and `LatexExtractionConfig` for robust answer extraction. The registry writes one aggregate result JSON.

- **`tasks/math_eval/utils_parser.py`** — Ground truth parsing utilities (gsm8k `####` format, olympiadbench, generic).

- **`tasks/mc_eval/mc_score.py`** — MC accuracy scoring. Loglikelihood mode argmaxes per-choice answer-token or continuation logprobs; generate mode extracts answer letters (`Answer: X` marker or last standalone A–J).

- **`tasks/code_eval/code_score.py` + `execute.py`** — Code scoring with pass@1: extracts code from generations, then executes candidate + test harness in a guarded subprocess. This guard is not a security sandbox; code evaluation must run in an externally isolated container or sandbox.

### Data Flow

1. Input: prepared JSONL with a unique `doc_id`, plus `prompt` and `answer`
2. Inference: Each prompt is sampled `n_samples` times → output JSONL with `gen` list appended (MC loglikelihood mode instead writes `choices` / `gold` / `logprobs`)
3. Scoring: `evaluator.py` reads the output JSONL and dispatches to the task-family scorer — math items go through `math-verify` → appends `accuracy`, `extracted_answer`, `extracted_gold` fields
4. Results: Metrics are logged and one aggregate JSON is written to `result_path`

### Resume Mechanism

Online, offline, and MC inference resume by counting completed output rows for each stable `doc_id`.

### Supported Task Names

Three task families, dispatched on the prefix before `/`:

- **Math** (`math_opensource/*`, math-verify scoring): `math500`, `math`, `gsm8k`, `aime24`, `aime25`, `aime26`, `hmmt25`, `gpqa_diamond`, `hle_full`
- **Multiple-choice** (`mc_opensource/*`, loglikelihood or generation scoring): `mmlu`, `mmlu_pro`, `ceval`
- **Code** (`code_opensource/*`, guarded-subprocess pass@k): `humaneval`, `mbpp`, `humaneval_plus`, `mbpp_plus`

Task-family dispatch and supported task names are defined in `llmeval/tasks/registry.py`.

## Key Dependencies

- `math-verify` — mathematical answer verification and metric computation
- `pebble` — process-pool based parallel processing with timeout
- `openai` — OpenAI-compatible API client for online inference
- `vllm` — local inference engine (optional, for offline mode)
- `transformers` — `HfArgumentParser` for CLI argument parsing
- `torch`, `datasets`, `tokenizers`, `tqdm`

## Environment

- Python >= 3.10
- Optional NPU support: Huawei Ascend with CANN >= 8.1.RC1, torch_npu >= 2.5.1, vllm-ascend
- `OPENAI_API_KEY` environment variable used for API authentication (defaults to `"EMPTY"` for self-hosted servers)
- Project uses `setuptools` as build backend (see `pyproject.toml`)
