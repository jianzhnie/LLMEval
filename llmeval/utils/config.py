from __future__ import annotations

"""
Configuration classes for a large language model evaluation pipeline.

This module defines a set of dataclasses to handle and validate all
the necessary arguments for a complete evaluation run. The arguments are
categorized into data, prompt formatting, generation parameters, server,
and vLLM-specific settings.

The module provides a comprehensive configuration system that supports:
- Data loading and processing arguments
- Prompt template configuration
- Text generation parameters
- vLLM engine configuration
- Server/API configuration
- Specialized inference modes (online, offline, multiple choice)

All configuration classes include validation logic to ensure parameter
consistency and prevent runtime errors.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY

__all__ = [
    "DataArguments",
    "EvalTaskArguments",
    "GenerationArguments",
    "MCInferConfig",
    "OfflineInferArguments",
    "OnlineInferArguments",
    "PromptArguments",
    "ServerArguments",
    "VLLMEngineArguments",
    "VLLMGenerationArguments",
]

logger = init_logger("eval_config")


@dataclass
class DataArguments:
    """Input, output, and resume paths shared by inference backends."""

    input_file: str = field(
        default="input.jsonl", metadata={"help": "Input JSONL file containing prompts."}
    )
    output_file: str = field(
        default="output.jsonl", metadata={"help": "Output JSONL file to save results."}
    )
    repair_resume: bool = field(
        default=False,
        metadata={
            "help": (
                "Ignore only an unterminated invalid final line in an existing "
                "resume JSONL file. Other malformed rows remain fatal."
            )
        },
    )

    def __post_init__(self) -> None:
        """Validate the input path without creating output-side directories."""
        if not Path(self.input_file).exists():
            raise ValueError(
                f"Input file '{self.input_file}' does not exist. "
                "Please provide a valid input file path."
            )


def _validate_field_names(**field_names: str) -> None:
    """Require non-empty string keys before records are read or written."""
    for name, value in field_names.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def _normalize_tool_choice(value: str) -> str:
    """Validate the string forms accepted by the OpenAI API."""
    if not isinstance(value, str) or value.strip().lower() not in {
        "none",
        "auto",
        "required",
    }:
        raise ValueError("tool_choice must be one of: none, auto, required")
    return value.strip().lower()


@dataclass
class PromptArguments:
    """
    Arguments for configuring prompt templates and formatting.

    This class handles prompt-related configuration including input/output
    keys and system prompt selection from predefined templates.

    Attributes:
        input_key (str): The key in the dataset dictionary for the input text.
        label_key (str): The key for the target/label text in the dataset.
        response_key (str): The key where model generated text will be stored.
        system_prompt_type (str): Optional system prompt type (see SYSTEM_PROMPT_FACTORY).
        system_prompt (Optional[str]): The resolved system prompt text (computed).

    Raises:
        ValueError: If input_key or label_key is empty, or if system_prompt_type
                   is not found in SYSTEM_PROMPT_FACTORY.
    """

    input_key: str = field(
        default="prompt", metadata={"help": "Key for input text in dataset."}
    )
    label_key: str = field(
        default="answer", metadata={"help": "Key for target/label text in dataset."}
    )
    response_key: str = field(
        default="gen", metadata={"help": "Key for model generated text."}
    )
    system_prompt_type: str = field(
        default="empty",
        metadata={
            "help": (
                "System prompt type. Valid: deepseek_r1, amthinking, openr1, "
                "default, empty (no prompt). Default: 'empty'."
            )
        },
    )
    # Computed value based on system_prompt_type; not settable via CLI
    system_prompt: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Validate prompt arguments and resolve system prompt.

        Raises:
            ValueError: If input_key or label_key is empty, or if system_prompt_type
                       is not found in SYSTEM_PROMPT_FACTORY.
        """
        _validate_field_names(
            input_key=self.input_key,
            label_key=self.label_key,
            response_key=self.response_key,
        )

        if (
            self.system_prompt_type is not None
            and self.system_prompt_type not in SYSTEM_PROMPT_FACTORY
        ):
            raise ValueError(
                f"Invalid system prompt type: {self.system_prompt_type}. "
                f"Valid options are: {list(SYSTEM_PROMPT_FACTORY.keys())}"
            )
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(self.system_prompt_type)
        prompt_length = len(self.system_prompt or "")
        logger.info(
            "Using system_prompt_type: %s, content_length: %d",
            self.system_prompt_type,
            prompt_length,
        )
        logger.info(
            "If you want to customize the system prompt, please modify the "
            "SYSTEM_PROMPT_FACTORY in llmeval/utils/prompts.py"
        )


@dataclass
class GenerationArguments:
    """Backend-independent text generation arguments."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    temperature: float = field(default=0.6, metadata={"help": "Sampling temperature."})
    top_p: float = field(
        default=0.95, metadata={"help": "Nucleus sampling probability threshold."}
    )
    max_completion_tokens: int = field(
        default=32768,
        metadata={"help": "Maximum completion tokens to generate per sequence."},
    )
    seed: int = field(
        default=0, metadata={"help": "Generation seed used for reproducible sampling."}
    )

    def __post_init__(self) -> None:
        """Validate backend-independent generation arguments."""
        if self.n_samples <= 0:
            raise ValueError(
                f"Number of samples must be positive, but got {self.n_samples}."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got: {self.temperature}"
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"Top-p must be between 0 and 1, but got {self.top_p}.")
        if self.max_completion_tokens <= 0:
            raise ValueError(
                "max_completion_tokens must be a positive integer, got: "
                f"{self.max_completion_tokens}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got: {self.seed}")


@dataclass
class VLLMGenerationArguments:
    """Generation options implemented by the local vLLM backend."""

    top_k: int = field(default=40, metadata={"help": "Top-k sampling parameter."})
    skip_special_tokens: bool = field(
        default=True, metadata={"help": "Remove special tokens from generated text."}
    )
    repetition_penalty: float = field(
        default=1.0, metadata={"help": "Local vLLM repetition penalty."}
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking through vLLM chat-template arguments."},
    )

    def __post_init__(self) -> None:
        """Validate local vLLM generation arguments."""
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"Top-k must be positive or -1 (disabled), got: {self.top_k}"
            )
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"Repetition penalty must be positive, got: {self.repetition_penalty}"
            )
        if not isinstance(self.skip_special_tokens, bool):
            raise ValueError(
                "skip_special_tokens must be a boolean, got: "
                f"{self.skip_special_tokens}"
            )
        if not isinstance(self.enable_thinking, bool):
            raise ValueError(
                f"enable_thinking must be a boolean, got: {self.enable_thinking}"
            )


@dataclass
class VLLMEngineArguments:
    """
    Arguments for configuring the vLLM inference backend.

    This class handles all vLLM-specific configuration including model
    loading, memory management, and parallel processing settings.

    Attributes:
        model_name_or_path (str): Path or name of the model to load.
        trust_remote_code (bool): Whether to trust remote code.
        dtype (str): Data type for model execution (e.g., "float16", "auto", "bfloat16").
        max_model_len (int): Maximum context length for the model.
        rope_scaling (str): RoPE scaling configuration as a JSON string. If empty,
            no scaling is applied. Parsed into `rope_scaling_dict`.
        rope_scaling_dict (Optional[Dict[str, Any]]): Parsed RoPE scaling configuration (computed).
        gpu_memory_utilization (float): Target GPU memory usage (0-1].
        tensor_parallel_size (int): Number of GPUs for tensor parallelism.
        pipeline_parallel_size (int): Number of GPUs for pipeline parallelism.
        enable_chunked_prefill (bool): Reduce memory usage during generation.
        enable_prefix_caching (bool): Enable KV cache prefix optimization.
        max_num_batched_tokens (Optional[int]): Maximum number of tokens per batch.
        max_num_seqs (Optional[int]): Maximum number of parallel sequences.
        enforce_eager (bool): Enforce eager execution for debugging purposes.
        device (str): Device to use for inference (e.g., "cuda", "auto").
        quantization (Optional[str]): Quantization method (e.g., "awq", "gptq", None).

    Raises:
        ValueError: If any parameter is outside its valid range or if
                   rope_scaling contains invalid JSON.
    """

    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B", metadata={"help": "Path to the model directory."}
    )
    model_revision: str | None = field(
        default=None,
        metadata={"help": "Optional model revision to load."},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Whether to trust remote code."}
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": 'Data type for model execution (e.g., "float16", "auto", "bfloat16").'
        },
    )
    max_model_len: int = field(
        default=32768, metadata={"help": "Maximum sequence length for the model."}
    )
    rope_scaling: str = field(
        default="{}", metadata={"help": "RoPE scaling configuration as a JSON string."}
    )
    # Parsed representation; not settable via CLI
    rope_scaling_dict: dict[str, Any] | None = field(init=False, default=None)
    gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "Target GPU memory utilization (0-1)."}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of GPUs to use for tensor parallelism."}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "Number of GPUs to use for pipeline parallelism."}
    )
    enable_chunked_prefill: bool = field(
        default=False,
        metadata={
            "help": "Enable chunked prefill to reduce memory usage during generation."
        },
    )
    enable_prefix_caching: bool = field(
        default=False, metadata={"help": "Enable KV cache prefix optimization."}
    )
    max_num_batched_tokens: int | None = field(
        default=512000, metadata={"help": "Maximum tokens per batch."}
    )
    max_num_seqs: int | None = field(
        default=4096, metadata={"help": "Maximum parallel sequences."}
    )
    enforce_eager: bool = field(
        default=True,
        metadata={"help": "Enforce eager execution for debugging purposes."},
    )
    device: str = field(
        default="cuda",
        metadata={"help": 'Device to use for inference (e.g., "cuda", "auto").'},
    )
    quantization: str | None = field(
        default=None,
        metadata={"help": 'Quantization method (e.g., "awq", "gptq", None).'},
    )

    def __post_init__(self) -> None:
        """
        Validate vLLM arguments and parse rope_scaling value.

        Raises:
            ValueError: If any parameter is outside its valid range or if
                       rope_scaling contains invalid JSON.
        """
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"GPU memory utilization must be between 0 and 1, but got {self.gpu_memory_utilization}."
            )
        if self.max_model_len <= 0:
            raise ValueError(
                f"Max model length must be positive, but got {self.max_model_len}."
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"Tensor parallel size must be at least 1, but got {self.tensor_parallel_size}."
            )
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                f"Pipeline parallel size must be at least 1, but got {self.pipeline_parallel_size}."
            )

        # Validate dtype
        valid_dtypes = ["auto", "float16", "float32", "bfloat16"]
        if self.dtype not in valid_dtypes:
            logger.warning(f"Unknown dtype {self.dtype}. Valid options: {valid_dtypes}")

        # Validate quantization
        if self.quantization and self.quantization not in [
            "awq",
            "gptq",
            "squeezellm",
            None,
        ]:
            logger.warning(
                f"Unknown quantization method {self.quantization}. Supported: awq, gptq, squeezellm"
            )

        # Parse rope_scaling into rope_scaling_dict, keeping the original field as string.
        text = (self.rope_scaling or "").strip()
        if not text:
            self.rope_scaling_dict = None
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON string for rope_scaling: {self.rope_scaling}. Error: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"rope_scaling must be a JSON object, but got "
                    f"{type(parsed).__name__}: {self.rope_scaling}"
                )
            self.rope_scaling_dict = parsed
            logger.info(f"Successfully parsed rope_scaling: {self.rope_scaling_dict}")

        # Validate optional engine limits
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens < 1:
            raise ValueError(
                f"max_num_batched_tokens must be positive, but got {self.max_num_batched_tokens}."
            )
        if self.max_num_seqs is not None and self.max_num_seqs < 1:
            raise ValueError(
                f"max_num_seqs must be positive, but got {self.max_num_seqs}."
            )


@dataclass
class ServerArguments:
    """
    Arguments for configuring an OpenAI-compatible server/API.

    This class handles server connection parameters and client-side
    concurrency settings for API-based inference.

    Attributes:
        max_workers (int): Maximum number of worker threads for client-side concurrency.
        base_url (str): Base URL of the OpenAI-compatible server.
        model_name (str): Model name to use on the server.
        request_timeout (int): Timeout (seconds) for requests to server.
        max_retries (int): Maximum number of retries for failed requests.
        api_key (Optional[str]): API key for authentication.
        organization (Optional[str]): Organization ID for API usage.
        extra_body (str): JSON object containing explicit provider extensions.

    Raises:
        ValueError: If any parameter is outside its valid range.
    """

    max_workers: int = field(
        default=128, metadata={"help": "Maximum number of worker threads."}
    )
    base_url: str = field(
        default="https://api.openai.com/v1",
        metadata={"help": "Base URL of the OpenAI-compatible API endpoint"},
    )
    model_name: str = field(
        default="gpt-4o", metadata={"help": "Model name served by the API endpoint"}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for failed API requests."},
    )
    request_timeout: int = field(
        default=99999, metadata={"help": "Timeout (seconds) for API requests."}
    )
    api_key: str | None = field(
        default=None,
        metadata={
            "help": "API key for authentication (can also use OPENAI_API_KEY env var)."
        },
    )
    organization: str | None = field(
        default=None, metadata={"help": "Organization ID for API usage."}
    )
    tool_choice: str = field(
        default="none",
        metadata={"help": "Tool choice mode: none, auto, or required."},
    )
    extra_body: str = field(
        default="{}",
        metadata={"help": "JSON object of non-standard provider request fields."},
    )
    extra_body_dict: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate server arguments after initialization.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if self.max_workers <= 0:
            raise ValueError(
                f"Maximum number of worker threads must be a positive integer, but got {self.max_workers}."
            )
        if self.request_timeout <= 0:
            raise ValueError(
                f"Request timeout must be a positive integer, but got {self.request_timeout}."
            )
        if self.max_retries < 0:
            raise ValueError(
                f"Max retries must be non-negative, but got {self.max_retries}."
            )
        # Validate URL format
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Base URL must start with http:// or https://, but got {self.base_url}"
            )
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        try:
            parsed_extra_body = json.loads(self.extra_body or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_body must be valid JSON: {exc}") from exc
        if not isinstance(parsed_extra_body, dict):
            raise ValueError("extra_body must be a JSON object")
        self.extra_body_dict = parsed_extra_body
        self.tool_choice = _normalize_tool_choice(self.tool_choice)

        # Check for API key from environment if not provided
        if self.api_key is None and "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
            logger.info("Using API key from OPENAI_API_KEY environment variable")


@dataclass
class OnlineInferArguments(
    DataArguments, PromptArguments, GenerationArguments, ServerArguments
):
    """
    Arguments specific to online (OpenAI-compatible API) inference.

    This class combines all necessary arguments for online inference,
    inheriting from DataArguments, PromptArguments, GenerationArguments,
    and ServerArguments.
    """

    def __post_init__(self) -> None:
        """Validate all inherited arguments."""
        # Only validate what online mode needs; no vLLM engine args
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        ServerArguments.__post_init__(self)


@dataclass
class OfflineInferArguments(
    DataArguments,
    PromptArguments,
    GenerationArguments,
    VLLMGenerationArguments,
    VLLMEngineArguments,
):
    """
    Arguments specific to offline (local vLLM engine) inference.

    This class combines all necessary arguments for offline inference,
    inheriting from DataArguments, PromptArguments, GenerationArguments,
    VLLMGenerationArguments, and VLLMEngineArguments.
    """

    batch_size: int = field(
        default=128, metadata={"help": "Batch size for local vLLM inference."}
    )
    fail_fast: bool = field(
        default=True,
        metadata={
            "help": (
                "Stop on the first failed inference batch. Set to false to "
                "record failed batches and continue."
            )
        },
    )

    def __post_init__(self) -> None:
        """Validate all inherited arguments."""
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        VLLMGenerationArguments.__post_init__(self)
        VLLMEngineArguments.__post_init__(self)
        if self.batch_size <= 0:
            raise ValueError(
                f"Batch size must be a positive integer, but got {self.batch_size}."
            )


@dataclass
class MCInferConfig(
    DataArguments, PromptArguments, GenerationArguments, ServerArguments
):
    """Configuration for multiple-choice inference.

    Shared data, prompt, generation, and server fields come from their argument
    classes. This class only overrides MC-specific defaults and adds MC-only
    scoring and few-shot options.
    """

    # MC validates paths when MCRunner starts, allowing default construction for
    # CLI introspection and tests.
    input_file: str = field(
        default="", metadata={"help": "Path to the input JSONL file (MC items)."}
    )
    output_file: str = field(
        default="", metadata={"help": "Path to the output JSONL file."}
    )
    base_url: str = field(
        default="http://127.0.0.1:8200/v1",
        metadata={"help": "Base URL of the OpenAI-compatible API endpoint."},
    )
    model_name: str = field(
        default="longcat-flash",
        metadata={"help": "Served model name used in requests."},
    )
    max_workers: int = field(
        default=32, metadata={"help": "Number of concurrent worker threads."}
    )
    request_timeout: int = field(
        default=300, metadata={"help": "Per-request timeout in seconds."}
    )
    max_completion_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum completion tokens in generate mode."},
    )
    temperature: float = field(
        default=0.0, metadata={"help": "Sampling temperature (0.0 = deterministic)."}
    )
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"),
        metadata={"help": "API key (default: OPENAI_API_KEY env var)."},
    )

    mode: str = field(
        default="loglikelihood",
        metadata={"help": "Inference mode: 'loglikelihood' or 'generate'."},
    )
    loglikelihood_mode: str = field(
        default="first_token",
        metadata={
            "help": (
                "MC scoring mode: first_token (default, Chat Completions), "
                "continuation (legacy Completions compatibility), or auto "
                "(alias for first_token)."
            )
        },
    )
    n_shot: int = field(
        default=0,
        metadata={"help": "Few-shot example count (0 = zero-shot)."},
    )
    few_shot_file: str = field(
        default="",
        metadata={
            "help": (
                "Dev file for few-shot examples. Required when n_shot > 0: "
                "few-shot examples are never sampled from the evaluation set, "
                "to avoid leakage."
            )
        },
    )

    def __post_init__(self) -> None:
        """Validate shared fields and MC-specific options."""
        # Keep default construction available for CLI introspection. Once a path
        # is supplied, apply the same existence check as other inference modes.
        if self.input_file:
            DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        ServerArguments.__post_init__(self)

        if self.mode not in ("loglikelihood", "generate"):
            raise ValueError(
                f"mode must be one of {('loglikelihood', 'generate')}, got: {self.mode!r}"
            )
        if self.loglikelihood_mode not in ("auto", "continuation", "first_token"):
            raise ValueError(
                "loglikelihood_mode must be one of ('auto', 'continuation', "
                f"'first_token'), got: {self.loglikelihood_mode!r}"
            )
        if self.n_shot < 0:
            raise ValueError(f"n_shot must be non-negative, got: {self.n_shot}")
        if self.n_shot > 0 and not self.few_shot_file:
            raise ValueError(
                "few_shot_file is required when n_shot is greater than zero; "
                "do not sample demonstrations from the evaluation set"
            )


@dataclass
class EvalTaskArguments:
    """
    Arguments for task-specific evaluation configuration.

    This class handles the configuration parameters for evaluating model outputs
    on specific tasks like math problems or code benchmarks.

    Attributes:
        input_path (str): Path to the input JSONL file containing evaluation data.
        task_name (str): Name of the evaluation task to run.
            Validation is delegated to the task registry.
        label_key (str): Key for target/label text in dataset.
        response_key (str): Key for model generated text.
        cache_path (str): Legacy name for the evaluation result JSONL path. Existing
            directory paths are resolved to a task-specific JSONL file.
        max_workers (int): Maximum number of worker threads for parallel processing.


        timeout (int): Timeout for LLM inference in seconds.
    """

    input_path: str = field(
        metadata={"help": "Path to the input JSONL file containing evaluation data."}
    )
    task_name: str = field(
        default="math_opensource/aime24",
        metadata={"help": "Evaluation task name (e.g. math_opensource/aime24)."},
    )
    label_key: str = field(
        default="answer", metadata={"help": "Key for target/label text in dataset."}
    )
    response_key: str = field(
        default="gen", metadata={"help": "Key for model generated text."}
    )
    mc_aggregation: str = field(
        default="first",
        metadata={
            "help": "MC generate aggregation: first, majority_vote, any_correct, or per_sample."
        },
    )
    allow_unsafe_code: bool = field(
        default=False,
        metadata={
            "help": "Explicitly allow execution of generated code during evaluation."
        },
    )
    code_k_values: str = field(
        default="1,10,64",
        metadata={"help": "Comma-separated pass@k values for code evaluation."},
    )
    code_k_values_tuple: tuple[int, ...] = field(init=False, default=(1, 10, 64))

    cache_path: str = field(
        default="./cache/results.jsonl",
        metadata={"help": "JSONL file path for saving detailed evaluation results."},
    )
    result_path: str = field(
        default="",
        metadata={
            "help": "Preferred result JSONL path; overrides legacy --cache_path."
        },
    )
    max_workers: int = field(
        default=128,
        metadata={"help": "Maximum number of worker threads for parallel processing."},
    )
    timeout: int = field(
        default=20, metadata={"help": "Timeout for LLM inference in seconds."}
    )
    exec_timeout: float = field(
        default=3.0,
        metadata={
            "help": "Per-item code execution timeout in seconds (code tasks only)."
        },
    )
    seed: int = field(
        default=0, metadata={"help": "Random seed for bootstrap uncertainty."}
    )
    bootstrap_samples: int = field(
        default=1000,
        metadata={"help": "Number of bootstrap resamples for uncertainty."},
    )
    confidence_level: float = field(
        default=0.95, metadata={"help": "Bootstrap confidence level."}
    )

    def __post_init__(self) -> None:
        """
        Validate evaluation task arguments after initialization.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not self.input_path:
            raise ValueError("input_path is required")
        if not Path(self.input_path).exists():
            raise ValueError(f"input_path {self.input_path} does not exist")
        if self.result_path:
            self.cache_path = self.result_path
        if not self.cache_path:
            raise ValueError("result_path or cache_path is required")
        _validate_field_names(
            label_key=self.label_key,
            response_key=self.response_key,
        )
        try:
            parsed_k = tuple(
                dict.fromkeys(
                    int(value.strip())
                    for value in self.code_k_values.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"code_k_values must be comma-separated integers, got {self.code_k_values!r}"
            ) from exc
        if not parsed_k or any(value <= 0 for value in parsed_k):
            raise ValueError(
                f"code_k_values must contain positive integers, got {self.code_k_values!r}"
            )
        self.code_k_values_tuple = parsed_k
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.exec_timeout <= 0:
            raise ValueError(f"exec_timeout must be positive, got {self.exec_timeout}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.mc_aggregation not in (
            "first",
            "majority_vote",
            "any_correct",
            "per_sample",
        ):
            raise ValueError(
                "mc_aggregation must be one of ('first', 'majority_vote', "
                f"'any_correct', 'per_sample'), got: {self.mc_aggregation!r}"
            )
        if self.bootstrap_samples < 0:
            raise ValueError(
                f"bootstrap_samples must be non-negative, got {self.bootstrap_samples}"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            )
