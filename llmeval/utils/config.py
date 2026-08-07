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
]

logger = init_logger("eval_config")


@dataclass
class DataArguments:
    """
    Arguments for configuring the dataset and data loading.

    This class handles all parameters related to data input/output,
    including file paths, caching, and batch processing.

    Attributes:
        input_file (str): Path to the input JSONL file containing prompts.
        cache_dir (str): Path to the directory for caching models and data.
        output_file (str): Path to the output JSONL file to save results.
        task (str): Optional evaluation task name. Callers should pass the
            actual benchmark when task-specific handling is needed.
        batch_size (int): The number of samples to process in each batch.
    """

    input_file: str = field(
        default="input.jsonl", metadata={"help": "Input JSONL file containing prompts."}
    )
    cache_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.cache/huggingface"),
        metadata={"help": "Cache directory for models."},
    )
    output_file: str = field(
        default="output.jsonl", metadata={"help": "Output JSONL file to save results."}
    )
    task: str = field(
        default="", metadata={"help": "Optional name of the evaluation task."}
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
    batch_size: int = field(
        default=128, metadata={"help": "Batch size for data loading."}
    )
    fail_fast: bool = field(
        default=True,
        metadata={
            "help": (
                "Stop on the first failed inference batch. Set to false to "
                "record the failed batch and continue."
            )
        },
    )

    def __post_init__(self) -> None:
        """
        Validate data arguments after initialization.

        Raises:
            ValueError: If batch_size is not a positive integer or if input file doesn't exist.
        """
        if self.batch_size <= 0:
            raise ValueError(
                f"Batch size must be a positive integer, but got {self.batch_size}."
            )

        # Validate input file exists
        if not Path(self.input_file).exists():
            raise ValueError(
                f"Input file '{self.input_file}' does not exist. "
                "Please provide a valid input file path."
            )


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
        if not self.input_key:
            raise ValueError("Input key must be a non-empty string.")
        if not self.label_key:
            raise ValueError("Label key must be a non-empty string.")

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
    """
    Arguments for controlling the text generation process.

    This class handles all parameters related to text generation including
    sampling strategies, token limits, and output formatting.

    Attributes:
        do_sample (bool): Whether to use sampling or greedy decoding.
        n_samples (int): Number of sequences to generate per prompt.
        temperature (float): Controls randomness; higher is more diverse.
        top_p (float): Nucleus sampling probability threshold.
        top_k (int): Top-k sampling parameter.
        max_tokens (Optional[int]): Maximum number of tokens to generate per sequence.
        skip_special_tokens (bool): Whether to remove special tokens from the output.
        repetition_penalty (float): Repetition penalty parameter.
        enable_thinking (bool): Enable thinking mode for LLMs (if supported).

    Raises:
        ValueError: If any parameter is outside its valid range.
    """

    do_sample: bool = field(
        default=True, metadata={"help": "Whether to use sampling vs greedy decoding."}
    )
    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    temperature: float = field(default=0.6, metadata={"help": "Sampling temperature."})
    top_p: float = field(
        default=0.95, metadata={"help": "Nucleus sampling probability threshold."}
    )
    top_k: int = field(default=40, metadata={"help": "Top-k sampling parameter."})
    max_tokens: int = field(
        default=32768, metadata={"help": "The Maximum number of tokens to generate."}
    )
    skip_special_tokens: bool = field(
        default=True, metadata={"help": "Remove special tokens from output."}
    )
    repetition_penalty: float = field(
        default=1.0, metadata={"help": "Repetition penalty parameter."}
    )
    enable_thinking: bool = field(
        default=False, metadata={"help": "Enable thinking mode for LLMs."}
    )

    def __post_init__(self) -> None:
        """
        Validate generation arguments after initialization.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got: {self.temperature}"
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"Top-p must be between 0 and 1, but got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"Top-k must be positive or -1 (disabled), got: {self.top_k}"
            )
        if self.max_tokens <= 0:
            raise ValueError(
                f"Max tokens must be a positive integer, but got {self.max_tokens}."
            )
        if self.n_samples <= 0:
            raise ValueError(
                f"Number of samples must be positive, but got {self.n_samples}."
            )
        if self.repetition_penalty < 0:
            raise ValueError(
                f"Repetition penalty must be non-negative, got: {self.repetition_penalty}"
            )
        if self.temperature <= 0.0:
            self.do_sample = False
            logger.info("Greedy decoding: temperature=0 → do_sample=False")


@dataclass
class VLLMEngineArguments:
    """
    Arguments for configuring the vLLM inference backend.

    This class handles all vLLM-specific configuration including model
    loading, memory management, and parallel processing settings.

    Attributes:
        model_name_or_path (str): Path or name of the model to load.
        trust_remote_code (bool): Whether to trust remote code.
        dtype (str): Data type for model execution (e.g., "fp16", "auto", "bfloat16").
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
        seed (int): Random seed for initialization.
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
            "help": 'Data type for model execution (e.g., "fp16", "auto", "bfloat16").'
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
    seed: int = field(default=0, metadata={"help": "Random seed for initialization."})
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
        valid_dtypes = ["auto", "float16", "float32", "bfloat16", "fp16"]
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
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got: {self.seed}")


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

    Raises:
        ValueError: If any parameter is outside its valid range.
    """

    max_workers: int = field(
        default=128, metadata={"help": "Maximum number of worker threads."}
    )
    base_url: str = field(
        default="https://api.openai.com/v1",
        metadata={"help": "Base URL of VLLM server"},
    )
    model_name: str = field(
        default="gpt-4o", metadata={"help": "Model name of VLLM server"}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for requests to VLLM server."},
    )
    request_timeout: int = field(
        default=99999, metadata={"help": "Timeout for requests to VLLM server."}
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
        metadata={"help": "Tool choice mode: 'none', 'auto', or a specific tool name."},
    )

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

    seed: int = field(default=0, metadata={"help": "Generation seed sent to the API."})

    def __post_init__(self) -> None:
        """Validate all inherited arguments."""
        # Only validate what online mode needs; no vLLM engine args
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        ServerArguments.__post_init__(self)
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got: {self.seed}")


@dataclass
class OfflineInferArguments(
    DataArguments, PromptArguments, GenerationArguments, VLLMEngineArguments
):
    """
    Arguments specific to offline (local vLLM engine) inference.

    This class combines all necessary arguments for offline inference,
    inheriting from DataArguments, PromptArguments, GenerationArguments,
    and VLLMEngineArguments.
    """

    def __post_init__(self) -> None:
        """Validate all inherited arguments."""
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        VLLMEngineArguments.__post_init__(self)


@dataclass
class MCInferConfig:
    """
    Configuration for MC (multiple-choice) inference.

    Used by llmeval/inference/mc.py and parsed from the command
    line by HfArgumentParser; field names map 1:1 to CLI flags
    (e.g. --input_file, --max_workers).

    Attributes:
        input_file (str): Path to the input JSONL file (MC items).
        output_file (str): Path to the output JSONL file (results appended).
        base_url (str): Base URL of the OpenAI-compatible API endpoint.
        model_name (str): Served model name used in requests.
        mode (str): Inference mode: "loglikelihood" or "generate".
        max_workers (int): Number of concurrent worker threads.
        request_timeout (int): Per-request timeout in seconds.
        max_retries (int): Maximum number of retries for transient failures.
        max_tokens (int): Maximum number of tokens to generate (generate mode).
        n_samples (int): Number of generations per MC prompt in generate mode.
        loglikelihood_mode (str): ``auto``, ``continuation`` or ``first_token``.
        temperature (float): Sampling temperature (0.0 = deterministic).
        system_prompt_type (str): System prompt template key ("empty" disables).
        tool_choice (str): Tool calling mode: "none", "auto", or a tool name.
        n_shot (int): Few-shot example count (0 = zero-shot).
        few_shot_file (str): Dev file for few-shot examples
            (required when n_shot > 0; never falls back to the evaluation set).
        api_key (str): API key; defaults to the OPENAI_API_KEY env var.

    Raises:
        ValueError: If any parameter is outside its valid range.
    """

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
    mode: str = field(
        default="loglikelihood",
        metadata={"help": "Inference mode: 'loglikelihood' or 'generate'."},
    )
    max_workers: int = field(
        default=32, metadata={"help": "Number of concurrent worker threads."}
    )
    request_timeout: int = field(
        default=300, metadata={"help": "Per-request timeout in seconds."}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for transient failures."},
    )
    max_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate (generate mode)."},
    )
    n_samples: int = field(
        default=1,
        metadata={"help": "Number of generations per prompt in generate mode."},
    )
    loglikelihood_mode: str = field(
        default="first_token",
        metadata={
            "help": (
                "MC scoring mode: first_token (default), continuation, or "
                "auto (compatibility alias for first_token)."
            )
        },
    )
    temperature: float = field(
        default=0.0, metadata={"help": "Sampling temperature (0.0 = deterministic)."}
    )
    system_prompt_type: str = field(
        default="empty",
        metadata={"help": "System prompt template key ('empty' disables it)."},
    )
    tool_choice: str = field(
        default="none",
        metadata={"help": "Tool calling mode: 'none', 'auto', or a tool name."},
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
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"),
        metadata={"help": "API key (default: OPENAI_API_KEY env var)."},
    )
    organization: str | None = field(
        default=None, metadata={"help": "Optional OpenAI organization ID."}
    )
    input_key: str = field(
        default="prompt",
        metadata={"help": "Field name for the input prompt text in dataset."},
    )
    label_key: str = field(
        default="answer",
        metadata={"help": "Field name for the gold label/answer in dataset."},
    )
    response_key: str = field(
        default="gen",
        metadata={"help": "Field name for model generation results in dataset."},
    )
    seed: int = field(
        default=0, metadata={"help": "Generation and few-shot sampling seed."}
    )
    repair_resume: bool = field(
        default=False,
        metadata={
            "help": (
                "Ignore only an unterminated invalid final line in an existing "
                "resume JSONL file."
            )
        },
    )

    def __post_init__(self) -> None:
        """
        Validate MC inference arguments after initialization.

        Note: input_file/output_file emptiness and existence are validated at
        pipeline start (MCRunner.run), not here, so a default-constructed
        config stays usable for inspection and testing.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if self.mode not in ("loglikelihood", "generate"):
            raise ValueError(
                f"mode must be one of {('loglikelihood', 'generate')}, got: {self.mode!r}"
            )
        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Base URL must start with http:// or https://, but got {self.base_url}"
            )
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got: {self.max_workers}")
        if self.request_timeout <= 0:
            raise ValueError(
                f"request_timeout must be positive, got: {self.request_timeout}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got: {self.max_retries}"
            )
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got: {self.max_tokens}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got: {self.n_samples}")
        if self.loglikelihood_mode not in ("auto", "continuation", "first_token"):
            raise ValueError(
                "loglikelihood_mode must be one of ('auto', 'continuation', "
                f"'first_token'), got: {self.loglikelihood_mode!r}"
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got: {self.temperature}"
            )
        if self.n_shot < 0:
            raise ValueError(f"n_shot must be non-negative, got: {self.n_shot}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got: {self.seed}")
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
        input_key (str): Key for input text in dataset.
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
    input_key: str = field(
        default="prompt", metadata={"help": "Key for input text in dataset."}
    )
    label_key: str = field(
        default="answer", metadata={"help": "Key for target/label text in dataset."}
    )
    response_key: str = field(
        default="gen", metadata={"help": "Key for model generated text."}
    )
    output_schema: str = field(
        default="compact",
        metadata={"help": "Per-item result schema: compact or debug."},
    )
    expected_samples: int = field(
        default=0,
        metadata={
            "help": "Expected generations per problem for multi-sample metrics (0=from output)."
        },
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
        if self.output_schema not in ("compact", "debug"):
            raise ValueError(
                "output_schema must be one of ('compact', 'debug'), "
                f"got: {self.output_schema!r}"
            )
        if self.expected_samples < 0:
            raise ValueError(
                f"expected_samples must be non-negative, got {self.expected_samples}"
            )
        if self.bootstrap_samples < 0:
            raise ValueError(
                f"bootstrap_samples must be non-negative, got {self.bootstrap_samples}"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            )
