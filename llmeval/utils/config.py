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
- Specialized inference modes (online, offline, verifier)

All configuration classes include validation logic to ensure parameter
consistency and prevent runtime errors.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import HfArgumentParser

from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY
from llmeval.utils.verifier_template import VERIFY_PROMPT_FACTORY

__all__ = [
    'DataArguments',
    'PromptArguments',
    'GenerationArguments',
    'VLLMEngineArguments',
    'ServerArguments',
    'OnlineInferArguments',
    'OfflineInferArguments',
    'VerifierInferArguments',
    'EvalTaskArguments',
]

logger = init_logger('eval_config')


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
        task (str): Name of the evaluation task (e.g., 'aime24').
        batch_size (int): The number of samples to process in each batch.
    """
    input_file: str = field(
        default='input.jsonl',
        metadata={'help': 'Input JSONL file containing prompts.'})
    cache_dir: str = field(default='/home/jianzhnie/llmtuner/hfhub/cache',
                           metadata={'help': 'Cache directory for models.'})
    output_file: str = field(
        default='output.jsonl',
        metadata={'help': 'Output JSONL file to save results.'})
    task: str = field(default='aime24',
                      metadata={'help': 'Name of the evaluation task.'})
    batch_size: int = field(default=128,
                            metadata={'help': 'Batch size for data loading.'})

    def __post_init__(self) -> None:
        """
        Validate data arguments after initialization.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        if self.batch_size <= 0:
            raise ValueError(
                f'Batch size must be a positive integer, but got {self.batch_size}.'
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
    input_key: str = field(default='prompt',
                           metadata={'help': 'Key for input text in dataset.'})
    label_key: str = field(
        default='answer',
        metadata={'help': 'Key for target/label text in dataset.'})
    response_key: str = field(
        default='gen', metadata={'help': 'Key for model generated text.'})
    system_prompt_type: str = field(
        default='empty',
        metadata={
            'help':
            'Optional system prompt to prepend to each input (if applicable).'
        })
    # Computed value based on system_prompt_type; not settable via CLI
    system_prompt: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Validate prompt arguments and resolve system prompt.

        Raises:
            ValueError: If input_key or label_key is empty, or if system_prompt_type
                       is not found in SYSTEM_PROMPT_FACTORY.
        """
        if not self.input_key:
            raise ValueError('Input key must be a non-empty string.')
        if not self.label_key:
            raise ValueError('Label key must be a non-empty string.')

        if (self.system_prompt_type is not None
                and self.system_prompt_type not in SYSTEM_PROMPT_FACTORY):
            raise ValueError(
                f'Invalid system prompt type: {self.system_prompt_type}. '
                f'Valid options are: {list(SYSTEM_PROMPT_FACTORY.keys())}')
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(self.system_prompt_type)
        logger.info(f'Using system_prompt_type: {self.system_prompt_type}, '
                    f'content: {self.system_prompt}')
        logger.info(
            'If you want to customize the system prompt, please modify the '
            'SYSTEM_PROMPT_FACTORY in llmeval/utils/template.py')


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
        default=True,
        metadata={'help': 'Whether to use sampling vs greedy decoding.'})
    n_samples: int = field(
        default=1,
        metadata={'help': 'Number of sequences to generate per prompt.'})
    temperature: float = field(default=0.6,
                               metadata={'help': 'Sampling temperature.'})
    top_p: float = field(
        default=0.95,
        metadata={'help': 'Nucleus sampling probability threshold.'})
    top_k: int = field(default=40,
                       metadata={'help': 'Top-k sampling parameter.'})
    max_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'The Maximum number of tokens to generate.'})
    skip_special_tokens: bool = field(
        default=True, metadata={'help': 'Remove special tokens from output.'})
    repetition_penalty: float = field(
        default=1.0, metadata={'help': 'Repetition penalty parameter.'})
    enable_thinking: bool = field(
        default=False, metadata={'help': 'Enable thinking mode for LLMs.'})

    def __post_init__(self) -> None:
        """
        Validate generation arguments after initialization.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f'Temperature must be between 0.0 and 2.0, got: {self.temperature}'
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(
                f'Top-p must be between 0 and 1, but got {self.top_p}.')
        if self.top_k <= 0:
            raise ValueError(f'Top-k must be positive, got: {self.top_k}')
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(
                f'Max tokens must be a positive integer when specified, '
                f'but got {self.max_tokens}.')
        if self.n_samples <= 0:
            raise ValueError(
                f'Number of samples must be positive, but got {self.n_samples}.'
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
        dtype (str): Data type for model execution (e.g., "fp16", "auto").
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

    Raises:
        ValueError: If any parameter is outside its valid range or if
                   rope_scaling contains invalid JSON.
    """
    model_name_or_path: str = field(
        default='Qwen/Qwen2.5-7B',
        metadata={'help': 'Path to the model directory.'})
    trust_remote_code: bool = field(
        default=True, metadata={'help': 'Whether to trust remote code.'})
    dtype: str = field(
        default='auto',
        metadata={
            'help': 'Data type for model execution (e.g., "fp16", "auto").'
        },
    )
    max_model_len: int = field(
        default=32768,
        metadata={'help': 'Maximum sequence length for the model.'})
    rope_scaling: str = field(
        default='{}',
        metadata={'help': 'RoPE scaling configuration as a JSON string.'})
    # Parsed representation; not settable via CLI
    rope_scaling_dict: Optional[Dict[str, Any]] = field(init=False,
                                                        default=None)
    gpu_memory_utilization: float = field(
        default=0.9, metadata={'help': 'Target GPU memory utilization (0-1).'})
    tensor_parallel_size: int = field(
        default=1,
        metadata={'help': 'Number of GPUs to use for tensor parallelism.'})
    pipeline_parallel_size: int = field(
        default=1,
        metadata={'help': 'Number of GPUs to use for pipeline parallelism.'})
    enable_chunked_prefill: bool = field(
        default=False,
        metadata={
            'help':
            'Enable chunked prefill to reduce memory usage during generation.'
        })
    enable_prefix_caching: bool = field(
        default=False,
        metadata={'help': 'Enable KV cache prefix optimization.'})
    max_num_batched_tokens: Optional[int] = field(
        default=512000, metadata={'help': 'Maximum tokens per batch.'})
    max_num_seqs: Optional[int] = field(
        default=4096, metadata={'help': 'Maximum parallel sequences.'})
    enforce_eager: bool = field(
        default=True,
        metadata={'help': 'Enforce eager execution for debugging purposes.'})
    seed: int = field(default=0,
                      metadata={'help': 'Random seed for initialization.'})

    def __post_init__(self) -> None:
        """
        Validate vLLM arguments and parse rope_scaling value.

        Raises:
            ValueError: If any parameter is outside its valid range or if
                       rope_scaling contains invalid JSON.
        """
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f'GPU memory utilization must be between 0 and 1, '
                f'but got {self.gpu_memory_utilization}.')
        if self.max_model_len < 0:
            raise ValueError(
                f'Max model length must be positive, but got {self.max_model_len}.'
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(f'Tensor parallel size must be at least 1, '
                             f'but got {self.tensor_parallel_size}.')
        if self.pipeline_parallel_size < 1:
            raise ValueError(f'Pipeline parallel size must be at least 1, '
                             f'but got {self.pipeline_parallel_size}.')

        # Parse rope_scaling into rope_scaling_dict, keeping the original field as string.
        text = (self.rope_scaling or '').strip()
        if not text:
            self.rope_scaling_dict = None
        else:
            try:
                self.rope_scaling_dict = json.loads(text)
                logger.info(
                    f'Successfully parsed rope_scaling: {self.rope_scaling_dict}'
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f'Invalid JSON string for rope_scaling: {self.rope_scaling}. '
                    f'Error: {e}') from e


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

    Raises:
        ValueError: If any parameter is outside its valid range.
    """
    max_workers: int = field(
        default=128, metadata={'help': 'Maximum number of worker threads.'})
    base_url: str = field(default='https://api.openai.com/v1',
                          metadata={'help': 'Base URL of VLLM server'})
    model_name: str = field(default='gpt-4o',
                            metadata={'help': 'Model name of VLLM server'})
    request_timeout: int = field(
        default=60, metadata={'help': 'Timeout for requests to VLLM server.'})

    def __post_init__(self) -> None:
        """
        Validate server arguments after initialization.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if self.max_workers <= 0:
            raise ValueError(
                f'Maximum number of worker threads must be a positive integer, '
                f'but got {self.max_workers}.')
        if self.request_timeout <= 0:
            raise ValueError(f'Request timeout must be a positive integer, '
                             f'but got {self.request_timeout}.')


@dataclass
class OnlineInferArguments(DataArguments, PromptArguments, GenerationArguments,
                           ServerArguments):
    """
    Arguments specific to online (OpenAI-compatible API) inference.

    This class combines all necessary arguments for online inference,
    inheriting from DataArguments, PromptArguments, GenerationArguments,
    and ServerArguments. It includes special handling for greedy decoding
    when temperature is set to 0.
    """

    def __post_init__(self) -> None:
        """
        Validate all inherited arguments and handle greedy decoding.

        When temperature is 0, automatically sets do_sample=False and top_p=1.0
        for greedy decoding behavior.
        """
        # Only validate what online mode needs; no vLLM engine args
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        ServerArguments.__post_init__(self)

        if self.temperature <= 0.0:
            self.do_sample = False
            self.top_p = 1.0
            logger.warning(
                'Temperature is 0, setting do_sample=False and top_p=1.0 '
                'for greedy decoding.')


@dataclass
class OfflineInferArguments(DataArguments, PromptArguments,
                            GenerationArguments, VLLMEngineArguments):
    """
    Arguments specific to offline (local vLLM engine) inference.

    This class combines all necessary arguments for offline inference,
    inheriting from DataArguments, PromptArguments, GenerationArguments,
    and VLLMEngineArguments. It includes special handling for greedy decoding
    when temperature is set to 0.
    """

    def __post_init__(self) -> None:
        """
        Validate all inherited arguments and handle greedy decoding.

        When temperature is 0, automatically sets do_sample=False and top_p=1.0
        for greedy decoding behavior.
        """
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        VLLMEngineArguments.__post_init__(self)

        if self.temperature <= 0.0:
            self.do_sample = False
            self.top_p = 1.0
            logger.warning(
                'Temperature is 0, setting do_sample=False and top_p=1.0 '
                'for greedy decoding.')


@dataclass
class VerifierInferArguments(DataArguments, PromptArguments,
                             GenerationArguments, VLLMEngineArguments):
    """
    Arguments specific to compass verifier inference using local vLLM engine.

    This class extends offline inference with verifier-specific parameters
    for answer verification tasks. It includes prompt template selection
    and data handling options.

    Attributes:
        verifier_prompt_type: The type of verifier prompt to use.
        keep_origin_data: Whether to keep the original data in output.
        verifier_prompt: The resolved verifier prompt text (computed automatically).

    Raises:
        ValueError: If verifier_prompt_type is not found in VERIFY_PROMPT_FACTORY.
    """
    verifier_prompt_type: str = field(
        default='fdd_prompt_cursor',
        metadata={
            'help':
            'The type of verifier prompt to use. '
            'Options: compass_verifier, compass_verifier_with_reasoning'
        },
    )
    keep_origin_data: bool = field(
        default=False,
        metadata={'help': 'Will keep the original data or not.'})
    # Computed value based on verifier_prompt_type; not settable via CLI
    verifier_prompt: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Validate all inherited arguments and resolve verifier prompt.

        When temperature is 0, automatically sets do_sample=False and top_p=1.0
        for greedy decoding behavior.

        Raises:
            ValueError: If verifier_prompt_type is not found in VERIFY_PROMPT_FACTORY.
        """
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        VLLMEngineArguments.__post_init__(self)

        if self.temperature <= 0.0:
            self.do_sample = False
            self.top_p = 1.0
            logger.warning(
                'Temperature is 0, setting do_sample=False and top_p=1.0 '
                'for greedy decoding.')

        # Validate and resolve verifier prompt
        if (self.verifier_prompt_type is not None
                and self.verifier_prompt_type not in VERIFY_PROMPT_FACTORY):
            raise ValueError(
                f'Invalid verifier prompt type: {self.verifier_prompt_type}. '
                f'Valid options are: {list(VERIFY_PROMPT_FACTORY.keys())}')

        self.verifier_prompt = VERIFY_PROMPT_FACTORY.get(
            self.verifier_prompt_type)
        logger.info(
            f'Using verifier_prompt_type: {self.verifier_prompt_type}, '
            f'content: {self.verifier_prompt}')
        logger.info(
            'If you want to customize the verifier prompt, please modify the '
            'VERIFY_PROMPT_FACTORY in llmeval/utils/verifier_template.py')


@dataclass
class EvalTaskArguments:
    """
    Arguments for task-specific evaluation configuration.

    This class handles the configuration parameters for evaluating model outputs
    on specific tasks like math problems or code benchmarks.

    Attributes:
        input_path (str): Path to the input JSONL file containing evaluation data.
        cache_path (str): Directory path for saving cached results.
        max_workers (int): Maximum number of worker threads for parallel processing.
        task_name (str): Name of the evaluation task to run.
            Must be one of: ['math_opensource/aime24', 'math_opensource/aime25',
                           'livecodebench', 'ifeval']
    """
    input_path: str = field(
        metadata={
            'help': 'Path to the input JSONL file containing evaluation data.'
        })
    cache_path: str = field(
        metadata={'help': 'Directory path for saving cached results.'})
    max_workers: int = field(
        default=128,
        metadata={
            'help': 'Maximum number of worker threads for parallel processing.'
        })
    task_name: str = field(
        metadata={
            'help':
            'Task must be in [math_opensource/aime24, math_opensource/aime25, '
            'livecodebench, ifeval].'
        })
    input_key: str = field(default='prompt',
                           metadata={'help': 'Key for input text in dataset.'})
    label_key: str = field(
        default='answer',
        metadata={'help': 'Key for target/label text in dataset.'})
    response_key: str = field(
        default='gen', metadata={'help': 'Key for model generated text.'})
    VALID_TASKS: List[str] = [
        'math_opensource/aime24', 'math_opensource/aime25', 'livecodebench',
        'ifeval'
    ]

    def __post_init__(self) -> None:
        """
        Validate evaluation task arguments after initialization.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not self.input_path:
            raise ValueError('input_path is required')
        if not self.cache_path:
            raise ValueError('cache_path is required')
        if self.max_workers <= 0:
            raise ValueError(
                f'max_workers must be positive, got {self.max_workers}')

        if self.task_name not in self.VALID_TASKS:
            raise ValueError(f'task_name must be one of {self.VALID_TASKS}, '
                             f'got {self.task_name}')


# Example usage
def main() -> None:
    """
    Example of using the configuration classes for argument parsing.

    This function demonstrates how to:
    1. Parse command-line arguments into strongly-typed dataclasses
    2. Validate configuration parameters
    3. Handle multiple configuration types (eval, online, offline)
    4. Process the parsed arguments

    Raises:
        ImportError: If transformers library is not installed
        ValueError: If required arguments are missing or invalid
    """
    # Create parser instances for different argument types
    parser = HfArgumentParser((EvalTaskArguments, ))

    # Parse command-line arguments into respective dataclasses
    eval_args = parser.parse_args_into_dataclasses()

    # Log the parsed arguments
    logger.info('Initializing with parsed command line arguments...')
    logger.info('\n=== Evaluation Task Arguments ===')
    logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

    # Example of how to use the parsed arguments
    logger.info(f'\nConfigured for task: {eval_args.task_name}')
    logger.info(f'Processing input file: {eval_args.input_path}')


if __name__ == '__main__':
    main()
