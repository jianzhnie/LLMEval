"""
Configuration classes for a large language model evaluation pipeline.

This module defines a set of dataclasses to handle and validate all
the necessary arguments for a complete evaluation run. The arguments are
categorized into data, prompt formatting, generation parameters, and
vLLM-specific settings.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

logger = init_logger(__name__)


@dataclass
class DataArguments:
    """
    Arguments for configuring the dataset and data loading.

    Attributes:
        dataset_dir (str): Path to the directory containing dataset files.
        dataset_name (str): The identifier for the specific dataset to use.
        split (DataSplit): The dataset split to load (e.g., 'train', 'test').
        cache_dir (str): Path to the directory for caching models and data.
        batch_size (int): The number of samples to process in each batch.
    """
    dataset_dir: str = field(
        default='./data',
        metadata={'help': 'Directory containing the dataset.'})
    dataset_name: str = field(default='math',
                              metadata={'help': 'Dataset identifier.'})
    split: Literal['train', 'test', 'validation', 'dev'] = field(
        default='test',
        metadata={
            'help': 'Dataset split to use (e.g., train, test, validation).'
        })
    input_file: str = field(
        default='input.jsonl',
        metadata={'help': 'Input JSONL file containing prompts.'})
    cache_dir: str = field(default='./cache',
                           metadata={'help': 'Cache directory for models.'})
    batch_size: int = field(default=128,
                            metadata={'help': 'Batch size for data loading.'})

    def __post_init__(self) -> None:
        """Validate data arguments after initialization."""
        if self.batch_size <= 0:
            raise ValueError(
                f'Batch size must be a positive integer, but got {self.batch_size}.'
            )


@dataclass
class PromptArguments:
    """
    Arguments for configuring prompt templates and formatting.

    Attributes:
        prompt_type (str): The type of prompt format to apply (e.g., 'qwen-base').
        input_key (str): The key in the dataset dictionary for the input text.
        label_key (str): The key for the target/label text in the dataset.
    """
    input_key: str = field(default='prompt',
                           metadata={'help': 'Key for input text in dataset.'})
    label_key: str = field(
        default='answer',
        metadata={'help': 'Key for target/label text in dataset.'})
    system_prompt_type: str = field(
        default='empty',
        metadata={
            'help':
            'Optional system prompt to prepend to each input (if applicable).'
        })

    def __post_init__(self) -> None:
        """Validate prompt arguments after initialization."""
        if not self.input_key:
            raise ValueError('Input key must be a non-empty string.')
        if not self.label_key:
            raise ValueError('Label key must be a non-empty string.')

        if self.system_prompt_type is not None and self.system_prompt_type not in SYSTEM_PROMPT_FACTORY:
            raise ValueError(
                f'Invalid system prompt type: {self.system_prompt_type}. '
                f'Valid options are: {list(SYSTEM_PROMPT_FACTORY.keys())}')
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(self.system_prompt_type)
        logger.info(
            f'Using system_prompt_type: {self.system_prompt_type}, content: {self.system_prompt}'
        )
        logger.info(
            'If you want to customize the system prompt, please modify the '
            'SYSTEM_PROMPT_FACTORY in llmeval/utils/template.py')


@dataclass
class GenerationArguments:
    """
    Arguments for controlling the text generation process.

    Attributes:
        do_sample (bool): Whether to use sampling or greedy decoding.
        n_sampling (int): Number of sequences to generate per prompt.
        temperature (float): Controls the randomness of sampling. Higher values
                             lead to more diverse outputs.
        top_p (float): Nucleus sampling probability threshold.
        top_k (int): Top-k sampling parameter.
        max_tokens (int): Maximum number of tokens to generate per sequence.
        skip_special_tokens (bool): Whether to remove special tokens from the output.
    """
    do_sample: bool = field(
        default=True,
        metadata={'help': 'Whether to use sampling vs greedy decoding.'})
    n_sampling: int = field(
        default=64,
        metadata={'help': 'Number of sequences to generate per prompt.'})
    temperature: float = field(default=0.6,
                               metadata={'help': 'Sampling temperature.'})
    top_p: float = field(
        default=0.95,
        metadata={'help': 'Nucleus sampling probability threshold.'})
    top_k: int = field(default=40,
                       metadata={'help': 'Top-k sampling parameter.'})
    max_tokens: int = field(
        default=32768,
        metadata={'help': 'Maximum number of tokens to generate.'})
    skip_special_tokens: bool = field(
        default=True, metadata={'help': 'Remove special tokens from output.'})
    repeat_penalty: float = field(
        default=1.0, metadata={'help': 'Repeat penalty parameter.'})
    enable_thinking: bool = field(
        default=False, metadata={'help': 'Enable thinking mode for LLMs.'})

    skip_if_empty: bool = field(
        default=True,
        metadata={'help': 'Skip processing if response is empty.'})
    max_retries: int = field(
        default=3,
        metadata={'help': 'Maximum number of retries for API calls.'})

    def __post_init__(self) -> None:
        """Validate generation arguments after initialization."""
        if self.temperature < 0:
            raise ValueError(
                f'Temperature must be non-negative, but got {self.temperature}.'
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(
                f'Top-p must be between 0 and 1, but got {self.top_p}.')
        if self.top_k < 0:
            raise ValueError(
                f'Top-k must be non-negative, but got {self.top_k}.')
        if self.max_tokens <= 0:
            raise ValueError(
                f'Max tokens must be a positive integer, but got {self.max_tokens}.'
            )
        if self.n_sampling <= 0:
            raise ValueError(
                f'Number of samples must be positive, but got {self.n_sampling}.'
            )


@dataclass
class VLLMEngineArguments:
    """
    Arguments for configuring the vLLM inference backend.

    Attributes:
        model_name_or_path (str): Path or name of the model to load.
        max_model_len (int): Maximum context length for the model.
        rope_scaling (Optional[dict]): RoPE scaling configuration as a dictionary.
        gpu_memory_utilization (float): Target GPU memory usage (0-1).
        tensor_parallel_size (int): Number of GPUs for tensor parallelism.
        enable_prefix_caching (bool): Whether to enable KV cache prefix optimization.
        max_num_batched_tokens (Optional[int]): Maximum number of tokens per batch.
        max_num_seqs (Optional[int]): Maximum number of parallel sequences.
        seed (int): The random seed for initialization.
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
    rope_scaling: Optional[str] = field(
        default=
        '{"rope_type":"yarn","factor":2.5,"original_max_position_embeddings":32768}',
        metadata={'help': 'RoPE scaling configuration in JSON format.'})
    gpu_memory_utilization: float = field(
        default=0.9, metadata={'help': 'Target GPU memory utilization (0-1).'})
    tensor_parallel_size: int = field(
        default=4,
        metadata={'help': 'Number of GPUs to use for tensor parallelism.'})
    pipeline_parallel_size: int = field(
        default=1,
        metadata={'help': 'Number of GPUs to use for pipeline parallelism.'})
    enable_chunked_prefill: bool = field(
        default=True,
        metadata={
            'help':
            'Enable chunked prefill to reduce memory usage during generation.'
        })
    enable_prefix_caching: bool = field(
        default=True,
        metadata={'help': 'Enable KV cache prefix optimization.'})
    max_num_batched_tokens: Optional[int] = field(
        default=None, metadata={'help': 'Maximum tokens per batch.'})
    max_num_seqs: Optional[int] = field(
        default=128, metadata={'help': 'Maximum parallel sequences.'})
    enforce_eager: bool = field(
        default=False,
        metadata={'help': 'Enforce eager execution for debugging purposes.'})
    seed: int = field(default=0,
                      metadata={'help': 'Random seed for initialization.'})

    def __post_init__(self) -> None:
        """Validate vLLM arguments and parse JSON string."""
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f'GPU memory utilization must be between 0 and 1, but got {self.gpu_memory_utilization}.'
            )

        # Parse the JSON string for rope_scaling.
        # This makes it easier to work with as a dictionary later.
        if self.rope_scaling:
            try:
                self.rope_scaling = json.loads(self.rope_scaling)
                logger.info(
                    f'Successfully parsed rope_scaling: {self.rope_scaling}')
            except json.JSONDecodeError as e:
                raise ValueError(
                    f'Invalid JSON string for rope_scaling: {self.rope_scaling}. Error: {e}'
                ) from e


@dataclass
class ServerArguments:
    """
    Arguments for configuring the vLLM server.

    Attributes:
        host (str): The hostname or IP address for the server.
        port (int): The port number for the server.
        num_workers (int): Number of worker processes to spawn.
        log_level (str): Logging level for the server.
    """
    max_workers: int = field(
        default=128, metadata={'help': 'Maximum number of worker threads.'})
    base_url: str = field(default='https://api.openai.com/v1',
                          metadata={'help': 'Base URL of VLLM server'})
    model_name: str = field(default='gpt-4o',
                            metadata={'help': 'Model name of VLLM server'})
    request_timeout: int = field(
        default=600, metadata={'help': 'Timeout for requests to VLLM server.'})

    def __post_init__(self) -> None:
        """Validate server arguments after initialization."""
        if self.max_workers <= 0:
            raise ValueError(
                f'Maximum number of worker threads must be a positive integer, but got {self.max_workers}.'
            )
        if self.request_timeout <= 0:
            raise ValueError(
                f'Request timeout must be a positive integer, but got {self.request_timeout}.'
            )


@dataclass
class EvaluationArguments(DataArguments, PromptArguments, GenerationArguments,
                          VLLMEngineArguments, ServerArguments):
    """
    Master configuration class for all evaluation arguments.

    This class inherits from all other argument classes to provide a single,
    convenient object for configuring an entire evaluation task.

    Attributes:
        task (str): Name of the specific evaluation task being run.
        output_dir (str): Path to the directory for saving results and outputs.
    """
    # Core evaluation settings
    task: str = field(default='math',
                      metadata={'help': 'Name of the evaluation task.'})
    # Output settings
    output_dir: str = field(
        default='./output',
        metadata={'help': 'Directory to save output results.'})
    output_file: str = field(
        default='output.jsonl',
        metadata={'help': 'Output JSONL file to save results.'})

    def __post_init__(self) -> None:
        """Validate and prepare evaluation arguments."""
        # Call post-init methods of parent classes
        DataArguments.__post_init__(self)
        PromptArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        VLLMEngineArguments.__post_init__(self)
        ServerArguments.__post_init__(self)

        # Ensure the output directory exists
        path_output_dir = Path(self.output_dir)
        path_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Created output directory at {path_output_dir}')

        # Adjust generation parameters based on temperature
        if self.temperature <= 0.0:
            self.do_sample = False
            self.top_p = 1.0
            logger.warning(
                'Temperature is 0, setting do_sample=False and top_p=1.0 for greedy decoding.'
            )


# Example usage
def main() -> None:
    """Demonstrates how to initialize and use the EvaluationArguments class."""
    try:
        from transformers import HfArgumentParser
    except ImportError:
        raise ImportError(
            'Please install the transformers library to use HfArgumentParser: pip install transformers'
        )

    # Create an HfArgumentParser instance for the EvaluationArguments class.
    # It automatically reads all fields and metadata from the dataclass.
    parser = HfArgumentParser(EvaluationArguments)

    # Parse the command-line arguments and get an instance of EvaluationArguments.
    # The return value is a tuple, we only need the first element.
    eval_args, = parser.parse_args_into_dataclasses()

    logger.info(
        'Initializing EvaluationArguments with parsed command line arguments...'
    )
    print('\n--- Parsed Arguments ---')
    print(eval_args)


if __name__ == '__main__':
    main()
