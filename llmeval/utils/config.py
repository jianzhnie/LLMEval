from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llmeval.utils.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DataArguments:
    """Arguments for dataset configuration and loading.

    Attributes:
        data_dir: Directory containing the dataset files
        data_name: Identifier for the dataset (e.g., 'math', 'aime')
        batch_size: Number of samples to process in each batch
    """

    data_dir: str = field(
        default='./data',
        metadata={'help': 'Directory containing the dataset.'})
    data_name: str = field(default='math',
                           metadata={'help': 'Dataset identifier.'})
    split: str = field(
        default='test',
        metadata={'help': 'Dataset split to use (e.g., train, test).'})
    cache_dir: str = field(default='./cache',
                           metadata={'help': 'Cache directory for models.'})

    batch_size: int = field(default=128,
                            metadata={'help': 'Batch size for data loading.'})

    def __post_init__(self) -> None:
        """Validate data arguments after initialization."""
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')


@dataclass
class PromptArguments:
    """Arguments for prompt configuration and formatting.

    Attributes:
        prompt_type: Type of prompt template to use
        prompt_file_path: Directory containing prompt template files
        surround_with_messages: Whether to wrap prompts in chat format
        use_few_shot: Whether to use few-shot examples
        n_shot: Number of few-shot examples to include
        input_key: Key for input text in the dataset
        label_key: Key for label/target text in the dataset
        input_template: Optional template for formatting input text
    """

    prompt_type: str = field(default='qwen-base',
                             metadata={'help': 'Type of prompt format used.'})
    input_key: str = field(default='question',
                           metadata={'help': 'Key for input text in dataset.'})
    label_key: str = field(
        default='solution',
        metadata={'help': 'Key for target/label text in dataset.'})


@dataclass
class GenerationArguments:
    """Arguments for controlling text generation.

    Attributes:
        do_sample: Whether to use sampling instead of greedy decoding
        n_sampling: Number of sequences to generate for each prompt
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum number of tokens to generate
        skip_special_tokens: Whether to remove special tokens from output
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
    top_k: int = field(default=50,
                       metadata={'help': 'Top-k sampling parameter.'})
    max_tokens: int = field(
        default=32768,
        metadata={'help': 'Maximum number of tokens to generate.'})
    skip_special_tokens: bool = field(
        default=True, metadata={'help': 'Remove special tokens from output.'})

    def __post_init__(self) -> None:
        """Validate generation arguments after initialization."""
        if self.temperature < 0:
            raise ValueError('temperature must be non-negative')
        if not 0 <= self.top_p <= 1:
            raise ValueError('top_p must be between 0 and 1')
        if self.top_k < 0:
            raise ValueError('top_k must be non-negative')
        if self.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        if self.n_sampling < 1:
            raise ValueError('n_sampling must be positive')


@dataclass
class VLLMArguments:
    """Arguments specific to vLLM inference backend.

    Attributes:
        gpu_memory_utilization: Target GPU memory usage (0-1)
        enable_prefix_caching: Whether to cache KV prefix for efficiency
        swap_space: Size of CPU swap space in GB
        block_size: Size of tensor parallel blocks
        quantization: Quantization method to use
        max_num_batched_tokens: Max tokens per batch
        max_num_seqs: Max sequences to process in parallel
        disable_custom_kernels: Whether to disable CUDA kernels
    """

    model_name_or_path: str = field(
        default='Qwen/Qwen2.5-7B',
        metadata={'help': 'Path to the model directory.'})
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
    enable_prefix_caching: bool = field(
        default=True,
        metadata={'help': 'Enable KV cache prefix optimization.'})
    max_num_batched_tokens: Optional[int] = field(
        default=None, metadata={'help': 'Max tokens per batch.'})
    max_num_seqs: Optional[int] = field(
        default=128, metadata={'help': 'Max parallel sequences.'})

    seed: int = field(default=0,
                      metadata={'help': 'Random seed for initialization.'})

    def __post_init__(self) -> None:
        """Validate vLLM arguments after initialization."""
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError('gpu_memory_utilization must be between 0 and 1')


@dataclass
class EvaluationArguments(DataArguments, PromptArguments, GenerationArguments,
                          VLLMArguments):
    """Master configuration class to store all evaluation arguments."""

    # Core evaluation settings
    task: str = field(metadata={'help': 'Name of the evaluation task.'})
    # Output settings
    output_dir: str = field(
        default='./outputs',
        metadata={'help': 'Directory to save output results.'})

    def __post_init__(self) -> None:
        """Validate and adjust evaluation arguments after initialization."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Adjust generation parameters based on temperature
        if self.temperature == 0:
            self.top_p = 1.0
            self.do_sample = False
