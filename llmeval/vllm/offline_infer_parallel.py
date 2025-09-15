"""Multi-node Data Parallel Offline vLLM inference runner.

This module extends the offline inference functionality with multi-node data parallel support:
- Load a line-delimited JSON dataset
- Resume generation per unique prompt up to a requested sample count
- Convert records into vLLM chat message format
- Run batched chat inference with data parallelism across multiple nodes/GPUs
- Persist unified results incrementally for robustness

Usage:
Single node:
    python llmeval/vllm/offline_infer_data_parallel.py \
            --model-name-or-path="Qwen/Qwen2.5-7B" \
            --input-file="input.jsonl" \
            --output-file="output.jsonl" \
            --dp-size=2 \
            --tp-size=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python llmeval/vllm/offline_infer_data_parallel.py \
                    --model-name-or-path="Qwen/Qwen2.5-7B" \
                    --input-file="input.jsonl" \
                    --output-file="output.jsonl" \
                    --dp-size=4 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python llmeval/vllm/offline_infer_data_parallel.py \
                    --model-name-or-path="Qwen/Qwen2.5-7B" \
                    --input-file="input.jsonl" \
                    --output-file="output.jsonl" \
                    --dp-size=4 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import collections
import copy
import json
import logging
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from multiprocessing import Process
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

# Initialize logger
logger = init_logger('vllm_dp_infer', logging.INFO)

DEFAULT_INPUT_KEY = 'prompt'
DEFAULT_LABEL_KEY = 'answer'
DEFAULT_RESPONSE_KEY = 'gen'


@dataclass
class DataParallelArguments:
    """Arguments for data parallel configuration."""

    dp_size: int = field(
        default=1,
        metadata={
            'help':
            'Data parallel size (total number of data parallel processes)'
        })
    tp_size: int = field(
        default=1,
        metadata={
            'help': 'Tensor parallel size (GPUs per data parallel process)'
        })
    node_size: int = field(default=1,
                           metadata={'help': 'Total number of nodes'})
    node_rank: int = field(default=0,
                           metadata={'help': 'Rank of the current node'})
    master_addr: str = field(
        default='',
        metadata={'help': 'Master node IP address for multi-node setup'})
    master_port: int = field(
        default=0, metadata={'help': 'Master node port for multi-node setup'})
    timeout: int = field(
        default=300,
        metadata={'help': 'Timeout in seconds for process completion'})

    def __post_init__(self):
        """Validate data parallel arguments."""
        if self.dp_size <= 0:
            raise ValueError(f'dp_size must be positive, got {self.dp_size}')
        if self.tp_size <= 0:
            raise ValueError(f'tp_size must be positive, got {self.tp_size}')
        if self.node_size <= 0:
            raise ValueError(
                f'node_size must be positive, got {self.node_size}')
        if self.node_rank < 0 or self.node_rank >= self.node_size:
            raise ValueError(
                f'node_rank must be in [0, {self.node_size}), got {self.node_rank}'
            )
        if self.dp_size % self.node_size != 0:
            raise ValueError(
                f'dp_size ({self.dp_size}) must be divisible by node_size ({self.node_size})'
            )


@dataclass
class OfflineInferDataParallelArguments(OfflineInferArguments,
                                        DataParallelArguments):
    """Combined arguments for offline inference with data parallelism."""

    def __post_init__(self):
        OfflineInferArguments.__post_init__(self)
        DataParallelArguments.__post_init__(self)


class DataParallelOfflineInferenceRunner:
    """Data Parallel version of OfflineInferenceRunner."""

    def __init__(self,
                 args: OfflineInferDataParallelArguments,
                 dp_rank: int = 0,
                 dp_size: int = 1):
        """Initialize the runner with data parallel configuration.

        Args:
            args: Combined arguments for offline inference and data parallelism
            dp_rank: Global data parallel rank of this process
            dp_size: Total number of data parallel processes
        """
        self.args = args
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self._file_lock = threading.Lock()
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.system_prompt: Optional[str] = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type)

        # Create rank-specific output file to avoid conflicts
        output_path = Path(args.output_file)
        self.rank_output_file = str(
            output_path.parent /
            f'{output_path.stem}_rank_{dp_rank}{output_path.suffix}')

    def setup_vllm_engine(self) -> Tuple[LLM, SamplingParams]:
        """Initialize the vLLM engine and sampling parameters."""
        logger.info('=' * 60)
        logger.info(
            f'🚀 Initializing vLLM Engine (DP Rank {self.dp_rank}/{self.dp_size})'
        )
        logger.info(f'Model: {self.args.model_name_or_path}')
        logger.info(f'Max Model Length: {self.args.max_model_len}')
        logger.info(f'Max tokens: {self.args.max_tokens}')
        logger.info(f'Tensor Parallel Size: {self.args.tensor_parallel_size}')
        logger.info(f'Data Parallel Size: {self.dp_size}')
        logger.info(f'Data Parallel Rank: {self.dp_rank}')
        logger.info('=' * 60)

        # Prepare HuggingFace overrides
        hf_overrides = self._prepare_hf_overrides()

        try:
            # Initialize vLLM engine
            logger.info('Loading vLLM engine...')
            llm = LLM(
                model=self.args.model_name_or_path,
                tensor_parallel_size=self.args.tensor_parallel_size,
                pipeline_parallel_size=self.args.pipeline_parallel_size,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                enable_chunked_prefill=self.args.enable_chunked_prefill,
                enable_prefix_caching=self.args.enable_prefix_caching,
                enforce_eager=self.args.enforce_eager,
                max_num_seqs=self.args.max_num_seqs,
                max_model_len=self.args.max_model_len,
                hf_overrides=hf_overrides,
                seed=self.args.seed,
                trust_remote_code=self.args.trust_remote_code,
                dtype=self.args.dtype,
            )
            logger.info('✅ vLLM engine loaded successfully')

        except Exception as e:
            logger.exception(f'❌ Failed to initialize vLLM engine {e}')
            raise RuntimeError(f'Engine initialization failed: {e}') from e

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
        )

        logger.info('✅ vLLM engine initialization completed')
        return llm, sampling_params

    def _prepare_hf_overrides(self) -> Dict[str, Any]:
        """Prepare HuggingFace model overrides from arguments."""
        hf_overrides: Dict[str, Any] = {}

        if self.args.rope_scaling:
            hf_overrides['rope_scaling'] = self.args.rope_scaling

        if self.args.max_model_len:
            hf_overrides['max_model_len'] = self.args.max_model_len

        return hf_overrides

    def convert_to_messages_format(
            self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """Convert an input record to the vLLM chat messages format."""
        # Determine field keys with fallbacks
        input_key = getattr(self.args, 'input_key', None) or DEFAULT_INPUT_KEY
        label_key = getattr(self.args, 'label_key', None) or DEFAULT_LABEL_KEY

        required_keys = [input_key, label_key]
        missing_keys = [key for key in required_keys if key not in item]

        if missing_keys:
            logger.warning(
                f'Missing required keys {missing_keys} in item: {list(item.keys())}'
            )
            return None

        # Prefer the user-provided input key, fallback to 'prompt'
        prompt = item.get(input_key)
        ground_truth = item.get(label_key)

        # Validate required fields
        if not prompt or not ground_truth:
            logger.warning(
                f'Empty required field in item - question: {bool(prompt)}, '
                f'ground_truth: {bool(ground_truth)}')
            return None

        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        return messages

    def _write_batch_results(self, original_items: Sequence[Dict[str, Any]],
                             outputs: Sequence[Any]) -> None:
        """Write batch results to rank-specific file."""
        with self._file_lock:
            try:
                with open(self.rank_output_file, 'a', encoding='utf-8') as f:
                    for idx, (original_item,
                              output) in enumerate(zip(original_items,
                                                       outputs)):
                        # Defensive checks around vLLM response objects
                        model_response: str = ''
                        if output is not None:
                            try:
                                model_response = output.outputs[
                                    0].text if output.outputs else ''
                            except Exception:
                                model_response = ''

                        # Only write if we got a valid response
                        if model_response and model_response.strip():
                            result = copy.deepcopy(original_item)
                            result.setdefault('gen', []).append(model_response)
                            result[
                                'dp_rank'] = self.dp_rank  # Add rank info for debugging

                            f.write(
                                json.dumps(result, ensure_ascii=False) + '\n')
                            f.flush()
                        else:
                            logger.warning(
                                f'Empty response for item {idx}, skipping write'
                            )
            except Exception as e:
                logger.error(f'Error writing batch results: {e}')
                raise

    def count_completed_samples(self) -> Dict[str, int]:
        """Count completed samples for resume functionality."""
        completed_counts: Dict[str, int] = collections.defaultdict(int)

        if not os.path.exists(self.rank_output_file):
            return completed_counts

        if os.path.getsize(self.rank_output_file) == 0:
            return completed_counts

        try:
            with open(self.rank_output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        prompt_key = item.get(
                            self.args.input_key) or item.get('prompt')
                        gen_count = len(item.get('gen', []))
                        if prompt_key is not None:
                            completed_counts[str(prompt_key)] += gen_count
                    except json.JSONDecodeError as e:
                        logger.warning(f'Invalid JSON on line {line_num}: {e}')
                        continue
        except Exception as e:
            logger.error(f'Error reading output file for resume check: {e}')

        return completed_counts

    def load_and_distribute_data(self) -> List[Dict[str, Any]]:
        """Load dataset and distribute across data parallel ranks."""
        logger.info(f'Loading data from: {self.args.input_file}')

        try:
            with open(self.args.input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError as e:
            logger.critical(
                f'Input file not found: {self.args.input_file}, {e}')
            raise
        except json.JSONDecodeError as e:
            logger.critical(f'Invalid JSON in input file: {e}')
            raise

        logger.info(f'Loaded {len(data)} items from input file')

        # Check for completed samples
        completed_counts = self.count_completed_samples()
        total_completed = sum(completed_counts.values())

        if total_completed > 0:
            logger.info(
                f'Found {total_completed} completed samples from previous run')

        # Expand data according to n_samples and resume functionality
        expanded_data: List[Dict[str, Any]] = []
        skipped_items = 0
        for item in data:
            prompt_val = item.get(self.args.input_key) or item.get('prompt')
            prompt = str(prompt_val) if prompt_val is not None else ''
            if not prompt.strip():
                logger.warning(
                    f'No valid prompt found under keys [{self.args.input_key!r}, "prompt"] for item with keys: {list(item.keys())}'
                )
                skipped_items += 1
                continue

            completed = completed_counts.get(prompt, 0)
            remaining = max(0, self.args.n_samples - completed)

            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

        if skipped_items > 0:
            logger.warning(
                f'Skipped {skipped_items} items due to missing or empty prompt'
            )

        # Distribute data across DP ranks
        floor = len(expanded_data) // self.dp_size
        remainder = len(expanded_data) % self.dp_size

        def start(rank):
            return rank * floor + min(rank, remainder)

        rank_data = expanded_data[start(self.dp_rank):start(self.dp_rank + 1)]

        if len(rank_data) == 0:
            # If any rank has no data to process, we need to set a placeholder
            logger.warning(f'DP rank {self.dp_rank} has no data to process')

        logger.info(
            f'DP rank {self.dp_rank} needs to process {len(rank_data)} samples'
        )
        return rank_data

    def process_and_write_batch(self, batch_data: Sequence[Dict[str,
                                                                Any]]) -> None:
        """Process a single batch of data and write results to file."""
        if not batch_data:
            logger.warning('Empty batch data provided')
            return

        if self.llm is None or self.sampling_params is None:
            raise RuntimeError(
                'vLLM engine is not initialized. Call setup_vllm_engine() first.'
            )

        # Keep only items that successfully convert to message format
        valid_items: List[Dict[str, Any]] = []
        valid_messages: List[List[Dict[str, str]]] = []

        for item in batch_data:
            messages = self.convert_to_messages_format(item)
            if messages is not None:
                valid_items.append(copy.deepcopy(item))
                valid_messages.append(messages)

        if not valid_messages:
            logger.warning(
                'All items in this batch failed message conversion; skipping.')
            return

        try:
            logger.debug(f'Processing batch of {len(valid_messages)} prompts')
            outputs: List[Any] = self.llm.chat(valid_messages,
                                               self.sampling_params,
                                               use_tqdm=False)
            self._write_batch_results(valid_items, outputs)
        except Exception as e:
            logger.error(f'❌ Error during vLLM processing for this batch: {e}')
            raise RuntimeError(f'Batch processing failed: {e}') from e

    def run(self) -> None:
        """Run the main inference process for this DP rank."""
        if not self.args.input_file or not Path(self.args.input_file).exists():
            raise FileNotFoundError(
                f'Input file not found: {self.args.input_file}')

        try:
            # Load and distribute data
            rank_dataset = self.load_and_distribute_data()
            if not rank_dataset:
                logger.info(
                    'No data to process for this rank, skipping inference')
                return

            # Create output directory if it doesn't exist
            output_path = Path(self.rank_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f'⏳ DP Rank {self.dp_rank} starting to process {len(rank_dataset)} entries'
            )

            # Initialize vLLM engine
            self.llm, self.sampling_params = self.setup_vllm_engine()

            # Process data in batches
            total_batches = (len(rank_dataset) + self.args.batch_size -
                             1) // self.args.batch_size
            logger.info(
                f'Processing {total_batches} batches with batch size {self.args.batch_size}'
            )

            with tqdm(total=total_batches,
                      desc=f'DP Rank {self.dp_rank} Processing batches',
                      unit='batch') as pbar:
                for i in range(0, len(rank_dataset), self.args.batch_size):
                    batch = rank_dataset[i:i + self.args.batch_size]
                    self.process_and_write_batch(batch)
                    pbar.update(1)

            logger.info(
                f'✨ DP Rank {self.dp_rank} completed. Results saved to {self.rank_output_file}'
            )

        except Exception as e:
            logger.critical(f'❌ Fatal error during inference: {e}')
            raise

        # Give engines time to pause their processing loops before exiting
        sleep(1)


def run_single_dp_process(
    args: OfflineInferDataParallelArguments,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    tp_size: int,
) -> None:
    """Run inference for a single data parallel process."""
    # Set up vLLM data parallel environment variables
    os.environ['VLLM_DP_RANK'] = str(global_dp_rank)
    os.environ['VLLM_DP_RANK_LOCAL'] = str(local_dp_rank)
    os.environ['VLLM_DP_SIZE'] = str(dp_size)
    os.environ['VLLM_DP_MASTER_IP'] = dp_master_ip
    os.environ['VLLM_DP_MASTER_PORT'] = str(dp_master_port)

    # Update tensor parallel size for this process
    args.tensor_parallel_size = tp_size

    # Create and run the inference runner
    runner = DataParallelOfflineInferenceRunner(args, global_dp_rank, dp_size)
    runner.run()


def merge_rank_outputs(args: OfflineInferDataParallelArguments) -> None:
    """Merge outputs from all ranks into the final output file."""
    logger.info('Merging outputs from all ranks...')

    output_path = Path(args.output_file)
    merged_results = []

    # Collect results from all rank files
    for rank in range(args.dp_size):
        rank_file = output_path.parent / f'{output_path.stem}_rank_{rank}{output_path.suffix}'
        if rank_file.exists():
            try:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            # Remove the rank info we added for debugging
                            result.pop('dp_rank', None)
                            merged_results.append(result)
                # Clean up rank-specific file
                rank_file.unlink()
            except Exception as e:
                logger.error(f'Error reading rank {rank} output file: {e}')

    # Write merged results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in merged_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info(
        f'✅ Merged {len(merged_results)} results into {args.output_file}')


def main() -> None:
    """Main function to run data parallel offline inference."""
    # Parse arguments
    parser = HfArgumentParser(OfflineInferDataParallelArguments)
    args, = parser.parse_args_into_dataclasses()

    # Log configuration
    logger.info('Initializing Data Parallel Offline Inference...')
    logger.info('\n--- Parsed Arguments ---')
    logger.info(json.dumps(asdict(args), indent=2, default=str))

    # Determine master IP and port
    if args.node_size == 1:
        dp_master_ip = '127.0.0.1'
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    if not dp_master_ip:
        raise ValueError(
            'Master address must be specified for multi-node setup')
    if dp_master_port == 0:
        raise ValueError('Master port must be specified for multi-node setup')

    # Calculate DP processes per node
    dp_per_node = args.dp_size // args.node_size

    logger.info(
        f'Starting {dp_per_node} DP processes on node {args.node_rank}')
    logger.info(f'Master: {dp_master_ip}:{dp_master_port}')

    # Start processes for this node
    processes = []
    for local_dp_rank in range(dp_per_node):
        global_dp_rank = args.node_rank * dp_per_node + local_dp_rank

        proc = Process(
            target=run_single_dp_process,
            args=(
                args,
                args.dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                args.tp_size,
            ),
        )
        proc.start()
        processes.append(proc)

    # Wait for all processes to complete
    exit_code = 0
    for proc in processes:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            logger.error(
                f"Killing process {proc.pid} that didn't stop within {args.timeout} seconds."
            )
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    # Merge outputs from all ranks (only on rank 0)
    if args.node_rank == 0:
        try:
            merge_rank_outputs(args)
        except Exception as e:
            logger.error(f'Error merging outputs: {e}')
            exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
