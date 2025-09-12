"""Offline vLLM inference runner.

This module provides a small, documented wrapper around vLLM to:
- Load a line-delimited JSON dataset
- Resume generation per unique prompt up to a requested sample count
- Convert records into vLLM chat message format
- Run batched chat inference
- Persist unified results incrementally for robustness

The output schema appends generations into a `gen` list for each input record.
"""

import collections
import copy
import json
import logging
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

# Initialize logger
logger = init_logger('vllm_infer', logging.INFO)


class OfflineInferenceRunner:
    """Main class to handle offline inference with the vLLM engine.

    This runner:
    - Loads input data and expands per-record sampling counts with resume support.
    - Converts input records into vLLM chat message format.
    - Runs batched inference using vLLM.
    - Writes results to a line-delimited JSON output file in a unified schema.
    """

    def __init__(self, args: OfflineInferArguments):
        """Initialize the runner with parsed CLI arguments.

        Args:
            args: Parsed `OfflineInferArguments` used to configure vLLM and IO.
        """
        self.args: OfflineInferArguments = args
        self._file_lock: threading.Lock = threading.Lock()
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.system_prompt: Optional[str] = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type)

    def setup_vllm_engine(self) -> Tuple[LLM, SamplingParams]:
        """
        Initialize the vLLM engine and sampling parameters.

        Returns:
            A tuple containing the LLM instance and SamplingParams instance.
        """
        logger.info('=' * 60)
        logger.info('🚀 Initializing vLLM Engine')
        logger.info(f'Model: {self.args.model_name_or_path}')
        logger.info(f'Max Model Length: {self.args.max_model_len}')
        logger.info(f'Max tokens: {self.args.max_tokens}')
        logger.info(f'RoPE Scaling: {self.args.rope_scaling}')
        logger.info(f'Tensor Parallel Size: {self.args.tensor_parallel_size}')
        logger.info(
            f'Pipeline Parallel Size: {self.args.pipeline_parallel_size}')
        logger.info(
            f'GPU Memory Utilization: {self.args.gpu_memory_utilization}')
        logger.info(f'Batch Size: {self.args.batch_size}')
        logger.info('=' * 50)

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
            # Include traceback for easier debugging
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
        """
        Prepare HuggingFace model overrides from arguments.

        Returns:
            Dictionary of overrides for HuggingFace model loading
        """
        hf_overrides: Dict[str, Any] = {}

        if self.args.rope_scaling:
            hf_overrides['rope_scaling'] = self.args.rope_scaling

        if self.args.max_model_len:
            hf_overrides['max_model_len'] = self.args.max_model_len

        return hf_overrides

    def convert_to_messages_format(
            self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        Convert an input record to the vLLM chat messages format.

        Expected item keys:
            - Prefer `self.args.input_key`; fallback to 'prompt'.

        Returns:
            The messages list if conversion succeeds, otherwise None.
        """
        # Prefer the user-provided input key, fallback to 'prompt'
        if 'prompt' in item or self.args.input_key in item:
            prompt_val = item.get(self.args.input_key, item.get('prompt', ''))
            prompt = str(prompt_val) if prompt_val is not None else ''
            if not prompt.strip():
                logger.warning(f'Empty prompt in item: {item}')
                return None

            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({
                    'role': 'system',
                    'content': self.system_prompt
                })
            messages.append({'role': 'user', 'content': prompt})
            return messages

        logger.warning(f'Unable to convert item to messages format: {item}')
        return None

    def _write_batch_results(self, original_items: Sequence[Dict[str, Any]],
                             outputs: Sequence[Any]) -> None:
        """Write batch results to file in unified schema with a 'gen' field.

        The output schema appends a generated string into `gen` list for each item.
        """
        with self._file_lock:
            try:
                with open(self.args.output_file, 'a', encoding='utf-8') as f:
                    for idx, original_item in enumerate(original_items):
                        # Defensive checks around vLLM response objects
                        output = outputs[idx] if idx < len(outputs) else None
                        model_response: str = ''
                        if output is not None:
                            try:
                                # vLLM chat returns RequestOutput objects with `.outputs`
                                # and each contains `.text`.
                                model_response = output.outputs[
                                    0].text if output.outputs else ''
                            except Exception:
                                model_response = ''

                        # Only write if we got a valid response
                        if model_response and model_response.strip():
                            result = copy.deepcopy(original_item)
                            result.setdefault('gen', []).append(model_response)
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
        """
        Count completed samples for resume functionality.

        This method scans the output file to determine how many samples have
        already been processed for each unique question, enabling resume
        functionality for interrupted runs.

        Returns:
            Dictionary mapping question content to count of completed samples
        """
        completed_counts: Dict[str, int] = collections.defaultdict(int)

        if not os.path.exists(self.args.output_file):
            return completed_counts

        if os.path.getsize(self.args.output_file) == 0:
            return completed_counts

        try:
            with open(self.args.output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line)
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

    def load_data(self) -> List[Dict[str, Any]]:
        """Load and expand the dataset, handling resume functionality per prompt.

        Returns:
            Expanded dataset where each record appears as many times as its
            remaining required generations.
        Raises:
            FileNotFoundError: If the input file does not exist.
            json.JSONDecodeError: If an input line is not valid JSON.
        """
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

        logger.info(
            f'Total remaining samples to process: {len(expanded_data)}')
        return expanded_data

    def process_and_write_batch(
        self,
        batch_data: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Process a single batch of data and write results to file.

        Steps:
            - Convert items to messages format.
            - Filter out invalid items safely.
            - Run vLLM chat inference.
            - Persist outputs for valid items.

        Raises:
            RuntimeError: If the vLLM engine is not initialized or processing fails.
        """
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
        """Run the main inference process end-to-end."""
        try:
            # Load data (including resume functionality)
            eval_dataset = self.load_data()
            if not eval_dataset:
                logger.info(
                    'All samples have already been processed, skipping inference'
                )
                return

            # Create output directory if it doesn't exist
            output_path = Path(self.args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f'⏳ Starting to process {len(eval_dataset)} entries')

            # Initialize vLLM engine
            self.llm, self.sampling_params = self.setup_vllm_engine()

            # Process data in batches
            total_batches = (len(eval_dataset) + self.args.batch_size -
                             1) // self.args.batch_size
            logger.info(
                f'Processing {total_batches} batches with batch size {self.args.batch_size}'
            )

            with tqdm(total=total_batches,
                      desc='Processing batches',
                      unit='batch') as pbar:
                for i in range(0, len(eval_dataset), self.args.batch_size):
                    batch = eval_dataset[i:i + self.args.batch_size]
                    self.process_and_write_batch(batch)
                    pbar.update(1)

            logger.info(
                f'✨ Final data processing completed. Results saved to {self.args.output_file}'
            )

        except Exception as e:
            logger.critical(f'❌ Fatal error during inference: {e}')
            raise


def main(args: OfflineInferArguments) -> None:
    """
    Main function to run the vLLM offline inference process.

    Args:
        args: Configuration arguments for the inference process

    Raises:
        RuntimeError: If inference process fails
    """
    try:
        runner = OfflineInferenceRunner(args)
        runner.run()
    except Exception as e:
        logger.critical(f'❌ Inference process failed: {e}')
        raise RuntimeError(f'Inference failed: {e}') from e


if __name__ == '__main__':
    """Command-line interface for vLLM offline inference."""
    try:
        # Parse command line arguments
        parser = HfArgumentParser(OfflineInferArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        # Log configuration
        logger.info(
            'Initializing OfflineInferArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        logger.info(json.dumps(asdict(eval_args), indent=2, default=str))

        # Run main inference process
        main(eval_args)

    except ImportError as e:
        logger.error(
            f'❌ A required library is missing: {e}. Please install it.')
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info('Process interrupted by user')
        sys.exit(0)
    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}')
        sys.exit(1)
