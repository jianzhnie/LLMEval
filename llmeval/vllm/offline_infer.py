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
        self.system_prompt: Optional[str] = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type)
        self._file_lock = threading.Lock()
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None

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
            logger.exception('❌ Failed to initialize vLLM engine')
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
        # Ensure output directory exists if a directory part is present
        out_dir = os.path.dirname(self.args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with self._file_lock:
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
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                    else:
                        logger.warning(
                            f'Empty response for item {idx}, skipping write')

    def count_completed_samples(self) -> Dict[str, int]:
        """
        Count the number of completed samples per unique prompt in the output file.

        Uses `args.input_key` to identify the prompt; falls back to 'prompt'.
        """
        completed_counts: Dict[str, int] = collections.defaultdict(int)
        if os.path.exists(self.args.output_file) and os.path.getsize(
                self.args.output_file) > 0:
            with open(self.args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        prompt_key = item.get(
                            self.args.input_key) or item.get('prompt')
                        gen_count = len(item.get('gen', []))
                        if prompt_key is not None:
                            completed_counts[str(prompt_key)] += gen_count
                    except json.JSONDecodeError:
                        continue
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
        try:
            with open(self.args.input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(
                f"Error loading input file '{self.args.input_file}': {e}")
            raise

        completed_counts = self.count_completed_samples()
        total_completed = sum(completed_counts.values())
        if total_completed > 0:
            logger.info(
                f'Found a total of {total_completed} samples from a previous run.'
            )

        expanded_data: List[Dict[str, Any]] = []
        for item in data:
            prompt_val = item.get(self.args.input_key) or item.get('prompt')
            prompt = str(prompt_val) if prompt_val is not None else ''
            if not prompt:
                logger.warning(
                    f'No {self.args.input_key} or "prompt" found in item: {item}'
                )
                continue
            completed = completed_counts.get(prompt, 0)
            remaining = max(0, self.args.n_samples - completed)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

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
        """
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError(
                'vLLM engine not initialized. Call setup_vllm_engine() first.')

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
            outputs: List[Any] = self.llm.chat(valid_messages,
                                               self.sampling_params,
                                               use_tqdm=False)
            self._write_batch_results(valid_items, outputs)
        except Exception:
            # Provide traceback to aid debugging without interrupting the loop
            logger.exception('Error during vLLM processing for this batch')

    def run(self) -> None:
        """Run the main inference process end-to-end."""
        eval_dataset = self.load_data()
        if not eval_dataset:
            logger.info('All samples have already been processed. Exiting.')
            return

        logger.info(f'Starting to process {len(eval_dataset)} entries.')

        # Initialize vLLM engine
        self.llm, self.sampling_params = self.setup_vllm_engine()

        # Process in batches
        total_batches = (len(eval_dataset) + self.args.batch_size -
                         1) // self.args.batch_size
        with tqdm(total=total_batches, desc='Processing batches') as pbar:
            for i in range(0, len(eval_dataset), self.args.batch_size):
                batch = eval_dataset[i:i + self.args.batch_size]
                self.process_and_write_batch(batch)
                pbar.update(1)

        logger.info(
            f'Final data processing completed. Results saved to {self.args.output_file}.'
        )


def main(args: OfflineInferArguments) -> None:
    """Main function to run the vLLM inference and evaluation process."""
    runner = OfflineInferenceRunner(args)
    runner.run()


if __name__ == '__main__':
    try:
        parser = HfArgumentParser(OfflineInferArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing OfflineInferArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        import dataclasses
        logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

        main(eval_args)

    except ImportError as e:
        logger.error(f'A required library is missing: {e}. Please install it.')
        sys.exit(1)
    except Exception:
        logger.exception('An unrecoverable error occurred during execution')
        sys.exit(1)
