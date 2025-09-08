import collections
import copy
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

# Initialize logger
logger = init_logger('vllm_infer', logging.INFO)


class OfflineInferenceRunner:
    """Main class to handle offline inference with vLLM engine."""

    def __init__(self, args: OfflineInferArguments):
        self.args: OfflineInferArguments = args
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(args.system_prompt_type)
        self._file_lock = threading.Lock()
        self.llm = None
        self.sampling_params = None

    def setup_vllm_engine(self) -> Tuple[LLM, SamplingParams]:
        """
        Initialize the vLLM engine and sampling parameters with improved error handling.

        Returns:
            A tuple containing the LLM instance and SamplingParams instance.
        """
        # Print engine initialization information
        logger.info('=' * 50)
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

        # Prepare hf_overrides from arguments
        hf_overrides = {
            'rope_scaling': self.args.rope_scaling,
            'max_model_len': self.args.max_model_len,
        }

        try:
            llm = LLM(
                model=self.args.model_name_or_path,
                tensor_parallel_size=self.args.tensor_parallel_size,
                pipeline_parallel_size=self.args.pipeline_parallel_size,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                enable_chunked_prefill=self.args.enable_chunked_prefill,
                enable_prefix_caching=self.args.enable_prefix_caching,
                enforce_eager=self.args.enforce_eager,
                max_num_seqs=self.args.max_num_seqs,
                hf_overrides=hf_overrides,
                seed=self.args.seed,
            )
        except Exception as e:
            logger.error(f'❌ Failed to initialize vLLM engine: {e}')
            raise

        sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
        )

        logger.info('✅ vLLM engine initialization completed.')
        return llm, sampling_params

    def convert_to_messages_format(
            self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        Convert different input formats to messages format for vLLM.

        Args:
            item: Input data item

        Returns:
            Messages format for vLLM or None if conversion fails
        """
        if 'prompt' in item or self.args.input_key in item:
            prompt = item.get(self.args.input_key, item.get('prompt', ''))
            if not prompt:
                logger.warning(f'Empty prompt in item: {item}')
                return None

            messages = []
            if self.system_prompt:
                messages.append({
                    'role': 'system',
                    'content': self.system_prompt
                })
            messages.append({'role': 'user', 'content': prompt})
            return messages

        logger.warning(f'Unable to convert item to messages format: {item}')
        return None

    def _write_batch_results(self, original_items: List[Dict],
                             outputs: List) -> None:
        """Write batch results to file in unified schema with 'gen' field."""
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                for idx, original_item in enumerate(original_items):
                    output = outputs[idx]
                    model_response = output.outputs[
                        0].text if output.outputs else ''
                    result = copy.deepcopy(original_item)
                    result.setdefault('gen', []).append(model_response)
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

    def _write_error_entries(self, original_items: List[Dict],
                             error_message: str) -> None:
        """Write error entries for failed items, unified schema with 'gen'."""
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                for _, original_item in enumerate(original_items):
                    error_item = copy.deepcopy(original_item)
                    error_item.setdefault('gen',
                                          []).append(f'ERROR: {error_message}')
                    f.write(json.dumps(error_item, ensure_ascii=False) + '\n')
                    f.flush()

    def count_completed_samples(self) -> Dict[str, int]:
        """
        Counts the number of completed samples for each prompt in the output file.
        Uses args.input_key to identify the prompt; falls back to 'prompt'.
        """
        completed_counts = collections.defaultdict(int)
        if os.path.exists(self.args.output_file) and os.path.getsize(
                self.args.output_file) > 0:
            with open(self.args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        prompt = item.get(
                            self.args.input_key) or item.get('prompt')
                        gen_count = len(item.get('gen', []))
                        if prompt is not None:
                            completed_counts[prompt] += gen_count
                    except json.JSONDecodeError:
                        continue
        return completed_counts

    def load_data(self) -> List[Dict[str, Any]]:
        """Loads and prepares the dataset, handling resume functionality per prompt."""
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

        expanded_data = []
        for item in data:
            prompt = item.get(self.args.input_key) or item.get('prompt')
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
        batch_data: List[Dict],
    ) -> None:
        """
        Process a single batch of data and write results to file with improved error handling.

        Args:
            batch_data: A list of data dictionaries for the current batch.
        """
        original_items = copy.deepcopy(batch_data)
        batch_messages = []
        # 转换数据格式并过滤无效项
        for item in batch_data:
            messages = self.convert_to_messages_format(item)
            batch_messages.append(messages)
        try:
            # 使用vLLM进行推理
            outputs = self.llm.chat(
                batch_messages,
                self.sampling_params,
                use_tqdm=False  # 避免进度条冲突
            )
            # 写入结果（统一schema）
            self._write_batch_results(
                original_items,
                outputs,
            )

        except Exception as e:
            logger.error(f'❌ Error during vLLM processing for this batch: {e}')
            self._write_error_entries(original_items,
                                      f'Batch processing error: {str(e)}')

    def run(self) -> None:
        """Run the main inference process."""
        # 加载数据（含断点续训展开）
        eval_dataset = self.load_data()
        if not eval_dataset:
            logger.info(
                'All samples have already been processed, skipping inference. Exiting.'
            )
            return

        logger.info(f'⏳ Starting to process {len(eval_dataset)} entries.')

        # 初始化vLLM引擎
        self.llm, self.sampling_params = self.setup_vllm_engine()

        # 批处理数据
        total_batches = (len(eval_dataset) + self.args.batch_size -
                         1) // self.args.batch_size
        with tqdm(total=total_batches, desc='Processing batches') as pbar:
            for i in range(0, len(eval_dataset), self.args.batch_size):
                batch = eval_dataset[i:i + self.args.batch_size]
                self.process_and_write_batch(batch)
                pbar.update(1)

        logger.info(
            f'✨ Final data processing completed. Results saved to {self.args.output_file}.'
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
        logger.error(
            f'❌ A required library is missing: {e}. Please install it.')
        exit(1)
    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}')
        exit(1)
