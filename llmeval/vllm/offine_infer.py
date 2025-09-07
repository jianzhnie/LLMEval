import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.config import EvaluationArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

# Initialize logger
logger = init_logger('vllm_infer', logging.INFO)


class OfflineInferenceRunner:
    """Main class to handle offline inference with vLLM engine."""

    def __init__(self, args: EvaluationArguments):
        self.args: EvaluationArguments = args
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(args.system_prompt_type)
        self._file_lock = threading.Lock()
        self.llm = None
        self.sampling_params = None

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load dataset from a JSONL file with improved error handling.

        Args:
            dataset_path: The path to the JSONL dataset file.

        Returns:
            A list of dictionaries, where each dictionary is a data entry.
        """
        data_list = []
        logger.info(f'🔄 Loading dataset from: {dataset_path}')

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in tqdm(enumerate(f, 1),
                                           desc='Loading data',
                                           unit=' lines'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            f'⚠️ Unable to parse JSON on line {line_num}. '
                            f'Skipping. Snippet: "{line[:50]}..."')
        except FileNotFoundError:
            logger.error(f'❌ Dataset file not found at: {dataset_path}')
            raise
        except Exception as e:
            logger.error(
                f'❌ An unexpected error occurred while loading the dataset: {e}'
            )
            raise

        logger.info(f'✅ Successfully loaded {len(data_list)} entries.')
        return data_list

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
        logger.info(f'RoPE Scaling: {self.args.rope_scaling}')
        logger.info(f'Tensor Parallel Size: {self.args.tensor_parallel_size}')
        logger.info(
            f'GPU Memory Utilization: {self.args.gpu_memory_utilization}')
        logger.info(f'Batch Size: {self.args.batch_size}')
        logger.info('=' * 50)

        # Set environment variable for vLLM sampler
        os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'

        # Prepare hf_overrides from arguments
        hf_overrides = {
            'rope_scaling': self.args.rope_scaling,
            'max_model_len': self.args.max_model_len,
        }

        try:
            llm = LLM(
                model=self.args.model_name_or_path,
                tensor_parallel_size=self.args.tensor_parallel_size,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                enable_prefix_caching=self.args.enable_prefix_caching,
                max_num_seqs=self.args.max_num_seqs,
                hf_overrides=hf_overrides,
                enforce_eager=self.args.enforce_eager,
                seed=self.args.seed,
            )
        except Exception as e:
            logger.error(f'❌ Failed to initialize vLLM engine: {e}')
            raise

        # 修复参数名不匹配问题
        sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,  # 修复：使用正确的参数名
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=getattr(self.args, 'repetition_penalty',
                                       1.0),  # 兼容性处理
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
        # 如果已经是messages格式，直接返回
        if 'messages' in item and isinstance(item['messages'], list):
            return item['messages']

        # 如果是prompt格式，转换为messages格式
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

    def process_and_write_batch(
        self,
        batch_data: List[Dict],
        start_index: int,
    ) -> None:
        """
        Process a single batch of data and write results to file with improved error handling.

        Args:
            batch_data: A list of data dictionaries for the current batch.
            start_index: The global starting index of this batch.
        """
        batch_messages = []
        original_items = []
        valid_indices = []

        # 转换数据格式并过滤无效项
        for i, item in enumerate(batch_data):
            original_items.append(item)
            messages = self.convert_to_messages_format(item)
            if messages is not None:
                batch_messages.append(messages)
                valid_indices.append(i)
            else:
                batch_messages.append(None)

        # 过滤出有效的消息
        valid_batch_messages = [
            msg for msg in batch_messages if msg is not None
        ]
        if not valid_batch_messages:
            logger.warning(
                'No valid messages in this batch. Writing error entries.')
            self._write_error_entries(original_items, start_index,
                                      'No valid messages')
            return

        try:
            # 使用vLLM进行推理
            outputs = self.llm.chat(
                valid_batch_messages,
                self.sampling_params,
                use_tqdm=False  # 避免进度条冲突
            )

            # 写入结果
            self._write_batch_results(original_items, batch_messages, outputs,
                                      valid_indices, start_index)

        except Exception as e:
            logger.error(f'❌ Error during vLLM processing for this batch: {e}')
            self._write_error_entries(original_items, start_index,
                                      f'Batch processing error: {str(e)}')

    def _write_batch_results(self, original_items: List[Dict],
                             batch_messages: List[Optional[List[Dict]]],
                             outputs: List, valid_indices: List[int],
                             start_index: int) -> None:
        """Write batch results to file."""
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                valid_output_index = 0
                for i, original_item in enumerate(original_items):
                    if batch_messages[i] is None:
                        # 写入错误条目
                        error_result = {
                            'id': start_index + i,
                            'meta': original_item,
                            'error': 'Invalid data format'
                        }
                        f.write(
                            json.dumps(error_result, ensure_ascii=False) +
                            '\n')
                        continue

                    # 处理有效项
                    if valid_output_index < len(outputs):
                        output = outputs[valid_output_index]
                        ai_response = output.outputs[
                            0].text if output.outputs else ''

                        # 构建完整消息
                        complete_messages = batch_messages[i].copy()
                        complete_messages.append({
                            'role': 'assistant',
                            'content': ai_response
                        })

                        result = {
                            'id': start_index + i,
                            'meta': original_item,
                            'messages': complete_messages
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        valid_output_index += 1
                f.flush()

    def _write_error_entries(self, original_items: List[Dict],
                             start_index: int, error_message: str) -> None:
        """Write error entries for failed items."""
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                for i, original_item in enumerate(original_items):
                    error_result = {
                        'id': start_index + i,
                        'meta': original_item,
                        'error': error_message
                    }
                    f.write(
                        json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()

    def count_completed_samples(self) -> int:
        """Count completed samples from previous run."""
        if not os.path.exists(self.args.output_file) or os.path.getsize(
                self.args.output_file) == 0:
            return 0

        count = 0
        with open(self.args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'error' not in item:  # 只计算成功的样本
                        count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def run(self) -> None:
        """Run the main inference process."""
        # 加载数据集
        dataset_path = os.path.join(self.args.dataset_dir,
                                    self.args.dataset_name)
        eval_dataset = self.load_dataset(dataset_path)

        # 扩展数据集（如果需要多次采样）
        original_len = len(eval_dataset)
        if self.args.n_sampling > 1:
            eval_dataset = eval_dataset * self.args.n_sampling
            logger.info(
                f'�� Dataset repeated {self.args.n_sampling} times, '
                f'expanded from {original_len} entries to {len(eval_dataset)}.'
            )

        # 确保输出目录存在
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.args.output_file = os.path.join(self.args.output_dir,
                                             'inference_results.jsonl')

        # 检查已完成的样本
        processed_count = self.count_completed_samples()
        if processed_count > 0:
            logger.info(
                f'📦 Found existing results file. Resuming from {processed_count} entries.'
            )

        remaining_data = eval_dataset[processed_count:]
        if not remaining_data:
            logger.info('✅ All data processing completed. Exiting.')
            return

        logger.info(
            f'⏳ Starting to process remaining {len(remaining_data)} entries.')

        # 初始化vLLM引擎
        self.llm, self.sampling_params = self.setup_vllm_engine()

        # 批处理数据
        total_batches = (len(remaining_data) + self.args.batch_size -
                         1) // self.args.batch_size
        with tqdm(total=total_batches, desc='Processing batches') as pbar:
            for i in range(0, len(remaining_data), self.args.batch_size):
                batch_start = i
                batch_end = min(i + self.args.batch_size, len(remaining_data))
                batch = remaining_data[batch_start:batch_end]

                global_start_index = processed_count + batch_start
                self.process_and_write_batch(batch, global_start_index)
                pbar.update(1)

        logger.info(
            f'✨ Final data processing completed. Results saved to {self.args.output_file}.'
        )


def main(args: EvaluationArguments) -> None:
    """Main function to run the vLLM inference and evaluation process."""
    runner = OfflineInferenceRunner(args)
    runner.run()


if __name__ == '__main__':
    try:
        parser = HfArgumentParser(EvaluationArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing EvaluationArguments with parsed command line arguments...'
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
