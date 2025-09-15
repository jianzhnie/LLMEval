import collections
import copy
import json
import logging
import os
import re
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from llmeval.utils.compass_template import CompassVerifier_PROMPT
from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.logger import init_logger

# Initialize logger
logger = init_logger('compass_verifier_infer', logging.INFO)

# Constants
VALID_JUDGMENTS = {'A', 'B', 'C'}
DEFAULT_INPUT_KEY = 'prompt'
DEFAULT_LABEL_KEY = 'gold_answer'
DEFAULT_RESPONSE_KEY = 'llm_response'


def extract_answer(response_string: str) -> str | None:
    """
    从模型响应中提取 <answer> 标签内的内容。

    Args:
        response_string: 包含 <answer> 标签的完整字符串。

    Returns:
        如果找到 <answer> 标签，则返回其内部的字符串内容；
        如果未找到，则返回 None。
    """
    # 定义正则表达式模式。
    # (.*?) 是一个非贪婪捕获组，用于匹配并提取标签内的所有内容。
    # re.DOTALL 标志确保 . 也能匹配换行符，以防 answer 内容有多行。
    pattern = r'<answer>(.*?)</answer>'

    # 在输入的字符串中搜索匹配项
    match = re.search(pattern, response_string, re.DOTALL)

    if match:
        # 如果找到匹配项，返回第一个捕获组（括号内的内容），并去除首尾空格
        return match.group(1).strip()
    else:
        # 如果没有找到匹配项，返回 None
        return response_string


def process_judgment(judgment_str: str) -> str:
    """
    Extract and clean judgment from model response.

    This function processes the raw model output to extract a clean judgment
    (A, B, or C) using multiple extraction strategies.

    Args:
        judgment_str: Raw judgment string from the model

    Returns:
        Clean judgment string (A, B, or C) or empty string if no valid judgment found

    Examples:
        >>> process_judgment("\\boxed{A}")
        'A'
        >>> process_judgment("Final Judgment: (B)")
        'B'
        >>> process_judgment("The answer is C")
        'C'
    """
    if not judgment_str or not isinstance(judgment_str, str):
        return ''

    judgment_str = judgment_str.strip()

    # Strategy 1: Look for \boxed{letter} pattern
    boxed_pattern = r'\\boxed\{([A-C])\}'
    boxed_matches = re.findall(boxed_pattern, judgment_str)
    if boxed_matches:
        return boxed_matches[-1]

    # Strategy 2: Direct match for single letter
    if judgment_str in VALID_JUDGMENTS:
        return judgment_str

    # Strategy 3: Extract from "Final Judgment:" section
    if 'Final Judgment:' in judgment_str:
        final_section = judgment_str.split('Final Judgment:')[-1]

        # Look for (A), (B), (C) pattern
        paren_pattern = r'\(([A-C])\)'
        paren_matches = re.findall(paren_pattern, final_section)
        if paren_matches:
            return paren_matches[-1]

        # Look for any A, B, or C in the final section
        letter_pattern = r'([A-C])'
        letter_matches = re.findall(letter_pattern, final_section)
        if letter_matches:
            return letter_matches[-1]

    # Strategy 4: Look for any A, B, or C in the entire string
    letter_pattern = r'([A-C])'
    all_matches = re.findall(letter_pattern, judgment_str)
    if all_matches:
        return all_matches[-1]

    return ''


class CompassVerifierOfflineInferenceRunner:
    """
    Main class for handling offline inference with vLLM engine for CompassVerifier.

    This class provides a comprehensive solution for running CompassVerifier inference
    with support for batch processing, resume functionality, and robust error handling.

    Attributes:
        args: Configuration arguments for the inference process
        _file_lock: Thread lock for safe file writing
        llm: vLLM engine instance
        tokenizer: HuggingFace tokenizer instance
        sampling_params: Sampling parameters for generation
    """

    def __init__(self, args: OfflineInferArguments) -> None:
        """
        Initialize the CompassVerifier inference runner.

        Args:
            args: Configuration arguments containing model settings, file paths, etc.

        Raises:
            ValueError: If required arguments are invalid
        """
        self.args: OfflineInferArguments = args
        self._file_lock: threading.Lock = threading.Lock()
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.sampling_params: Optional[SamplingParams] = None

    def setup_vllm_engine(self) -> Tuple[LLM, AutoTokenizer, SamplingParams]:
        """
        Initialize the vLLM engine, tokenizer, and sampling parameters.

        This method sets up the complete inference pipeline including model loading,
        tokenizer initialization, and sampling parameter configuration.

        Returns:
            A tuple containing:
                - LLM instance for inference
                - AutoTokenizer for text processing
                - SamplingParams for generation control

        Raises:
            RuntimeError: If engine initialization fails
            ImportError: If required dependencies are missing
        """
        logger.info('=' * 60)
        logger.info('🚀 Initializing CompassVerifier vLLM Engine')
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
        logger.info('=' * 60)

        # Prepare HuggingFace overrides
        hf_overrides = self._prepare_hf_overrides()

        try:
            # Initialize tokenizer
            logger.info('Loading tokenizer...')
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path,
                trust_remote_code=self.args.trust_remote_code,
                cache_dir=self.args.cache_dir)
            logger.info('✅ Tokenizer loaded successfully')

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
            logger.error(f'❌ Failed to initialize vLLM engine: {e}')
            raise RuntimeError(f'Engine initialization failed: {e}') from e

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
        )

        logger.info('✅ CompassVerifier vLLM engine initialization completed')
        return llm, tokenizer, sampling_params

    def _prepare_hf_overrides(self) -> Dict[str, Any]:
        """
        Prepare HuggingFace model overrides from arguments.

        Returns:
            Dictionary of overrides for HuggingFace model loading
        """
        hf_overrides: Dict[str, Any] = {}

        # Use the parsed rope_scaling_dict instead of the raw string
        if hasattr(self.args,
                   'rope_scaling_dict') and self.args.rope_scaling_dict:
            hf_overrides['rope_scaling'] = self.args.rope_scaling_dict

        if self.args.max_model_len:
            hf_overrides['max_model_len'] = self.args.max_model_len

        return hf_overrides

    def convert_to_compass_verifier_format(
            self, item: Dict[str, Any]) -> Optional[str]:
        """
        Convert input data item to CompassVerifier prompt format.

        This method extracts the required fields from the input data and formats
        them according to the CompassVerifier template.

        Args:
            item: Input data item containing question, gold_answer, and llm_response

        Returns:
            Formatted prompt string for CompassVerifier or None if conversion fails

        Raises:
            KeyError: If required keys are missing from the input item
            ValueError: If required fields are empty or invalid
        """
        # Determine field keys with fallbacks
        input_key = getattr(self.args, 'input_key', None) or DEFAULT_INPUT_KEY
        label_key = getattr(self.args, 'label_key', None) or DEFAULT_LABEL_KEY
        response_key = getattr(self.args, 'response_key',
                               None) or DEFAULT_RESPONSE_KEY

        # Check for required keys
        required_keys = [input_key, label_key, response_key]
        missing_keys = [key for key in required_keys if key not in item]

        if missing_keys:
            logger.warning(
                f'Missing required keys {missing_keys} in item: {list(item.keys())}'
            )
            return None

        # Extract required fields
        question = item.get(input_key)
        ground_truth = item.get(label_key)
        llm_response_raw = item.get(response_key)

        # Handle different response formats
        if isinstance(llm_response_raw, list) and llm_response_raw:
            llm_response = llm_response_raw[0]
        elif isinstance(llm_response_raw, str):
            llm_response = llm_response_raw
        else:
            logger.warning(
                f'Invalid response format in item: {type(llm_response_raw)}')
            return None

        # Validate required fields
        if not question or not ground_truth or not llm_response:
            logger.warning(
                f'Empty required field in item - question: {bool(question)}, '
                f'ground_truth: {bool(ground_truth)}, llm_response: {bool(llm_response)}'
            )
            return None

        llm_response = extract_answer(llm_response)
        # Format the prompt using CompassVerifier template
        try:
            formatted_prompt = CompassVerifier_PROMPT.format(
                question=question,
                gold_answer=ground_truth,
                llm_response=llm_response)
            return formatted_prompt
        except Exception as e:
            logger.error(f'Error formatting CompassVerifier prompt: {e}')
            return None

    def _write_batch_results(self, original_items: List[Dict[str, Any]],
                             outputs: List[RequestOutput]) -> None:
        """
        Write batch results to output file with thread-safe operations.

        This method processes the model outputs and writes them to the output file
        in JSONL format, maintaining the original data structure while adding
        the model's judgment.

        Args:
            original_items: List of original data items
            outputs: List of model outputs from vLLM
        """
        with self._file_lock:
            try:
                with open(self.args.output_file, 'a', encoding='utf-8') as f:
                    for idx, (original_item,
                              output) in enumerate(zip(original_items,
                                                       outputs)):
                        # Extract model response
                        model_response = ''
                        if output.outputs:
                            model_response = output.outputs[0].text or ''

                        # Only write if we got a valid response
                        if model_response and model_response.strip():
                            result = copy.deepcopy(original_item)
                            # Safely extract answer from 'gen' field if it exists and is a string
                            if 'gen' in result and isinstance(
                                    result['gen'], str):
                                result['gen'] = extract_answer(result['gen'])
                            # result['origin_judgment'] = model_response.strip()
                            result['judgment'] = process_judgment(
                                model_response)

                            # Write to file
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
                        item = json.loads(line.strip())
                        question = item.get('question')
                        if question is not None and 'judgment' in item:
                            completed_counts[question] += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f'Invalid JSON on line {line_num}: {e}')
                        continue
        except Exception as e:
            logger.error(f'Error reading output file for resume check: {e}')

        return completed_counts

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load and prepare dataset with resume functionality.

        This method loads the input data, checks for previously completed samples,
        and expands the dataset according to the n_samples parameter while
        respecting the resume functionality.

        Returns:
            List of data items to be processed

        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If input file contains invalid JSON
        """
        logger.info(f'Loading data from: {self.args.input_file}')
        input_key = getattr(self.args, 'input_key', None) or DEFAULT_INPUT_KEY

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
            question = item.get(input_key)
            if not question:
                logger.warning(
                    f'No "question" field found in item: {list(item.keys())}')
                skipped_items += 1
                continue

            completed = completed_counts.get(question, 0)
            remaining = max(0, self.args.n_samples - completed)

            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

        if skipped_items > 0:
            logger.warning(
                f'Skipped {skipped_items} items due to missing question field')

        logger.info(f'Total samples to process: {len(expanded_data)}')
        return expanded_data

    def process_and_write_batch(self, batch_data: List[Dict[str,
                                                            Any]]) -> None:
        """
        Process a single batch of data and write results.

        This method handles the complete batch processing pipeline including
        data conversion, model inference, and result writing.

        Args:
            batch_data: List of data dictionaries for the current batch

        Raises:
            RuntimeError: If batch processing fails
        """
        if not batch_data:
            logger.warning('Empty batch data provided')
            return

        original_items = copy.deepcopy(batch_data)
        batch_prompts: List[str] = []

        # Convert data format and filter invalid items
        for item in batch_data:
            prompt = self.convert_to_compass_verifier_format(item)
            if prompt is not None:
                batch_prompts.append(prompt)
            else:
                logger.warning(
                    'Failed to convert item to CompassVerifier format')
                batch_prompts.append('')

        # Filter out empty prompts and corresponding original items
        valid_prompts: List[str] = []
        valid_original_items: List[Dict[str, Any]] = []

        for i, prompt in enumerate(batch_prompts):
            if prompt:  # Only include non-empty prompts
                valid_prompts.append(prompt)
                valid_original_items.append(original_items[i])

        if not valid_prompts:
            logger.warning('No valid prompts in this batch, skipping')
            return

        try:
            # Convert prompts to messages format for vLLM
            batch_messages: List[str] = []
            for prompt in valid_prompts:
                messages = [{'role': 'user', 'content': prompt}]
                model_inputs = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                batch_messages.append(model_inputs)

            # Use vLLM for inference
            logger.debug(f'Processing batch of {len(batch_messages)} prompts')
            outputs = self.llm.generate(
                batch_messages,
                self.sampling_params,
                use_tqdm=False  # Avoid progress bar conflicts
            )

            # Write results
            self._write_batch_results(valid_original_items, outputs)
            logger.debug(
                f'Successfully processed batch of {len(valid_original_items)} items'
            )

        except Exception as e:
            logger.error(f'❌ Error during vLLM processing for this batch: {e}')
            raise RuntimeError(f'Batch processing failed: {e}') from e

    def run(self) -> None:
        """
        Run the main inference process.

        This method orchestrates the complete inference pipeline including
        data loading, engine initialization, batch processing, and result writing.
        """
        if not self.args.input_file or not Path(self.args.input_file).exists():
            raise FileNotFoundError(
                f'Input file not found: {self.args.input_file}')
        if not self.args.output_file:
            raise ValueError('Output file path is required')
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
            self.llm, self.tokenizer, self.sampling_params = self.setup_vllm_engine(
            )

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
    Main function to run the CompassVerifier vLLM inference process.

    Args:
        args: Configuration arguments for the inference process

    Raises:
        RuntimeError: If inference process fails
    """
    try:
        runner = CompassVerifierOfflineInferenceRunner(args)
        runner.run()
    except Exception as e:
        logger.critical(f'❌ Inference process failed: {e}')
        raise RuntimeError(f'Inference failed: {e}') from e


if __name__ == '__main__':
    """Command-line interface for CompassVerifier offline inference."""
    try:
        # Parse command line arguments
        parser = HfArgumentParser(OfflineInferArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        # Log configuration
        logger.info(
            'Initializing CompassVerifier OfflineInferArguments with parsed command line arguments...'
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
        logger.info('�� Process interrupted by user')
        sys.exit(0)
    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}')
        sys.exit(1)
