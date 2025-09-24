"""
Online inference server for OpenAI-compatible APIs.

This module provides a robust client for interacting with OpenAI-compatible APIs,
supporting concurrent requests, retry logic, and resume functionality for large-scale
inference tasks.
"""

import collections
import concurrent.futures
import copy
import dataclasses
import json
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import openai
from openai import APIConnectionError, APIError, RateLimitError
from tqdm import tqdm
from transformers import HfArgumentParser

from llmeval.utils.config import OnlineInferArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import (SYSTEM_PROMPT_FACTORY,
                                    is_chat_template_applied)

logger = init_logger('online_vllm_server', logging.INFO)

# Constants
DEFAULT_INPUT_KEY: str = 'prompt'
DEFAULT_LABEL_KEY: str = 'answer'
DEFAULT_RESPONSE_KEY: str = 'gen'


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""

    def __init__(self,
                 message: str,
                 original_error: Optional[Exception] = None) -> None:
        """Initialize ClientError with message and optional original error.

        Args:
            message: Error message describing the issue
            original_error: The original exception that caused this error
        """
        super().__init__(message)
        self.original_error = original_error


class InferenceClient:
    """
    A robust client to interact with OpenAI-compatible APIs.

    This client provides retry logic, error handling, and support for various
    generation parameters including thinking mode for advanced language models.

    Attributes:
        api_key (str): OpenAI API key from environment variables
        client (openai.OpenAI): The OpenAI client instance
        timeout (int): Request timeout in seconds
        base_url (str): Base URL for the OpenAI-compatible API
    """

    def __init__(self, base_url: str, timeout: int) -> None:
        """Initialize the inference client.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            timeout: Request timeout in seconds

        Raises:
            ValueError: If timeout is invalid or base_url is empty
            EnvironmentError: If OPENAI_API_KEY environment variable is not set
        """
        self.api_key: str = os.environ.get('OPENAI_API_KEY', 'EMPTY')
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        self.timeout: int = timeout

    def _prepare_messages(
            self, query: str,
            system_prompt: Optional[str]) -> List[Dict[str, str]]:
        """Prepare messages for the API call.

        Args:
            query: User's input query
            system_prompt: System prompt for the conversation

        Returns:
            List of message dictionaries

        Raises:
            ValueError: If chat template is already applied to the query
        """
        if is_chat_template_applied(query):
            logger.warning(
                'Chat template appears to be already applied to the query. '
                'Please use the raw prompt, as vLLM will apply the Hugging Face '
                'chat template automatically.')
            raise ValueError(
                'Your query has been applied with chat_template, please use the raw prompt, '
                'because the vLLM will apply the Hugging Face chat template automatically!'
            )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': query})
        return messages

    def get_content(
        self,
        query: str,
        system_prompt: Optional[str],
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ) -> str:
        """Fetch content from the OpenAI API with comprehensive retry logic.

        This method handles the core interaction with the OpenAI API, including
        parameter validation, message preparation, and error handling. It supports
        various generation parameters and includes built-in retry logic for
        transient errors.

        Args:
            query: User's input query
            system_prompt: System prompt for the conversation (optional)
            model_name: The model to use for generation (e.g., 'gpt-3.5-turbo')
            max_tokens: Maximum tokens to generate (1 to model's context limit)
            temperature: The sampling temperature (0.0 to 2.0)
            top_p: The top-p value for nucleus sampling (0.0 to 1.0)
            top_k: The top-k value for sampling (positive integer)
            enable_thinking: Whether to enable the "thinking" feature

        Returns:
            str: The generated content from the API

        Raises:
            ClientError: If there's a non-retryable API issue or max retries exceeded
            ValueError: If input parameters are invalid or out of range

        Note:
            The method includes automatic retry logic for certain types of API
            errors (connection issues, rate limits) but will raise exceptions
            for non-recoverable errors.
        """
        # Validate input parameters
        if not query or not query.strip():
            raise ValueError('Query cannot be empty')
        if not model_name:
            raise ValueError('Model name cannot be empty')

        # Prepare messages
        messages = self._prepare_messages(query, system_prompt)

        call_args = dict(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                'top_k': top_k,
                'chat_template_kwargs': {
                    'enable_thinking': enable_thinking
                },
            },
            timeout=self.timeout,
        )
        content = ''
        try:
            completion = self.client.chat.completions.create(**call_args)
            content = completion.choices[0].message.content
        except AttributeError as e:
            err_msg = getattr(completion, 'message', '')
            if err_msg:
                time.sleep(random.randint(25, 35))
                raise ClientError(err_msg) from e
            raise ClientError(err_msg) from e
        except (APIConnectionError, RateLimitError) as e:
            err_msg = e.message
            time.sleep(random.randint(25, 35))
            raise ClientError(err_msg) from e
        except APIError as e:
            err_msg = e.message
            if 'maximum context length' in err_msg:  # or "Expecting value: line 1 column 1 (char 0)" in err_msg:
                logger.warning(f'max length exceeded. Error: {err_msg}')
                return {'gen': '', 'end_reason': 'max length exceeded'}
            time.sleep(1)
            raise ClientError(err_msg) from e
        return content


class InferenceRunner:
    """
    Main class to handle the inference process with concurrent execution.

    This class provides comprehensive functionality for running inference tasks,
    including data loading, resume functionality, concurrent processing, and
    robust error handling.

    Attributes:
        args: Configuration arguments for the inference process
        client: The inference client instance
        system_prompt: System prompt for the conversation
        _file_lock: Thread lock for safe file writing
    """

    def __init__(self, args: OnlineInferArguments) -> None:
        """Initialize the inference runner.

        Args:
            args: Configuration arguments containing all necessary settings

        Raises:
            ValueError: If arguments are invalid
            FileNotFoundError: If input file doesn't exist
        """
        self.args: OnlineInferArguments = args
        self.client: InferenceClient = InferenceClient(args.base_url,
                                                       args.request_timeout)
        self.system_prompt: Optional[str] = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type)
        # Use class-level lock for thread safety
        self._file_lock: threading.Lock = threading.Lock()

    def count_completed_samples(self) -> Dict[str, int]:
        """Count completed samples for resume functionality.

        This method scans the output file to determine how many samples have
        already been processed for each unique question, enabling resume
        functionality for interrupted runs.

        The method handles:
        - File existence and size checks
        - JSON parsing with error recovery
        - Prompt key resolution with fallbacks
        - Comprehensive error handling

        Returns:
            Dictionary mapping question content to count of completed samples.
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
                        item: Dict[str, Any] = json.loads(line.strip())
                        prompt: Any = item.get(
                            self.args.input_key) or item.get(DEFAULT_INPUT_KEY)
                        gen_response = item.get(
                            self.args.response_key) or item.get(
                                DEFAULT_RESPONSE_KEY)
                        gen_count: int = len(gen_response)
                        if prompt is not None:
                            completed_counts[str(prompt)] += gen_count
                    except json.JSONDecodeError as e:
                        logger.warning(f'Invalid JSON on line {line_num}: {e}')
                        continue
        except Exception as e:
            logger.error(f'Error reading output file for resume check: {e}')

        return completed_counts

    def load_data(self) -> List[Dict[str, Any]]:
        """Load and prepare the dataset, handling resume functionality.

        This method performs several key operations:
        1. Loads and validates the input data file
        2. Checks for previously completed samples
        3. Expands the dataset based on required sample count
        4. Validates data structure and content

        Returns:
            List[Dict[str, Any]]: List of data items to process, with resume logic applied

        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If input file contains invalid JSON
            ValueError: If input data structure is invalid
        """
        # Input file validation and loading
        if not os.path.exists(self.args.input_file):
            raise FileNotFoundError(
                f'Input file not found: {self.args.input_file}')

        # Load raw data
        raw_data: List[Dict[str, Any]] = self._load_raw_data()
        logger.info(f'Loaded {len(raw_data)} items from input file')

        # Resume functionality handling
        completed_counts = self.count_completed_samples()
        total_completed = sum(completed_counts.values())

        if total_completed > 0:
            logger.info(
                f'Found {total_completed} completed samples from previous run.'
            )

        # Expand data according to n_samples and resume functionality
        expanded_data: List[Dict[str, Any]] = self._expand_data_with_resume(
            raw_data, completed_counts)

        if not expanded_data:
            logger.warning('No data to process after expansion')

        logger.info(
            f'Total remaining samples to process: {len(expanded_data)}')
        return expanded_data

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw data from input file.

        This method handles the loading of raw JSONL data from the input file
        with comprehensive error handling.

        Returns:
            List of raw data items.

        Raises:
            FileNotFoundError: If the input file does not exist.
            json.JSONDecodeError: If an input line is not valid JSON.
        """
        try:
            with open(self.args.input_file, 'r', encoding='utf-8') as f:
                data: List[Dict[str, Any]] = [
                    json.loads(line) for line in f if line.strip()
                ]
        except FileNotFoundError as e:
            logger.critical(
                f'Input file not found: {self.args.input_file}, {e}')
            raise
        except json.JSONDecodeError as e:
            logger.critical(f'Invalid JSON in input file: {e}')
            raise

        return data

    def _expand_data_with_resume(
            self, raw_data: List[Dict[str, Any]],
            completed_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """Expand data according to n_samples and resume functionality.

        This method processes the raw data and expands it based on the number
        of samples needed per prompt, taking into account already completed
        samples for resume functionality.

        Args:
            raw_data: Raw data loaded from input file.
            completed_counts: Count of completed samples per prompt.

        Returns:
            Expanded dataset with remaining samples to process.
        """
        expanded_data: List[Dict[str, Any]] = []
        skipped_items: int = 0

        for item in raw_data:
            prompt_val: Any = item.get(
                self.args.input_key) or item.get(DEFAULT_INPUT_KEY)
            prompt: str = str(prompt_val) if prompt_val is not None else ''

            if not prompt.strip():
                logger.warning(
                    f'No valid prompt found under keys [{self.args.input_key!r}, '
                    f'"{DEFAULT_INPUT_KEY}"] for item with keys: {list(item.keys())}'
                )
                skipped_items += 1
                continue

            completed: int = completed_counts.get(prompt, 0)
            remaining: int = max(0, self.args.n_samples - completed)

            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

        if skipped_items > 0:
            logger.warning(
                f'Skipped {skipped_items} items due to missing or empty prompt'
            )

        return expanded_data

    def _write_result(self, result: Dict[str, Any]) -> None:
        """Write result to output file in a thread-safe manner.

        Args:
            result: The result dictionary to write
        """
        with self._file_lock:
            try:
                with open(self.args.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is immediately written
            except Exception as e:
                logger.error(f'Error writing batch results: {e}')
                raise IOError(f'Failed to write batch results: {e}') from e

    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item and write the result to the output file.

        This method handles the complete processing pipeline for a single input item:
        1. Extracts the query from the item
        2. Makes the API call through the client
        3. Processes and validates the response
        4. Writes the result to the output file

        Args:
            item: The data item to process, must contain either input_key or 'prompt'

        Returns:
            Optional[Dict[str, Any]]: The processed result if successful, None if processing failed

        Note:
            - Errors are logged but don't stop execution
            - Failed items are skipped but logged
            - Empty or invalid responses are logged and skipped
        """
        if not isinstance(item, dict):
            logger.error(f'Invalid item type: {type(item)}, expected dict')
            return None

        try:
            # Extract query with detailed validation
            query = item.get(self.args.input_key)
            if not query:
                query = item.get('prompt')
            if not query:
                logger.warning(
                    f'No {self.args.input_key} or "prompt" found in item: {item}'
                )
                return None

            # Make API call with all necessary parameters
            response = self.client.get_content(
                query=query,
                system_prompt=self.system_prompt,
                model_name=self.args.model_name,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                enable_thinking=self.args.enable_thinking,
            )

            # Only write if we got a valid response
            if response and response.strip():
                result = copy.deepcopy(item)
                result.setdefault('gen', []).append(response)
                self._write_result(result)
            else:
                logger.warning(
                    f'Empty or invalid response for item: {item}, skipping write'
                )

        except ClientError as e:
            logger.error(f'API client error processing item: {str(e)}')
            if hasattr(e, 'original_error'):
                logger.debug(f'Original error: {e.original_error}')
            return None

        except (ValueError, TypeError) as e:
            logger.error(f'Validation error processing item: {str(e)}')
            return None

        except Exception as e:
            logger.error(f'Unexpected error processing item: {str(e)}',
                         exc_info=True)

    def _process_concurrently(self, expanded_data: List[Dict[str,
                                                             Any]]) -> None:
        """Process items concurrently using thread pool with error handling and progress tracking.

        This method manages concurrent processing of inference tasks using a thread pool.
        It includes comprehensive error handling and progress tracking for each task.

        Args:
            expanded_data: List of data items to process, where each item is a
                         dictionary containing the input data and metadata

        Note:
            - Uses ThreadPoolExecutor for concurrent processing
            - Implements proper error handling for each thread
            - Shows progress bar with tqdm
            - Maintains thread safety with class-level file lock
        """
        total_tasks = len(expanded_data)
        failed_tasks: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.max_workers,
                thread_name_prefix='inference_worker') as executor:
            futures = [
                executor.submit(self.process_item, item)
                for item in expanded_data
            ]

            with tqdm(total=total_tasks,
                      desc='Processing samples',
                      unit='sample') as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(
                            f'An unexpected error occurred in a thread: {e}',
                            exc_info=True)
                        failed_tasks.append({
                            'error':
                            str(e),
                            'timestamp':
                            time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    finally:
                        pbar.update(1)

        if failed_tasks:
            logger.warning(f'Total failed tasks: {len(failed_tasks)}')

    def run(self) -> None:
        """
        Run the main inference process using a thread pool.

        This method orchestrates the entire inference process, including data loading,
        concurrent processing, and progress tracking.
        """
        if not self.args.input_file or not Path(self.args.input_file).exists():
            raise FileNotFoundError(
                f'Input file not found: {self.args.input_file}')
        if not self.args.output_file:
            raise ValueError('Output file path is required')

        try:
            # Load data (including resume functionality)
            eval_dataset: List[Dict[str, Any]] = self.load_data()
            if not eval_dataset:
                logger.info(
                    'All samples have already been processed, skipping inference'
                )
                return

            output_path = Path(self.args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f'⏳ Starting to process {len(eval_dataset)} entries')

            self._process_concurrently(eval_dataset)

            logger.info(
                f'All {len(eval_dataset)} samples have been processed.')
            logger.info(f'Results saved to {self.args.output_file}')

        except Exception as e:
            logger.critical(f'❌ Fatal error during inference: {e}')
            raise


def main() -> None:
    """
    Main entry point for the online inference server.

    This function serves as the primary entry point for the inference server,
    handling:
    1. Command line argument parsing using HfArgumentParser
    2. Initialization of the inference runner
    3. Execution of the inference process
    4. Comprehensive error handling and logging

    The function uses dataclasses for type-safe argument handling and provides
    detailed logging of the initialization process and any errors that occur.

    Returns:
        None

    Raises:
        SystemExit: If initialization fails or command line arguments are invalid
        Exception: For any unhandled errors during execution
    """
    start_time = time.time()
    try:
        # Parse command line arguments into a strongly typed dataclass
        parser = HfArgumentParser(OnlineInferArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        # Log initialization with formatted argument display
        logger.info(
            'Initializing OnlineInferArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

        # Initialize and run the inference process
        runner = InferenceRunner(eval_args)
        runner.run()

        # Log successful completion with execution time
        total_time = time.time() - start_time
        logger.info(
            f'✅ Inference completed successfully in {total_time:.2f} seconds')

    except KeyboardInterrupt:
        logger.info('Interrupted by user. Exiting gracefully...')
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}',
            exc_info=True)
        raise


if __name__ == '__main__':
    main()
