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
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

logger = init_logger(__name__, logging.INFO)


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""

    def __init__(self,
                 message: str,
                 original_error: Optional[Exception] = None) -> None:
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
        max_retries (int): Maximum number of retry attempts
    """

    def __init__(self,
                 base_url: str,
                 timeout: int,
                 max_retries: int = 3) -> None:
        """
        Initialize the inference client.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests

        Raises:
            ValueError: If timeout or max_retries are invalid
        """
        self.api_key: str = os.environ.get('OPENAI_API_KEY', 'EMPTY')
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        self.timeout: int = timeout
        self.max_retries: int = max_retries

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
        """
        Fetch content from the OpenAI API with comprehensive retry logic.

        Args:
            query: User's input query
            system_prompt: System prompt for the conversation (optional)
            model_name: The model to use for generation
            max_tokens: Maximum tokens to generate
            temperature: The sampling temperature (0.0 to 2.0)
            top_p: The top-p value for nucleus sampling (0.0 to 1.0)
            top_k: The top-k value for sampling
            enable_thinking: Whether to enable the "thinking" feature

        Returns:
            The generated content from the API

        Raises:
            ClientError: If there's a non-retryable API issue or max retries exceeded
            ValueError: If input parameters are invalid
        """
        # Validate input parameters
        if not query or not query.strip():
            raise ValueError('Query cannot be empty')
        if not model_name:
            raise ValueError('Model name cannot be empty')
        # Prepare messages
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': query})

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
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

                content = completion.choices[0].message.content
                if content is None:
                    logger.warning('Received None content from API')
                    return ''

                return content

            except (APIConnectionError, RateLimitError) as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = random.uniform(25, 35) * (2**attempt)
                    logger.warning(
                        f'API connection error or rate limit exceeded '
                        f'(attempt {attempt + 1}/{self.max_retries}): {e.message}. '
                        f'Retrying in {wait_time:.1f}s...')
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f'Max retries exceeded for API connection error: {e.message}'
                    )
                    raise ClientError(
                        f'Retryable error after {self.max_retries} attempts: {e.message}',
                        original_error=e) from e

            except APIError as e:
                if 'maximum context length' in e.message:
                    logger.warning(
                        f'Maximum context length exceeded. Error: {e.message}')
                    return ''
                logger.error(f'Unrecoverable API error: {e.message}')
                raise ClientError(f'Unrecoverable error: {e.message}',
                                  original_error=e) from e

            except Exception as e:
                logger.error(f'An unexpected error occurred: {e}')
                raise ClientError(f'Unexpected error: {e}',
                                  original_error=e) from e

        # This should never be reached, but just in case
        raise ClientError(
            f'Unexpected end of retry loop after {self.max_retries} attempts')


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
        """
        Initialize the inference runner.

        Args:
            args: Configuration arguments containing all necessary settings

        Raises:
            ValueError: If arguments are invalid
            FileNotFoundError: If input file doesn't exist
        """
        self.args: OnlineInferArguments = args
        self.client: InferenceClient = InferenceClient(args.base_url,
                                                       args.request_timeout,
                                                       args.max_retries)
        self.system_prompt: Optional[str] = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type)
        # 使用类级别的锁，确保线程安全
        self._file_lock: threading.Lock = threading.Lock()

    def count_completed_samples(self) -> Dict[str, int]:
        """
        Count the number of completed samples for each prompt in the output file.

        Returns:
            A dictionary mapping prompts to their completion counts
        """
        # Count completed samples from a previous run
        completed_counts: Dict[str, int] = collections.defaultdict(int)
        if not os.path.exists(self.args.output_file) or os.path.getsize(
                self.args.output_file) == 0:
            return completed_counts
        try:
            with open(self.args.output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        prompt = item.get(
                            self.args.input_key) or item.get('prompt')
                        if not prompt:
                            logger.warning(
                                f'No {self.args.input_key} found in item at line {line_num}: {item}'
                            )
                            continue

                        gen_count = len(item.get('gen', []))
                        completed_counts[prompt] += gen_count
                    except json.JSONDecodeError as e:
                        logger.warning(f'Invalid JSON at line {line_num}: {e}')
                        continue
        except Exception as e:
            logger.error(
                f'Error reading output file {self.args.output_file}: {e}')
            raise

        return completed_counts

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load and prepare the dataset, handling resume functionality.

        Returns:
            List of data items to process, with resume logic applied

        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If input file contains invalid JSON
        """
        try:
            with open(self.args.input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(
                f"Error loading input file '{self.args.input_file}': {e}")
            raise

        if not data:
            logger.warning(
                f'No data found in input file: {self.args.input_file}')
            return []

        # Handle resume functionality
        completed_counts = self.count_completed_samples()
        total_completed = sum(completed_counts.values())

        if total_completed > 0:
            logger.info(
                f'Found {total_completed} completed samples from previous run.'
            )

        # Expand data based on remaining samples needed
        expanded_data: List[Dict[str, Any]] = []
        for item in data:
            prompt = item.get(self.args.input_key) or item.get('prompt')
            if not prompt:
                logger.warning(
                    f'No {self.args.input_key} found in item: {item}')
                continue

            completed = completed_counts.get(prompt, 0)
            remaining = max(0, self.args.n_samples - completed)

            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

        return expanded_data

    def process_item(self, item: Dict[str, Any]) -> None:
        """
        Process a single item and write the result to the output file.

        Args:
            item: The data item to process
        """
        try:
            query = item.get(self.args.input_key) or item.get('prompt')
            if not query:
                logger.warning(
                    f'No {self.args.input_key} found in item: {item}')
                return

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

                # 使用类级别的锁确保线程安全
                with self._file_lock:
                    with open(self.args.output_file, 'a',
                              encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()  # 确保数据立即写入
            else:
                logger.warning(
                    f'Empty or invalid response for item : {item}, skipping write'
                )

        except ClientError as e:
            logger.error(f'Failed to get content for item. Error: {e}')
            # Don't write error entries - just log and continue
        except Exception as e:
            logger.error(f'Unexpected error processing item: {e}')
            # Don't write error entries - just log and continue

    def run(self) -> None:
        """
        Run the main inference process using a thread pool.

        This method orchestrates the entire inference process, including data loading,
        concurrent processing, and progress tracking.
        """
        expanded_data = self.load_data()
        total_tasks = len(expanded_data)

        if total_tasks == 0:
            logger.info(
                'All samples have already been processed, skipping inference. Exiting.'
            )
            return

        logger.info(f'Total remaining samples to process: {total_tasks}')

        # Create output directory if it doesn't exist
        output_path = Path(self.args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process items concurrently
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.max_workers) as executor:
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
                            f'An unexpected error occurred in a thread: {e}')
                    pbar.update(1)

        logger.info(f'All {total_tasks} samples have been processed.')
        logger.info(f'Results saved to {self.args.output_file}')


def main() -> None:
    """
    Main entry point for the online inference server.

    This function handles argument parsing, initialization, and execution of the
    inference process with comprehensive error handling.
    """
    try:
        parser = HfArgumentParser(OnlineInferArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing OnlineInferArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

        runner = InferenceRunner(eval_args)
        runner.run()

    except KeyboardInterrupt:
        logger.info('Interrupted by user. Exiting gracefully...')
    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}')
        raise


if __name__ == '__main__':
    main()
