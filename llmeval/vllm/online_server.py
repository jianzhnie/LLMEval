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
from llmeval.utils.template import (SYSTEM_PROMPT_FACTORY,
                                    is_chat_template_applied)

logger = init_logger('online_vllm_server', logging.INFO)


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
        api_key: OpenAI API key from environment variables
        client: The OpenAI client instance
        timeout: Request timeout in seconds
    """

    def __init__(self, base_url: str, timeout: int) -> None:
        """Initialize the inference client.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            timeout: Request timeout in seconds

        Raises:
            ValueError: If timeout is invalid
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
                logging.warn(f'max length exceeded. Error: {err_msg}')
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
        """Count the number of completed samples for each prompt in the output file.

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
        """Load and prepare the dataset, handling resume functionality.

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

    def _write_result(self, result: Dict[str, Any]) -> None:
        """Write result to output file in a thread-safe manner.

        Args:
            result: The result dictionary to write
        """
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is immediately written

    def process_item(self, item: Dict[str, Any]) -> None:
        """Process a single item and write the result to the output file.

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
                self._write_result(result)
            else:
                logger.warning(
                    f'Empty or invalid response for item: {item}, skipping write'
                )

        except ClientError as e:
            logger.error(f'Failed to get content for item. Error: {e}')
            # Don't write error entries - just log and continue
        except Exception as e:
            logger.error(f'Unexpected error processing item: {e}')
            # Don't write error entries - just log and continue

    def _validate_arguments(self) -> None:
        """Validate runner arguments.

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If output file path is not provided
        """
        if not self.args.input_file or not Path(self.args.input_file).exists():
            raise FileNotFoundError(
                f'Input file not found: {self.args.input_file}')
        if not self.args.output_file:
            raise ValueError('Output file path is required')

    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        output_path = Path(self.args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _process_concurrently(self, expanded_data: List[Dict[str,
                                                             Any]]) -> None:
        """Process items concurrently using thread pool.

        Args:
            expanded_data: List of data items to process
        """
        total_tasks = len(expanded_data)

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

    def run(self) -> None:
        """
        Run the main inference process using a thread pool.

        This method orchestrates the entire inference process, including data loading,
        concurrent processing, and progress tracking.
        """
        self._validate_arguments()

        expanded_data = self.load_data()
        total_tasks = len(expanded_data)

        if total_tasks == 0:
            logger.info(
                'All samples have already been processed, skipping inference. Exiting.'
            )
            return

        logger.info(f'Total remaining samples to process: {total_tasks}')

        self._setup_output_directory()
        self._process_concurrently(expanded_data)

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
