import collections
import concurrent.futures
import copy
import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import openai
from openai import APIConnectionError, APIError, RateLimitError
from tqdm import tqdm
from transformers import HfArgumentParser

from llmeval.utils.config import EvaluationArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

logger = init_logger(__name__, logging.INFO)


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""
    pass


class InferenceClient:
    """A client to interact with the OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout: int, max_retries: int = 3):
        self.api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        self.timeout = timeout
        self.max_retries = max_retries

    def get_content(self, query: str, system_prompt: Optional[str],
                    model_name: str, max_tokens: int, temperature: float,
                    top_p: float, top_k: int, presence_penalty: float,
                    enable_thinking: bool) -> str:
        """
        Fetches content from the OpenAI API with retry logic.

        Args:
            query (str): User's input query.
            system_prompt (Optional[str]): System prompt for the conversation.
            model_name (str): The model to use.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): The sampling temperature.
            top_p (float): The top-p value.
            top_k (int): The top-k value.
            presence_penalty (float): Presence penalty value.
            enable_thinking (bool): Whether to enable the "thinking" feature.


        Returns:
            str: The generated content from the API.

        Raises:
            ClientError: If there's a non-retryable API issue.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': query})

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body={
                        'top_k': top_k,
                        'chat_template_kwargs': {
                            'enable_thinking': enable_thinking
                        },
                    },
                    timeout=self.timeout,
                )

                # reasoning_content = completion.choices[0].message.reasoning_content
                content = completion.choices[0].message.content

                return content

            except (APIConnectionError, RateLimitError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = random.uniform(25, 35) * (
                        2**attempt)  # Exponential backoff
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
                        f'Retryable error after {self.max_retries} attempts: {e.message}'
                    ) from e

            except APIError as e:
                if 'maximum context length' in e.message:
                    logger.warning(
                        f'Maximum context length exceeded. Error: {e.message}')
                    return ''
                logger.error(f'Unrecoverable API error: {e.message}')
                raise ClientError(f'Unrecoverable error: {e.message}') from e

            except Exception as e:
                logger.error(f'An unexpected error occurred: {e}')
                raise ClientError(f'Unexpected error: {e}') from e


class InferenceRunner:
    """Main class to handle the inference process, including data loading and multi-threading."""

    def __init__(self, args: EvaluationArguments):
        self.args: EvaluationArguments = args
        self.client = InferenceClient(args.base_url, args.request_timeout)
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(args.system_prompt)
        logger.info(f'Using system prompt: {self.system_prompt}')
        # 使用类级别的锁，确保线程安全
        self._file_lock = threading.Lock()

    def count_completed_samples(self) -> int:
        """
        Counts the number of completed samples for each prompt in the output file.

        Args:
            output_file (str): Path to the output JSONL file.

        Returns:
            collections.defaultdict: A dictionary mapping prompts to their
            completion counts.
        """
        # Count completed samples from a previous run
        completed_counts = collections.defaultdict(int)
        if os.path.exists(self.args.output_file) and os.path.getsize(
                self.args.output_file) > 0:
            with open(self.args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        prompt = item.get('prompt')
                        gen_count = len(item.get('gen', []))
                        completed_counts[prompt] += gen_count
                    except json.JSONDecodeError:
                        continue
        return completed_counts

    def load_data(self) -> List[Dict[str, Any]]:
        """Loads and prepares the dataset, handling resume functionality."""
        try:
            with open(self.args.input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(
                f"Error loading input file '{self.args.input_file}': {e}")
            raise

        if os.path.exists(self.args.output_file):
            completed_counts = self.count_completed_samples(
                self.args.output_file)
            total_completed = sum(completed_counts.values())
            logger.info(
                f'Found a total of {total_completed} samples from a previous run.'
            )
        else:
            completed_counts = {}

        expanded_data = []
        for item in data:
            try:
                prompt = item.get(self.args.input_key)
            except KeyError:
                logger.warning(
                    f'No {self.args.input_key} found in item: {item}')
                continue
            completed = completed_counts.get(prompt, 0)
            remaining = self.args.n_sampling - completed
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

        logger.info(
            f'Total remaining samples to process: {len(expanded_data)}')
        return expanded_data

    def process_item(self, item: Dict[str, Any]) -> None:
        """Processes a single item and writes the result to the output file."""
        response = None
        try:
            query = item.get(self.args.input_key)
            if not query:
                logger.warning(
                    f'No {self.args.input_key} found in item: {item}')
                return
            response = self.client.get_content(
                query=query,
                system_prompt=self.system_prompt,
                model_name=self.args.model_name,
                max_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                presence_penalty=self.args.presence_penalty,
                enable_thinking=self.args.enable_thinking,
            )

            # 统一处理空响应
            if not response:
                logger.warning(f'Empty response for prompt: {query[:100]}...')
                response = ''

        except ClientError as e:
            logger.error(
                f'Failed to get content for prompt: {query}. Error: {e}')
            response = f'ERROR: {str(e)}'

        result = copy.deepcopy(item)
        result.setdefault('gen', []).append(response)

        # 使用类级别的锁确保线程安全
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # 确保数据立即写入

    def run(self):
        """Runs the main inference process using a thread pool."""
        expanded_data = self.load_data()
        total_tasks = len(expanded_data)
        if total_tasks == 0:
            logger.info(
                'All samples have already been processed, skipping inference. Exiting.'
            )
            return
        logger.info(f'Total remaining samples to process: {total_tasks}')
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.max_workers) as executor:
            futures = [
                executor.submit(self.process_item, item)
                for item in expanded_data
            ]

            with tqdm(total=total_tasks, desc='Processing samples') as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(
                            f'An unexpected error occurred in a thread: {e}')
                    pbar.update(1)

        logger.info(f'All {total_tasks} samples have been processed.')
        logger.info(f'Results saved to {self.args.output_file}')


def main():
    try:
        parser = HfArgumentParser(EvaluationArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing EvaluationArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        # Use dataclasses.asdict to print arguments cleanly
        import dataclasses
        logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

        runner = InferenceRunner(eval_args)
        runner.run()

    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}')
        raise


if __name__ == '__main__':
    main()
