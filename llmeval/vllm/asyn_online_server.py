import asyncio
import collections
import copy
import json
import logging
import os
import random
import threading
from typing import Any, Dict, List, Optional

import httpx
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from tqdm import tqdm
from transformers import HfArgumentParser

from llmeval.utils.config import OnlineInferArguments
from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

logger = init_logger(__name__, logging.INFO)


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""
    pass


class AsyncInferenceClient:
    """An async client to interact with the OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout: int, max_retries: int = 3):
        self.api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        self.timeout = timeout
        self.max_retries = max_retries

    async def get_content(
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
        Fetches content from the OpenAI API with retry logic (async).
        """
        # Allow direct messages input; otherwise build from query + optional system prompt
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': query})

        for attempt in range(self.max_retries):
            try:
                completion = await self.client.chat.completions.create(
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
                return content

            except (APIConnectionError, RateLimitError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = random.uniform(0.5, 1.5) * (2**attempt)
                    logger.warning(
                        f'API connection error or rate limit exceeded '
                        f'(attempt {attempt + 1}/{self.max_retries}): {str(e)}. '
                        f'Retrying in {wait_time:.1f}s...')
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f'Max retries exceeded for API connection error: {str(e)}'
                    )
                    raise ClientError(
                        f'Retryable error after {self.max_retries} attempts: {str(e)}'
                    ) from e

            except APIError as e:
                e_str = str(e)
                if 'maximum context length' in e_str:
                    logger.warning(
                        f'Maximum context length exceeded. Error: {e_str}')
                    return ''
                logger.error(f'Unrecoverable API error: {e_str}')
                raise ClientError(f'Unrecoverable error: {e_str}') from e

            except Exception as e:
                logger.error(f'An unexpected error occurred: {str(e)}')
                raise ClientError(f'Unexpected error: {str(e)}') from e


class InferenceRunner:
    """Main class to handle the async inference process, including data loading and concurrency control."""

    def __init__(self, args: OnlineInferArguments):
        self.args: OnlineInferArguments = args
        self.client = AsyncInferenceClient(args.base_url,
                                           args.request_timeout,
                                           max_retries=self.args.max_retries)
        self.system_prompt = SYSTEM_PROMPT_FACTORY.get(args.system_prompt_type)
        # File lock remains useful to serialize writes even in async context (thread-safe file IO)
        self._file_lock = threading.Lock()

    def count_completed_samples(self) -> int:
        """
        Counts the number of completed samples for each prompt in the output file.
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
                        if not prompt:
                            logger.warning(
                                f'No {self.args.input_key} found in item: {item}'
                            )
                            continue
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
            completed_counts = self.count_completed_samples()
            total_completed = sum(completed_counts.values())
            logger.info(
                f'Found a total of {total_completed} samples from a previous run.'
            )
        else:
            completed_counts = {}

        expanded_data = []
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

    async def process_item(self, item: Dict[str,
                                            Any]) -> Optional[Dict[str, Any]]:
        """
        Process one item asynchronously and return the result dict to be written,
        or None if failed/empty (we only write successful results).
        """
        try:
            query = item.get(self.args.input_key) or item.get('prompt')
            if not query:
                logger.warning(
                    f'No {self.args.input_key} found in item: {item}')
                return

            response = await self.client.get_content(
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
                return result
            return None

        except ClientError as e:
            logger.error(f'Failed to get content for item. Error: {e}')
            return None
        except Exception as e:
            logger.error(f'Unexpected error processing item: {e}')
            return None

    async def run(self) -> None:
        """Run async inference with bounded concurrency and streaming progress."""
        expanded_data = self.load_data()
        total_tasks = len(expanded_data)
        if total_tasks == 0:
            logger.info(
                'All samples have already been processed, skipping inference. Exiting.'
            )
            return

        sem = asyncio.Semaphore(self.args.max_workers)

        async def worker(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with sem:
                return await self.process_item(item)

        tasks = [worker(it) for it in expanded_data]
        written = 0
        with tqdm(total=total_tasks, desc='Processing samples') as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    with self._file_lock:
                        with open(self.args.output_file, 'a',
                                  encoding='utf-8') as f:
                            f.write(
                                json.dumps(result, ensure_ascii=False) + '\n')
                            f.flush()
                    written += 1
                pbar.update(1)

        logger.info(f'All {total_tasks} samples processed. {written} written.')
        logger.info(f'Results saved to {self.args.output_file}')


async def _amain() -> None:
    parser = HfArgumentParser(OnlineInferArguments)
    eval_args, = parser.parse_args_into_dataclasses()

    logger.info(
        'Initializing OnlineInferArguments with parsed command line arguments...'
    )
    logger.info('\n--- Parsed Arguments ---')
    import dataclasses
    logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

    runner = InferenceRunner(eval_args)
    await runner.run()


def main():
    asyncio.run(_amain())


if __name__ == '__main__':
    main()
