import argparse
import collections
import concurrent.futures
import copy
import json
import logging
import os
import random
import threading
import time

import openai
from packaging.version import parse as parse_version
from tqdm import tqdm

from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

IS_OPENAI_V1 = parse_version(openai.__version__) >= parse_version('1.0.0')

if IS_OPENAI_V1:
    from openai import APIConnectionError, APIError, RateLimitError
else:
    from openai.error import APIConnectionError, APIError, RateLimitError


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""

    pass


file_lock = threading.Lock()


def get_content(query: str, system_prompt: str, base_url: str,
                model_name: str) -> str:
    """
    Fetches content from the OpenAI API based on the provided query, base URL, and model name.

    Args:
        query (str): The user's input query.
        system_prompt (str): The system prompt for the conversation.
        base_url (str): The base URL for the OpenAI API.
        model_name (str): The name of the model to use for generating the response.

    Returns:
        str: The generated content from the API.

    Raises:
        ClientError: If there are issues with the API request or response.
    """
    API_KEY = os.environ.get('OPENAI_API_KEY', 'EMPTY')
    API_REQUEST_TIMEOUT = int(os.getenv('OPENAI_API_REQUEST_TIMEOUT', '99999'))

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': query})

    if IS_OPENAI_V1:
        import httpx

        client = openai.OpenAI(
            api_key=API_KEY,
            base_url=base_url,
            timeout=httpx.Timeout(API_REQUEST_TIMEOUT),
        )
        call_func = client.chat.completions.create
        call_args = {
            'model': model_name,
            'messages': messages,
            'temperature': 0.6,
            'top_p': 0.95,
            'extra_body': {},
            'timeout': API_REQUEST_TIMEOUT,
        }
        call_args['extra_body'].update({'top_k': 40})
    else:
        call_func = openai.ChatCompletion.create
        call_args = {
            'api_key': API_KEY,
            'api_base': base_url,
            'model': model_name,
            'messages': messages,
            'temperature': 0.6,
            'top_p': 0.95,
            'request_timeout': API_REQUEST_TIMEOUT,
        }
        call_args.update({'top_k': 40})

    try:
        completion = call_func(**call_args)
        return completion.choices[0].message.content
    except AttributeError as e:
        err_msg = getattr(completion, 'message', '')
        logging.warning(f'AttributeError encountered: {err_msg}')
        time.sleep(random.randint(25, 35))
        raise ClientError(err_msg) from e
    except (APIConnectionError, RateLimitError) as e:
        err_msg = e.message if IS_OPENAI_V1 else e.user_message
        logging.warning(
            f'API connection error or rate limit exceeded: {err_msg}')
        time.sleep(random.randint(25, 35))
        raise ClientError(err_msg) from e
    except APIError as e:
        err_msg = e.message if IS_OPENAI_V1 else e.user_message
        if 'maximum context length' in err_msg:
            logging.warning(
                f'Maximum context length exceeded. Error: {err_msg}')
            return {'gen': '', 'end_reason': 'max length exceeded'}
        logging.warning(f'API error encountered: {err_msg}')
        time.sleep(1)
        raise ClientError(err_msg) from e


def count_completed_samples(output_file: str) -> collections.defaultdict:
    """
    Counts the number of completed samples for each prompt in the output file.

    Args:
        output_file (str): Path to the output JSONL file.

    Returns:
        collections.defaultdict: A dictionary mapping prompts to their
        completion counts.
    """
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item['prompt']
                    gen_count = len(item.get('gen', []))
                    prompt_counts[prompt] += gen_count
                except json.JSONDecodeError:
                    continue
    return prompt_counts


def process_item(item: dict, system_prompt: str, output_file: str,
                 base_url: str, model_name: str) -> dict:
    """
    Processes an individual item by fetching content from the OpenAI API and
    appending it to the output file.

    Args:
        item (dict): The input item containing the prompt.
        system_prompt (str): The system prompt to use for generating the response.
        output_file (str): Path to the output JSONL file.
        base_url (str): Base URL for the OpenAI API.
        model_name (str): Name of the model to use for generating the response.

    Returns:
        dict: The processed item with the generated content appended.
    """
    result = copy.deepcopy(item)

    response = get_content(item['prompt'], system_prompt, base_url, model_name)

    if 'gen' not in result:
        result['gen'] = []

    result['gen'].append(response)
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as g:
            g.write(json.dumps(result, ensure_ascii=False) + '\n')
            g.flush()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on model with prompts from a JSONL file')
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Input JSONL file path')
    parser.add_argument('--output_file',
                        type=str,
                        required=True,
                        help='Output file path')
    parser.add_argument('--n_samples',
                        type=int,
                        default=64,
                        help='Number of samples per prompt')
    parser.add_argument('--max_workers',
                        type=int,
                        default=128,
                        help='Maximum number of worker threads')
    parser.add_argument(
        '--base_url',
        type=str,
        default='http://10.77.249.36:8030/v1',
        help='Base URL of VLLM server',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/QwQ-32B',
        help='Model name of VLLM server',
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt for VLLM server',
    )
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    if os.path.exists(args.output_file):
        completed_counts = count_completed_samples(args.output_file)
        total_completed = sum(completed_counts.values())
        print(f'Found {total_completed} completed samples from previous run')
    else:
        completed_counts = {}

    expanded_data = []
    for item in data:
        prompt = item['prompt']
        completed = completed_counts.get(prompt, 0)
        remaining = args.n_samples - completed
        for _ in range(remaining):
            expanded_data.append(copy.deepcopy(item))

    total_tasks = len(expanded_data)
    print(f'Total remaining samples to process: {total_tasks}')

    completed_count = 0

    system_prompt = SYSTEM_PROMPT_FACTORY[args.system_prompt]

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers) as executor:
        future_to_item = {
            executor.submit(
                process_item,
                item,
                system_prompt,
                args.output_file,
                args.base_url,
                args.model_name,
            ): i
            for i, item in enumerate(expanded_data)
        }

        with tqdm(total=len(expanded_data), desc='Processing samples') as pbar:
            for future in concurrent.futures.as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    future.result()
                    completed_count += 1
                except Exception as exc:
                    print(f'Error processing sample {idx}: {exc}')
                pbar.update(1)

    print(f'Completed {completed_count}/{len(expanded_data)} samples')
    print(f'Results saved to {args.output_file}')


if __name__ == '__main__':
    main()
