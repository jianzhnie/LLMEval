"""
Model Output Evaluation Script

This script provides functionality to evaluate language model outputs for various tasks.
It processes input data in JSONL format and computes performance metrics based on the
specified task type. Currently supports evaluation for 'math_opensource' tasks with
extensibility for additional task types.

Features:
    - JSONL input file processing
    - Flexible task-specific evaluation
    - Caching support for efficiency
    - Parallel processing capabilities
    - Robust error handling

Example:
    $ python eval.py --input_path data.jsonl --task_name math_opensource --cache_path cache/

Author: jianzhnie
Date: 2025
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from transformers import HfArgumentParser

from llmeval.tasks.math_eval.math_score import compute_scores
from llmeval.utils.config import EvalTaskArguments
from llmeval.utils.logger import init_logger

# Initialize logger for the evaluation module
logger = init_logger('math_eval')


def _get_after_think(text: str) -> str:
    """
    Extract the text content that appears after the '</think>' tag in the input string.

    This helper function is used to process model outputs that may contain thinking steps
    or reasoning enclosed in think tags. It efficiently extracts the final answer or
    conclusion that follows the thinking process.

    Args:
        text: The input string that may contain a '</think>' tag followed by text.
            Example: "Let me think...\n</think>\n\nThe answer is 42"

    Returns:
        str: The substring after '</think>\n\n', or the original text if the tag is not found.
            In the example above, would return "The answer is 42"
    """
    # Using str.partition for efficiency instead of split
    # partition returns a 3-tuple: (before_separator, separator, after_separator)
    return text.partition('</think>\n\n')[2]


def _process_item(item: Dict[str, Any],
                  task_name: str,
                  label_key: str = 'answer',
                  response_key: str = 'gen') -> Dict[str, Any]:
    """
    Process and validate a single data item from the input dataset.

    This function performs validation checks on the input dictionary to ensure it contains
    the required keys for evaluation. It creates a copy of the input item to avoid
    modifying the original data and adds the task name for reference.

    Args:
        item: A dictionary containing the evaluation data with the following expected keys:
            - label_key: Contains the ground truth answer
            - response_key: Contains the model's generated response
        task_name: The identifier for the evaluation task (e.g., 'math_opensource')
        label_key: The dictionary key used to access the ground truth answer (default: 'answer')
        response_key: The dictionary key used to access the model's response (default: 'gen')

    Returns:
        Dict[str, Any]: A new dictionary containing the validated data with added task field

    Raises:
        ValueError: If either the label_key or response_key is missing from the input item
        TypeError: If the item argument is not a dictionary
    """
    if not isinstance(item, dict):
        raise TypeError(
            f'Expected dictionary input, got {type(item).__name__}')

    # Validate required keys with detailed error messages
    if label_key not in item:
        raise ValueError(
            f"Missing ground truth label key '{label_key}' in item. "
            f"Available keys: {', '.join(item.keys())}")
    if response_key not in item:
        raise ValueError(
            f"Missing model response key '{response_key}' in item. "
            f"Available keys: {', '.join(item.keys())}")

    # Create a new copy to avoid modifying the original dictionary
    processed_item = item.copy()
    processed_item['task'] = task_name
    return processed_item


def evaluate_task(eval_dataset: List[Dict[str, Any]],
                  task_name: str,
                  label_key: str,
                  response_key: str,
                  cache_path: Union[str, Path],
                  max_workers: int,
                  timeout: int = 20) -> Optional[float]:
    """
    Evaluate model outputs against ground truth data for a specific task.

    This function handles the evaluation process for different types of tasks.
    Currently supports 'math_opensource' tasks, but is designed to be extensible
    for additional task types.

    Args:
        eval_dataset: List of dictionaries containing the evaluation data
        task_name: Identifier for the evaluation task (format: 'source/specific_task')
        label_key: Dictionary key for accessing ground truth answers
        response_key: Dictionary key for accessing model responses
        cache_path: Path where evaluation results will be cached
        max_workers: Maximum number of parallel workers for processing
        timeout: Maximum time in seconds to wait for each evaluation (default: 20)

    Returns:
        Optional[float]: Evaluation accuracy score if successful, None if evaluation fails

    Example:
        >>> data = [{"input": "2+2", "answer": "4", "gen": "4"}]
        >>> accuracy = evaluate_task(
        ...     data, "math_opensource", "answer", "gen",
        ...     "cache/results", max_workers=4
        ... )
        >>> print(f"Accuracy: {accuracy:.2f}")
    """
    if not eval_dataset:
        logger.warning('Empty dataset provided for evaluation')
        return None

    # Parse task name to determine evaluation type
    task_parts = task_name.split('/')
    dataset_source = task_parts[0] if task_parts else task_name

    # Convert cache_path to Path object for consistent handling
    cache_path = Path(cache_path)

    if dataset_source == 'math_opensource':
        try:
            accuracy = compute_scores(
                eval_dataset=eval_dataset,
                label_key=label_key,
                response_key=response_key,
                cache_path=str(
                    cache_path),  # compute_scores expects string path
                max_workers=max_workers,
                timeout=timeout)
            logger.info(f'✅ Task: {task_name}, Accuracy: {accuracy:.2%}')
            return accuracy
        except Exception as e:
            logger.error(f'❌ Evaluation failed: {str(e)}', exc_info=True)
            return None
    else:
        logger.error(f"🤷‍♂️ Unsupported task type: '{task_name}'")
        return None


def main() -> int:
    """
    Main entry point for the evaluation script.

    This function orchestrates the entire evaluation process:
    1. Parses command line arguments
    2. Sets up logging and cache directory
    3. Loads and validates input data
    4. Processes the data
    5. Runs the evaluation
    6. Reports results

    Returns:
        int: Exit code (0 for success, 1 for errors)
    """
    try:
        # Parse command line arguments using HuggingFace's argument parser
        parser = HfArgumentParser(EvalTaskArguments)
        args, = parser.parse_args_into_dataclasses()

        # Log initialization with formatted argument display
        logger.info(
            'Initializing evaluation with the following configuration:')
        logger.info('\n--- Parsed Arguments ---')
        logger.info(json.dumps(dataclasses.asdict(args), indent=2))

        # Ensure cache directory exists
        cache_dir = Path(args.cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate input data
        try:
            with open(args.input_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        except FileNotFoundError:
            logger.error(f"❌ Input file not found: '{args.input_path}'")
            return 1
        except json.JSONDecodeError as e:
            logger.error(
                f"❌ Invalid JSON format in '{args.input_path}': {str(e)}")
            return 1

        if not data:
            logger.error('❌ Input file is empty')
            return 1

        # Process data items and handle potential errors
        try:
            processed_data = [
                _process_item(item, args.task_name, args.label_key,
                              args.response_key) for item in data
            ]
        except (ValueError, TypeError) as e:
            logger.error(f'❌ Error processing data: {str(e)}')
            return 1

        # Run evaluation and get results
        accuracy = evaluate_task(processed_data, args.task_name,
                                 args.label_key, args.response_key,
                                 args.cache_path, args.max_workers,
                                 args.timeout)

        if accuracy is not None:
            logger.info('🎉 Evaluation completed successfully!')
            return 0
        else:
            logger.error('❌ Evaluation failed to produce results')
            return 1

    except Exception as e:
        logger.error(f'❌ Unexpected error: {str(e)}', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
