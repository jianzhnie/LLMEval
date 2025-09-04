"""
This script evaluates model outputs for various tasks.

It takes a JSONL file as input, processes the data, and computes scores based on the specified task name.
Currently, it supports evaluation for 'math_opensource' tasks.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from llmeval.tasks.math_eval.math_score import compute_scores


def _get_after_think(text: str) -> str:
    """
    Extracts the text that comes after the '</think>' tag.

    Args:
        text: The input string, which may contain a '</think>' tag.

    Returns:
        The substring after '</think>\n\n', or the original text if the tag is not found.
    """
    # Using str.partition for efficiency and clarity
    # It returns a 3-tuple: (before, separator, after)
    return text.partition('</think>\n\n')[2]


def _process_item(item: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """
    Adds a 'task' key to a dictionary item.

    Args:
        item: A dictionary representing a single data entry.
        task_name: The name of the evaluation task.

    Returns:
        The updated dictionary with the 'task' key.
    """
    # Create a new copy to avoid modifying the original dictionary in place
    processed_item = item.copy()
    processed_item['task'] = task_name
    return processed_item


def _evaluate_task(data: List[Dict[str, Any]], task_name: str,
                   max_workers: int, cache_path: str) -> None:
    """
    Evaluates the data based on the specified task name.

    Args:
        data: A list of dictionaries to be evaluated.
        task_name: The name of the evaluation task.
        max_workers: The maximum number of worker threads for parallel processing.
        cache_path: The path to save cache results.
    """
    if 'math_opensource' in task_name:
        # The compute_scores function is assumed to handle the evaluation for math tasks
        try:
            acc = compute_scores(data, max_workers, cache_path)
            print(f'✅ Task: {task_name}, Accuracy: {acc:.4f}')
        except Exception as e:
            print(f'❌ An error occurred during evaluation: {e}')
    else:
        print(
            f"🤷‍♂️ No evaluation function found for task name: '{task_name}'")


def main() -> None:
    """
    Main function to parse arguments, load data, and run evaluation.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate model outputs based on a specified task.')
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to the input JSONL file.')
    parser.add_argument('--cache_path',
                        type=str,
                        required=True,
                        help='Path to save cache results.')
    parser.add_argument(
        '--max_workers',
        type=int,
        default=128,
        help='Maximum number of worker threads for parallel processing.')
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        help=
        "Task must be in ['math_opensource/aime24', 'math_opensource/aime25' , 'livecodebench', 'ifeval']."
    )

    args = parser.parse_args()

    # Create the directory for the cache file if it doesn't exist
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    # Load data from the input JSONL file
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{args.input_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in '{args.input_path}'")
        sys.exit(1)

    # Process each item to add the task name
    # Using a list comprehension for a more concise and readable loop
    processed_data = [_process_item(item, args.task_name) for item in data]

    # Run the evaluation
    _evaluate_task(processed_data, args.task_name, args.max_workers,
                   args.cache_path)

    print('🎉 Evaluation complete!')


if __name__ == '__main__':
    main()
