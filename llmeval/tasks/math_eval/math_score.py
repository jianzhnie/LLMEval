"""
This module provides functionality to evaluate the accuracy of a large number of
model-generated answers against their ground truth using the `math-verify` library.
It leverages multiprocessing to speed up the evaluation process.
"""

import json
import os
from concurrent.futures import TimeoutError
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth
from llmeval.utils.logger import init_logger

# Configure a dedicated logger for the math scoring module
logger = init_logger(__name__)

# Define package requirements for better dependency management
REQUIRED_PACKAGES = {
    'math-verify': 'math-verify>=1.0.0',
    'pebble': 'pebble>=4.6.3',
    'tqdm': 'tqdm>=4.65.0'
}

# Attempt to import necessary components from math-verify.
# Provides helpful error messages if dependencies are missing.
try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as e:
    logger.error(f'Missing required dependency: {e}\n'
                 f'To use Math-Verify, install required packages:\n'
                 f'pip install {" ".join(REQUIRED_PACKAGES.values())}')
    import sys
    sys.exit(1)


def process_answers(
    args: Tuple[int, Dict[str, Any], str, str]
) -> Optional[Tuple[int, float, Optional[str], Optional[str]]]:
    """
    Processes a single model output by extracting the answer and comparing it with the
    ground truth using the `math-verify` metric.

    This function handles all aspects of processing a single mathematical answer:
    1. Extracts the task name and validates input format
    2. Parses ground truth answer from input data
    3. Processes model-generated text
    4. Verifies answer correctness using math-verify library
    5. Handles various error cases and timeouts

    Args:
        args (Tuple[int, Dict[str, Any], str, str]): Processing arguments containing:
            - index (int): Unique job identifier for tracking
            - input_data (Dict[str, Any]): Data dictionary with model output and ground truth
            - label_key (str): Key for accessing ground truth in input_data
            - response_key (str): Key for accessing model response in input_data

    Returns:
        Optional[Tuple[int, float, Optional[str], Optional[str]]]: Processing results:
            - index (int): Original job identifier
            - score (float): Verification score (1.0=correct, 0.0=incorrect)
            - pred_ans (Optional[str]): Extracted predicted answer, None if failed
            - gold_ans (Optional[str]): Extracted gold answer, None if failed

    Note:
        - Returns None if any unexpected error occurs during processing
        - Uses math-verify library with configurable precision (default=6)
        - Handles both expression and LaTeX answer formats
        - Implements timeout protection for long-running verifications
    """
    index, input_data, label_key, response_key = args
    try:
        data_name = input_data.get('task', '').split('/')[1]
    except (IndexError, AttributeError):
        logger.warning(f'⚠️ Invalid task format for job {index}')
        return index, 0.0, None, None

    # Parse the ground truth answer from the input data
    # The first return value (cot_answer) is unused for this metric
    try:
        # The first return value (cot_answer) is unused for this metric.
        _, gold_answer_text = parse_ground_truth(input_data, data_name,
                                                 label_key)
    except (ValueError, NotImplementedError, KeyError) as e:
        logger.error(
            f'❌ [Error] Parsing gold truth for job {index} failed: {e}')
        return index, 0.0, None, None

    # Get the generated text. Handles cases where response might be missing or empty.
    generated_text = input_data.get(response_key, [])
    if not generated_text:
        logger.warning(f'⚠️ No generated text found for job {index}')
        return index, 0.0, None, None
    generated_text = generated_text[0] if isinstance(
        generated_text, list) else str(generated_text)

    # Initialize the verification function from math_verify
    verify_func = math_metric(
        # The gold answer can be an expression or LaTeX.
        # We use both parsers to be robust to different formats.
        gold_extraction_target=(ExprExtractionConfig(),
                                LatexExtractionConfig()),
        # The predicted answer can also be an expression or LaTeX.
        pred_extraction_target=(ExprExtractionConfig(),
                                LatexExtractionConfig()),
        # Use max to select the best score if multiple extractions are successful.
        aggregation_function=max,
        precision=6,
    )

    try:
        # Run the verification using math-verify metric
        grade, extracted_answers = verify_func([generated_text],
                                               [gold_answer_text])

        if not extracted_answers:
            logger.warning(f'⚠️ No answers could be extracted for job {index}')
            return index, 0.0, None, None

        # Extract answers with validation
        try:
            pred_ans = extracted_answers[0]
            gold_ans = extracted_answers[1]
        except IndexError:
            logger.error(
                f'❌ [Error] Invalid extraction format for job {index}')
            return index, 0.0, None, None

        # Validate grade value
        if not (isinstance(grade, (int, float)) and 0 <= grade <= 1):
            logger.error(
                f'❌ [Error] Invalid grade value {grade} for job {index}')
            return index, 0.0, pred_ans, gold_ans

        return index, float(grade), pred_ans, gold_ans

    except TimeoutError as te:
        logger.warning(f'⏰ [Timeout] Job {index} timed out after {te} seconds')
        return index, 0.0, 'Timeout', 'Timeout'
    except ValueError as ve:
        logger.error(
            f'❌ [Value Error] Invalid input format for job {index}: {ve}')
        return index, 0.0, f'Format Error: {ve}', None
    except Exception as e:
        logger.error(
            f'❌ [Error] An unexpected error occurred for job {index}: {e}',
            exc_info=True)
        return index, 0.0, f'Error: {e}', f'Error: {e}'


def compute_scores(eval_dataset: List[Dict[str, Any]],
                   label_key: str,
                   response_key: str,
                   cache_path: str,
                   max_workers: int,
                   timeout: int = 20) -> float:
    """
    Computes accuracy scores for a batch of mathematical evaluation jobs using parallel processing.

    This function orchestrates the parallel evaluation process:
    1. Validates input parameters and dataset
    2. Optimizes worker count based on system resources and workload
    3. Processes jobs in parallel with timeout protection
    4. Tracks statistics (correct answers, timeouts, errors)
    5. Saves detailed results to cache
    6. Provides comprehensive logging and progress tracking

    Args:
        eval_dataset (List[Dict[str, Any]]): Evaluation dataset where each dictionary contains:
            - task: Task identifier (required)
            - model output and ground truth fields (specified by label_key and response_key)
            - Other optional metadata
        label_key (str): Dictionary key for accessing ground truth answers
        response_key (str): Dictionary key for accessing model-generated answers
        cache_path (str): File system path for saving processed results
        max_workers (int): Upper limit on parallel worker processes
        timeout (int, optional): Maximum seconds allowed per job. Defaults to 20.

    Returns:
        float: Average accuracy score across all processed jobs (0.0 to 1.0)

    Raises:
        ValueError: On empty dataset or missing required data fields
        IOError: When cache file cannot be written
        RuntimeError: On critical parallel processing failures

    Note:
        Results are cached in JSONL format with additional metadata including:
        - Performance statistics (correct/timeout/error counts)
        - Processing parameters (workers, timeout)
        - Individual job results and extracted answers
    """
    if not eval_dataset:
        logger.info('No jobs to process. Returning 0.0 accuracy.')
        return 0.0

    total = len(eval_dataset)
    processed_indices = set()
    counts = {'timeout': 0, 'error': 0, 'correct': 0}

    # Optimize worker count based on CPU count and dataset size
    cpu_count = os.cpu_count() or 1
    optimal_workers = min(max_workers, max(1, min(cpu_count - 1, total // 4)))

    with tqdm(total=total, desc='Processing jobs', unit='job') as pbar:
        with ProcessPool(max_workers=optimal_workers) as pool:
            # `pool.map` submits jobs and returns a future
            future = pool.map(process_answers,
                              [(i, data, label_key, response_key)
                               for i, data in enumerate(eval_dataset)],
                              timeout=timeout)

            # Iterate over the results as they become available.
            iterator = future.result()
            while True:
                try:
                    result = next(iterator)
                    if result is not None:
                        idx, is_correct, extracted_answer, extracted_gold = result

                        # Update results atomically
                        eval_dataset[idx].update({
                            'accuracy':
                            is_correct,
                            'extracted_gold':
                            extracted_gold,
                            'extracted_answer':
                            extracted_answer
                        })
                        processed_indices.add(idx)

                        # Count different types of results
                        if is_correct == 1.0:
                            counts['correct'] += 1
                        elif extracted_answer == 'Timeout':
                            counts['timeout'] += 1
                        elif extracted_answer and (
                                isinstance(extracted_answer, str)
                                and extracted_answer.startswith('Error')):
                            counts['error'] += 1
                except StopIteration:
                    break
                except TimeoutError:
                    # Handle global timeout for the entire operation
                    logger.warning('Global timeout reached for processing')
                    break
                except Exception as e:
                    # Catch exceptions from the iterator, e.g., if a worker fails.
                    logger.error(
                        f'❌ An error occurred while retrieving a result: {e}')
                    # We can't identify the specific job, so we continue.
                finally:
                    pbar.update(1)

    # Handle any jobs that were not processed (e.g., due to a process crash or other unforeseen error).
    for idx in range(total):
        if idx not in processed_indices:
            eval_dataset[idx].update({
                'accuracy': 0.0,
                'extracted_gold': 'Error',
                'extracted_answer': 'Error'
            })

    logger.info(f'Summary: {total} eval_dataset processed.')

    # Log detailed performance statistics
    correct_rate = counts['correct'] / total * 100
    timeout_rate = counts['timeout'] / total * 100
    error_rate = counts['error'] / total * 100

    logger.info(f"""
    Performance Summary:
    -------------------
    Total Jobs: {total}
    Correct: {counts['correct']} ({correct_rate:.1f}%)
    Timeouts: {counts['timeout']} ({timeout_rate:.1f}%)
    Errors: {counts['error']} ({error_rate:.1f}%)
    Workers Used: {optimal_workers}
    """)

    # Save results with performance metadata
    metadata = {
        'total_jobs': total,
        'correct_count': counts['correct'],
        'timeout_count': counts['timeout'],
        'error_count': counts['error'],
        'workers_used': optimal_workers,
        'timeout_setting': timeout
    }

    for dataset in eval_dataset:
        dataset['_metadata'] = metadata

    # Save the results to the cache file
    save_cache(eval_dataset, cache_path)

    # Calculate and return the average accuracy
    accuracy = mean(data['accuracy'] for data in eval_dataset)
    logger.info(f'Final Accuracy: {accuracy:.2%}')
    return accuracy


def save_cache(eval_dataset: List[Dict[str, Any]], cache_path: str) -> None:
    """
    Persists evaluation results and metadata to a JSONL file.

    Each line in the output file contains a JSON object representing one evaluation
    result, including the original input, computed metrics, and processing metadata.

    Args:
        eval_dataset (List[Dict[str, Any]]): Evaluation results to save, where each dict contains:
            - Original input data
            - Computed accuracy score
            - Extracted answers (predicted and gold)
            - Processing metadata (e.g., worker count, timeout settings)
        cache_path (str): Filesystem path for the output JSONL file

    Raises:
        IOError: If the cache file cannot be written due to permissions or disk space

    Note:
        - Uses UTF-8 encoding for proper handling of mathematical symbols
        - Preserves full floating-point precision in JSON serialization
        - Ensures one complete JSON object per line for easy streaming
    """
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            for dataset in eval_dataset:
                f.write(json.dumps(dataset, ensure_ascii=False) + '\n')
        logger.info(f'✅ Results successfully saved to {cache_path}')
    except IOError as e:
        logger.error(f'❌ Failed to save cache file to {cache_path}: {e}')
