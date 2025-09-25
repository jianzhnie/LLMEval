"""
This module provides functionality to evaluate the accuracy of model-generated mathematical answers
against their ground truth using the `math-verify` library. It leverages multiprocessing to speed
up the evaluation process and includes robust error handling and caching mechanisms.

The module implements a parallel processing architecture to efficiently handle large batches of
mathematical evaluation tasks while providing detailed logging and progress tracking.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth
from llmeval.utils.logger import init_logger

# Configure a dedicated logger for the math scoring module
logger = init_logger('math_score')

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

# Type aliases for better code readability
ProcessResult = Optional[Tuple[int, float, Optional[str], Optional[str]]]
DataDict = Dict[str, Any]
EvalDataset = List[DataDict]


@dataclass
class ProcessingStats:
    """Container for tracking processing statistics."""
    total: int = 0
    correct: int = 0
    timeout: int = 0
    error: int = 0

    @property
    def correct_rate(self) -> float:
        """Calculate percentage of correct answers."""
        return (self.correct / self.total * 100) if self.total > 0 else 0.0

    @property
    def timeout_rate(self) -> float:
        """Calculate percentage of timeouts."""
        return (self.timeout / self.total * 100) if self.total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate percentage of errors."""
        return (self.error / self.total * 100) if self.total > 0 else 0.0


def process_answers(args: Tuple[int, DataDict, str, str]) -> ProcessResult:
    """
    Process a single model output by extracting and comparing with ground truth.

    This function handles:
    1. Task name extraction and validation
    2. Ground truth parsing
    3. Model output processing
    4. Answer verification
    5. Error handling and timeout management

    Args:
        args: Processing arguments containing:
            - index: Unique job identifier
            - input_data: Data dictionary with model output and ground truth
            - label_key: Key for ground truth in input_data
            - response_key: Key for model response in input_data

    Returns:
        A tuple containing:
            - Original job index
            - Verification score (1.0=correct, 0.0=incorrect)
            - Extracted predicted answer (None if failed)
            - Extracted gold answer (None if failed)

    Note:
        Uses math-verify library with configurable precision for answer verification.
    """
    index, input_data, label_key, response_key = args

    # Validate and extract task name
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
            gold_ans = extracted_answers[0]
            pred_ans = extracted_answers[1]
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


def compute_scores(eval_dataset: EvalDataset, label_key: str,
                   response_key: str, cache_path: str, max_workers: int,
                   timeout: int) -> float:
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
    stats = ProcessingStats(total=total)
    processed_indices = set()

    # Optimize worker count based on system resources
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

                        # Update statistics
                        if is_correct == 1.0:
                            stats.correct += 1
                        elif extracted_answer == 'Timeout':
                            stats.timeout += 1
                        elif isinstance(
                                extracted_answer,
                                str) and extracted_answer.startswith('Error'):
                            stats.error += 1
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

    # Log performance summary
    logger.info(f'''
    Performance Summary:
    -------------------
    Total Jobs: {stats.total}
    Correct: {stats.correct} ({stats.correct_rate:.1f}%)
    Timeouts: {stats.timeout} ({stats.timeout_rate:.1f}%)
    Errors: {stats.error} ({stats.error_rate:.1f}%)
    Workers Used: {optimal_workers}
    ''')

    # Add metadata and save results
    metadata = {
        'total_jobs': stats.total,
        'correct_count': stats.correct,
        'timeout_count': stats.timeout,
        'error_count': stats.error,
        'workers_used': optimal_workers,
        'timeout_setting': timeout
    }

    logger.debug(f'Processing metadata: {metadata}')
    # Save the results to the cache file
    save_cache(eval_dataset, cache_path)

    # Calculate and return the average accuracy
    accuracy = mean(data['accuracy'] for data in eval_dataset)
    logger.info(f'Final Accuracy: {accuracy:.4f}')
    return accuracy


def save_cache(eval_dataset: EvalDataset, cache_path: str) -> None:
    """
    Save evaluation results and metadata to a JSONL file.

    Args:
        eval_dataset: Evaluation results to save
        cache_path: Output file path for JSONL data

    Raises:
        IOError: If the cache file cannot be written
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            for dataset in eval_dataset:
                f.write(json.dumps(dataset, ensure_ascii=False) + '\n')
        logger.info(f'✅ Results saved to {cache_path}')
    except IOError as e:
        logger.error(f'❌ Failed to save cache: {e}')
        raise
