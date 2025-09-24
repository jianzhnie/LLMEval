"""
This module provides functionality to evaluate the accuracy of a large number of
model-generated answers against their ground truth using the `math-verify` library.
It leverages multiprocessing to speed up the evaluation process.
"""

import json
from concurrent.futures import TimeoutError
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth
from llmeval.utils.logger import init_logger

# Initialize a logger for better error handling and debugging.
logger = init_logger(__name__)

# Attempt to import necessary components from math-verify.
# If the package is not installed, it provides instructions to the user.
try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    logger.error(
        'To use Math-Verify, please install it first by running `pip install math-verify`.'
    )
    # Exiting the program here is crucial as the main functions will fail without it.
    import sys
    sys.exit(1)


def process_answers(
    args: Tuple[int, Dict[str, Any]]
) -> Optional[Tuple[int, float, Optional[str], Optional[str]]]:
    """
    Processes a single model output by extracting the answer and comparing it with the
    ground truth using the `math-verify` metric.

    Args:
        args (Tuple[int, Dict[str, Any]]): A tuple containing the job index and
                                           the input data dictionary.

    Returns:
        Optional[Tuple[int, float, Optional[str], Optional[str]]]: A tuple containing
        the index, correctness score (1.0 for correct, 0.0 for incorrect),
        the extracted predicted answer, and the extracted gold answer.
        Returns None if an unexpected error occurs.
    """
    index, input_data, label_key, response_key = args
    data_name = input_data.get('task', '').split('/')[1]

    # Parse the ground truth answer from the input data
    # The first return value (cot_answer) is unused for this metric
    try:
        # The first return value (cot_answer) is unused for this metric.
        _, gold_answer_text = parse_ground_truth(input_data,
                                                 data_name,
                                                 label_key=label_key)
    except (ValueError, NotImplementedError) as e:
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
        # Run the verification. The `math_metric` returns a tuple of (grade, extracted_answers).
        grade, extracted_answers = verify_func([generated_text],
                                               [gold_answer_text])

        # Safely extract the predicted and gold answers from the returned tuple
        # The tuple will contain extracted answers in the order of `pred_extraction_target`
        # and `gold_extraction_target` respectively.
        # Note: The order of `extracted_answers` is (gold_ans, pred_ans).
        gold_ans = extracted_answers[0] if len(extracted_answers) > 0 else None
        pred_ans = extracted_answers[1] if len(extracted_answers) > 1 else None

        return index, float(grade), pred_ans, gold_ans

    except TimeoutError as te:
        logger.warning(f'⏰ [Timeout] Job {index} timed out {te}')
        return index, 0.0, 'Timeout', 'Timeout'
    except Exception as e:
        # Catch any other unexpected errors during processing.
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
    Computes accuracy scores for a list of jobs using a multiprocessing pool.

    Args:
        eval_dataset (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                     represents a job (model-generated output).
        label_key (str): The key in the input data dictionary that contains the ground truth.
        response_key (str): The key in the input data dictionary that contains the model output.
        cache_path (str): The file path to save the processed results.
        max_workers (int): The maximum number of worker processes to use.
        timeout (int): The maximum time (in seconds) to wait for each job to complete

    Returns:
        float: The overall accuracy score, averaged across all jobs.
    """
    total = len(eval_dataset)
    if total == 0:
        logger.info('No jobs to process. Returning 0.0 accuracy.')
        return 0.0

    processed_indices = set()
    timeout_count = 0

    with tqdm(total=total, desc='Processing jobs') as pbar:
        # Using ProcessPool to parallelize the evaluation
        with ProcessPool(max_workers=max_workers) as pool:
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
                        eval_dataset[idx]['accuracy'] = is_correct
                        eval_dataset[idx]['extracted_gold'] = extracted_gold
                        eval_dataset[idx][
                            'extracted_answer'] = extracted_answer
                        processed_indices.add(idx)
                        if is_correct == 0.0 and extracted_answer == 'Timeout':
                            timeout_count += 1
                except StopIteration:
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
            eval_dataset[idx]['accuracy'] = 0.0
            eval_dataset[idx]['extracted_gold'] = 'Error'
            eval_dataset[idx]['extracted_answer'] = 'Error'

    logger.info(
        f'Summary: {total} eval_dataset processed. {timeout_count} timed out.')

    # Save the results to the cache file.
    save_cache(eval_dataset, cache_path)

    # Calculate the average accuracy.
    accuracy = mean(data['accuracy'] for data in eval_dataset)
    return accuracy


def save_cache(eval_dataset: List[Dict[str, Any]], cache_path: str) -> None:
    """
    Saves a list of dataset dictionaries to a JSONL file.

    Args:
        eval_dataset (List[Dict[str, Any]]): The list of dictionaries to save.
        cache_path (str): The file path to save the data.
    """
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            for dataset in eval_dataset:
                f.write(json.dumps(dataset, ensure_ascii=False) + '\n')
        logger.info(f'✅ Results successfully saved to {cache_path}')
    except IOError as e:
        logger.error(f'❌ Failed to save cache file to {cache_path}: {e}')
