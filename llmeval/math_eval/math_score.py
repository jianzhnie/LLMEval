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

from llmeval.math_eval.utils_parser import parse_ground_truth

# Attempt to import necessary components from math-verify.
# If the package is not installed, it provides instructions to the user.
try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print(
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
    index, input_data = args
    data_name = input_data['task'].split('/')[1]

    # Parse the ground truth answer from the input data
    # The first return value (cot_answer) is unused for this metric
    try:
        _, gold_answer_text = parse_ground_truth(input_data, data_name)
    except (ValueError, NotImplementedError) as e:
        print(f'❌ [Error] Parsing gold truth for job {index} failed: {e}')
        return index, 0.0, None, None

    # Get the generated text. Handles cases where 'gen' might be missing or empty.
    generated_text = (input_data['gen'][0]
                      if 'gen' in input_data and input_data['gen'] else '')

    # Initialize the verification function from math_verify
    verify_func = math_metric(
        # The gold answer is expected to be in LaTeX format
        gold_extraction_target=(LatexExtractionConfig(), ),
        # The predicted answer can be an expression or LaTeX
        pred_extraction_target=(ExprExtractionConfig(),
                                LatexExtractionConfig()),
        # Use max to select the best score if multiple extractions are successful
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
        pred_ans = extracted_answers[1] if len(extracted_answers) > 1 else None
        gold_ans = extracted_answers[0] if len(extracted_answers) > 0 else None

        return index, float(grade), pred_ans, gold_ans

    except TimeoutError as te:
        # This handles the case where the `math_verify` function times out
        print(f'⏰ [Timeout] Job {index} failed: {te}')
        return index, 0.0, 'Timeout', None
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f'❌ [Error] Job {index} failed: {e}')
        return index, 0.0, f'Error: {str(e)}', None


def compute_scores(jobs: List[Dict[str, Any]], max_workers: int,
                   cache_path: str) -> float:
    """
    Computes accuracy scores for a list of jobs using a multiprocessing pool.

    Args:
        jobs (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                     represents a job (model-generated output).
        max_workers (int): The maximum number of worker processes to use.
        cache_path (str): The file path to save the processed results.

    Returns:
        float: The overall accuracy score, averaged across all jobs.
    """
    total = len(jobs)
    processed_indices = set()

    with tqdm(total=total) as pbar:
        # Using ProcessPool to parallelize the evaluation
        with ProcessPool(max_workers=max_workers) as pool:
            # `pool.map` submits jobs and returns a future
            future = pool.map(process_answers,
                              list(enumerate(jobs)),
                              timeout=10)  # Set a timeout for each task

            # Iterate over the results as they become available
            for result in future.result():
                if result is not None:
                    idx, is_correct, extracted_ans, gold_ans = result
                    jobs[idx]['accuracy'] = is_correct
                    jobs[idx]['extracted_answer'] = extracted_ans
                    jobs[idx]['gold_answer'] = gold_ans
                    processed_indices.add(idx)
                pbar.update(1)

    # Handle any jobs that were not processed (e.g., due to timeout)
    for idx in range(total):
        if idx not in processed_indices:
            jobs[idx]['accuracy'] = 0.0
            jobs[idx]['extracted_answer'] = 'Timeout'
            jobs[idx]['gold_answer'] = ''
            jobs[idx]['timeout_cnt'] = jobs[idx].get('timeout_cnt', 0) + 1

    # Save the results to the cache file
    save_cache(jobs, cache_path)

    # Calculate the average accuracy
    if not jobs:
        return 0.0
    accuracy = mean(job['accuracy'] for job in jobs)
    return accuracy


def save_cache(jobs: List[Dict[str, Any]], cache_path: str) -> None:
    """
    Saves a list of job dictionaries to a JSONL file.

    Args:
        jobs (List[Dict[str, Any]]): The list of dictionaries to save.
        cache_path (str): The file path to save the data.
    """
    with open(cache_path, 'w', encoding='utf-8') as g:
        for job in jobs:
            # Use json.dumps to write each dictionary as a JSON line
            g.write(json.dumps(job, ensure_ascii=False) + '\n')
