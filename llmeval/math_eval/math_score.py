import json
from concurrent.futures import TimeoutError
from statistics import mean
from typing import Any, List, Tuple

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.math_eval.utils_parser import parse_ground_truth

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print(
        'To use Math-Verify, please install it first by running `pip install math-verify`.'
    )


def process_answers(args) -> Tuple[int, float, Any, Any]:
    """Process each answer through the sympy extraction workflow and compare with gold using math_verify."""
    index, input_data = args

    data_name = input_data['task'].split('/')[1]
    cot_answer, answer = parse_ground_truth(input_data, data_name)

    gen_text = (input_data['gen'][0]
                if 'gen' in input_data and len(input_data['gen']) > 0 else '')

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ),
        pred_extraction_target=(ExprExtractionConfig(),
                                LatexExtractionConfig()),
        aggregation_function=max,
        precision=6,
    )

    try:
        grade, extracted_answers = verify_func([gen_text], [answer])
        pred_ans = extracted_answers[1] if extracted_answers else None
        gold_ans = extracted_answers[0] if extracted_answers else None

        return index, float(grade), pred_ans, gold_ans

    except TimeoutError as te:
        print(f'[Timeout] Job {index} failed: {te}')
        return index, 0.0, 'Timeout', None

    except Exception as e:
        print(f'[Error] Job {index} failed: {e}')
        return index, 0.0, f'Error: {str(e)}', None


def compute_scores(jobs: List[dict], max_workers: int,
                   cache_path: str) -> float:
    results = []
    total = len(jobs)
    with tqdm(total=total) as pbar:
        with ProcessPool(max_workers=max_workers) as pool:
            future = pool.map(process_answers,
                              list(enumerate(jobs)),
                              timeout=10)
            for result in future.result():
                if result is not None:
                    idx, is_correct, extracted_ans, gold_ans = result
                    jobs[idx]['accuracy'] = is_correct
                    jobs[idx]['extracted_answer'] = extracted_ans
                    jobs[idx]['gold_answer'] = gold_ans
                    jobs[idx].pop('timeout_cnt', None)
                pbar.update(1)
                results.append(result)

    # Handle any missing results (e.g., due to timeout or crash)
    for idx, job in enumerate(jobs):
        if 'accuracy' not in job:
            job['accuracy'] = 0.0
            job['extracted_answer'] = 'Timeout'
            job['gold_answer'] = ''
            job['timeout_cnt'] = 1

    save_cache(jobs, cache_path)
    accuracy = mean(x['accuracy'] for x in jobs)
    return accuracy


def save_cache(jobs: List[dict], cache_path: str):
    with open(cache_path, 'w', encoding='utf-8') as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + '\n')
        g.flush()
