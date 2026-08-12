"""
This module provides functionality to evaluate the accuracy of model-generated mathematical answers
against their ground truth using the `math-verify` library. It leverages multiprocessing to speed
up the evaluation process and includes robust error handling.

The module implements a parallel processing architecture to efficiently handle large batches of
mathematical evaluation tasks while providing detailed logging and progress tracking.
"""

from __future__ import annotations

import re
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth
from llmeval.tasks.postprocess import (
    DEFAULT_FILTER_REGISTRY,
    TextFilterPipeline,
    normalize_single_generation_samples,
    resolve_max_workers,
)
from llmeval.tasks.registry import ScorerResult
from llmeval.utils.log import init_logger

# Configure a dedicated logger for the math scoring module
logger = init_logger("math_score")

# Python < 3.11: builtins.TimeoutError and concurrent.futures.TimeoutError
# are distinct classes.  Catch both so timeouts from math-verify internals
# (builtin) and from pebble (concurrent.futures) are handled identically.
# In Python 3.11+ they are the same class, so the tuple deduplicates naturally.
_TIMEOUT_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConcurrentTimeoutError)


def _is_math_task_name(task_name: Any) -> bool:
    """Accept the registered family itself or one of its named tasks."""
    return isinstance(task_name, str) and (
        task_name == "math_opensource" or task_name.startswith("math_opensource/")
    )


try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e}\n"
        "To use Math-Verify, install:  "
        "pip install math-verify>=1.0.0 pebble>=4.6.3 tqdm>=4.65.0"
    ) from e

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ImportError:  # pragma: no cover - optional fallback dependency
    sympy = None
    parse_latex = None

# Pre-built metric instance — reused across all items (stateless config).
# Both expression and LaTeX parsers are enabled for robust answer extraction.
_verify_func = math_metric(
    gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    aggregation_function=max,
    precision=6,
)

MATH_RESPONSE_PIPELINE: TextFilterPipeline = DEFAULT_FILTER_REGISTRY.build_pipeline(
    "math_response", "strip_reasoning"
)

INVALID_ANSWER = "[invalidanswer]"
_FINAL_ANSWER_RE = re.compile(
    r"Final Answer:\s*The final answer is(.*?)(?:\.?\s*I hope it is correct\.)?$",
    re.IGNORECASE | re.DOTALL,
)

_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
)
_REMOVED_EXPRESSIONS: tuple[str, ...] = (
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "\\text{s}",
    "\\text{.}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
)


@dataclass(frozen=True)
class MathAnswerResult:
    """Worker result with answer and compact postprocessing diagnostics."""

    index: int
    grade: float
    predicted: str | None
    gold: str | None
    fallback_matched: bool = False
    failed: bool = False
    filter_trace: dict[str, Any] = field(default_factory=dict)


def _last_boxed_only_string(text: str) -> str:
    """Return the last boxed/fboxed expression from a string."""
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return INVALID_ANSWER

    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

    if right_brace_idx is None:
        return INVALID_ANSWER
    return text[idx : right_brace_idx + 1]


def _remove_boxed(text: str) -> str:
    """Remove a leading boxed/fboxed wrapper."""
    if text.startswith("\\boxed "):
        return text[len("\\boxed ") :]
    for prefix in ("\\boxed{", "\\fbox{"):
        if text.startswith(prefix) and text.endswith("}"):
            return text[len(prefix) : -1]
    return INVALID_ANSWER


def _get_unnormalized_answer(text: str) -> str:
    """Extract the explicit Minerva-style final answer when present."""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def _normalize_final_answer(final_answer: str) -> str:
    """Normalize a final math answer using harness-style cleanup rules."""
    final_answer = final_answer.split("=")[-1]

    for before, after in _SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expression in _REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expression, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def _normalize_math_text(text: Any) -> str:
    """Normalize a math answer into a compact comparison string."""
    normalized = str(text).strip()
    if not normalized:
        return ""

    explicit_answer = _get_unnormalized_answer(normalized)
    if explicit_answer != INVALID_ANSWER:
        normalized = explicit_answer
    else:
        boxed = _last_boxed_only_string(normalized)
        if boxed != INVALID_ANSWER:
            unboxed = _remove_boxed(boxed)
            if unboxed != INVALID_ANSWER:
                normalized = unboxed

    normalized = _normalize_final_answer(normalized)
    return "" if normalized == INVALID_ANSWER else normalized


def _math_text_equiv(gold_text: Any, pred_text: Any) -> bool:
    """Return whether two math answers are equivalent under the fallback path."""
    gold_norm = _normalize_math_text(gold_text)
    pred_norm = _normalize_math_text(pred_text)

    if not gold_norm or not pred_norm:
        return False
    if gold_norm == pred_norm:
        return True

    if sympy is None or parse_latex is None:
        return False

    try:
        gold_expr = parse_latex(gold_norm)
        pred_expr = parse_latex(pred_norm)
        return bool(sympy.simplify(gold_expr - pred_expr) == 0)
    except Exception:
        return False


def _process_answers_impl(
    args: tuple[int, dict[str, Any], str, str],
) -> MathAnswerResult:
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
        A ``MathAnswerResult`` carrying the original job index, the
        verification grade (1.0=correct, 0.0=incorrect), the extracted
        predicted and gold answers (None when extraction failed), plus
        fallback/filter diagnostics.

    Note:
        Uses math-verify library with configurable precision for answer verification.
    """
    index, input_data, label_key, response_key = args

    # Family-only names are valid registry inputs and use generic gold parsing.
    task_name = input_data.get("task", "")
    if not _is_math_task_name(task_name):
        logger.warning("Invalid task format for job %d", index)
        return MathAnswerResult(index, 0.0, None, None, failed=True)
    _, _, data_name = task_name.partition("/")

    # Parse the ground truth answer from the input data
    # The first return value (cot_answer) is unused for this metric
    try:
        _, gold_answer_text = parse_ground_truth(input_data, data_name, label_key)
    except (ValueError, NotImplementedError, KeyError) as e:
        logger.error("Parsing gold truth for job %d failed: %s", index, e)
        return MathAnswerResult(index, 0.0, None, None, failed=True)

    inference_error = input_data.get("error")
    if inference_error:
        return MathAnswerResult(index, 0.0, None, gold_answer_text, failed=True)

    # Get the generated text. Handles cases where response might be missing or empty.
    generated_text = input_data.get(response_key, [])
    if not generated_text:
        logger.warning("No generated text found for job %d", index)
        return MathAnswerResult(index, 0.0, None, gold_answer_text)
    raw_generated_text = (
        generated_text[0] if isinstance(generated_text, list) else str(generated_text)
    )
    generated_text, filter_trace = MATH_RESPONSE_PIPELINE.apply_with_trace(
        raw_generated_text
    )

    def result(
        grade: float,
        predicted: str | None,
        gold: str | None,
        *,
        fallback_matched: bool = False,
        failed: bool = False,
    ) -> MathAnswerResult:
        return MathAnswerResult(
            index,
            grade,
            predicted,
            gold,
            fallback_matched=fallback_matched,
            failed=failed,
            filter_trace=filter_trace,
        )

    try:
        grade, extracted_answers = _verify_func([gold_answer_text], [generated_text])

        if not extracted_answers:
            if _math_text_equiv(gold_answer_text, generated_text):
                logger.debug("Fallback normalization matched job %d", index)
                return result(
                    1.0, generated_text, gold_answer_text, fallback_matched=True
                )
            logger.warning(
                "No answers could be extracted for job %d; fallback did not match",
                index,
            )
            return result(0.0, None, gold_answer_text)

        # Extract answers with validation
        gold_ans = extracted_answers[0] if len(extracted_answers) > 0 else None
        pred_ans = extracted_answers[1] if len(extracted_answers) > 1 else None

        # Validate grade value
        if not (isinstance(grade, int | float) and 0 <= grade <= 1):
            logger.error("Invalid grade value %r for job %d", grade, index)
            return result(0.0, pred_ans, gold_ans, failed=True)

        return result(float(grade), pred_ans, gold_ans)

    # Note: Pebble enforces timeouts at the pool level (terminating subprocess),
    # so a timeout here is a safety net for timeouts from math_verify internals.
    except _TIMEOUT_ERRORS:
        logger.warning("Job %d timed out", index)
        return result(0.0, None, gold_answer_text, failed=True)
    except Exception as e:
        if _math_text_equiv(gold_answer_text, generated_text):
            logger.debug("Fallback normalization matched job %d after %s", index, e)
            return result(1.0, generated_text, gold_answer_text, fallback_matched=True)
        logger.warning("Math verification failed for job %d: %s", index, e)
        return result(0.0, None, gold_answer_text, failed=True)


def process_answers(
    args: tuple[int, dict[str, Any], str, str],
) -> MathAnswerResult:
    """Return an indexed failure sentinel if any worker-stage operation crashes."""
    index = args[0]
    try:
        return _process_answers_impl(args)
    except Exception as exc:
        logger.warning("Math scoring worker failed for job %d: %s", index, exc)
        return MathAnswerResult(index, 0.0, None, None, failed=True)


def _score_math_records(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    max_workers: int,
    timeout: int,
) -> float:
    """Score normalized math rows in place and return sample-level accuracy."""
    if not eval_dataset:
        logger.info("No jobs to process. Returning 0.0 accuracy.")
        return 0.0

    total = len(eval_dataset)
    correct_count = 0
    failed_count = 0
    processed_indices = set()

    optimal_workers = resolve_max_workers(total, max_workers)

    with (
        tqdm(total=total, desc="Processing jobs", unit="job") as pbar,
        ProcessPool(max_workers=optimal_workers) as pool,
    ):
        # `pool.map` submits jobs and returns a future
        future = pool.map(
            process_answers,
            [(i, data, label_key, response_key) for i, data in enumerate(eval_dataset)],
            timeout=timeout,
        )

        # Iterate over the results as they become available.
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except _TIMEOUT_ERRORS:
                logger.warning("Individual task timed out; marked as failed")
                pbar.update(1)
                continue
            except Exception as e:
                logger.error("Failed to retrieve math scoring result: %s", e)
                pbar.update(1)
                continue

            pbar.update(1)
            if result is not None:
                idx = result.index
                is_correct = result.grade
                extracted_answer = result.predicted
                extracted_gold = result.gold
                status = "failed" if result.failed else "completed"

                eval_dataset[idx].update(
                    {
                        "accuracy": is_correct,
                        "extracted_gold": extracted_gold,
                        "extracted_answer": extracted_answer,
                        "evaluation_status": status,
                        "fallback_matched": result.fallback_matched,
                        "filter_trace": result.filter_trace,
                    }
                )
                processed_indices.add(idx)

                if status == "completed" and is_correct == 1.0:
                    correct_count += 1
                elif status == "failed":
                    failed_count += 1

    # Handle any jobs that were not processed. Workers convert their own
    # errors into sentinel results, so a missing index means the pool-level
    # timeout killed the worker (or an unrecoverable crash).
    for idx in range(total):
        if idx not in processed_indices:
            eval_dataset[idx].update(
                {
                    "accuracy": 0.0,
                    "extracted_gold": None,
                    "extracted_answer": None,
                    "evaluation_status": "failed",
                    **_filter_artifacts(eval_dataset[idx].get(response_key)),
                }
            )
            failed_count += 1

    logger.info(f"Summary: {total} eval_dataset processed.")

    effective_count = total - failed_count
    correct_rate = correct_count / effective_count * 100 if effective_count else 0.0
    failed_rate = failed_count / total * 100
    logger.info(
        "Math scoring: total=%d effective=%d correct=%d (%.1f%%) "
        "failed=%d (%.1f%%) workers=%d",
        total,
        effective_count,
        correct_count,
        correct_rate,
        failed_count,
        failed_rate,
        optimal_workers,
    )

    # Calculate and return the average accuracy
    eligible_accuracy = [
        float(data["accuracy"])
        for data in eval_dataset
        if data.get("evaluation_status", "completed") == "completed"
    ]
    accuracy = mean(eligible_accuracy) if eligible_accuracy else 0.0
    logger.info(f"Final Accuracy: {accuracy:.4f}")
    return accuracy


def score_math_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    max_workers: int,
    timeout: int,
) -> ScorerResult:
    """Score math generations and return the shared structured result."""
    # Each inference row carries one generation; normalize its representation
    # before scoring and problem-level aggregation.
    scoring_dataset = _normalize_math_samples(eval_dataset, label_key, response_key)
    accuracy = _score_math_records(
        eval_dataset=scoring_dataset,
        label_key=label_key,
        response_key=response_key,
        max_workers=max_workers,
        timeout=timeout,
    )
    observations = [
        float(item.get("accuracy", 0.0))
        for item in scoring_dataset
        if item.get("evaluation_status", "completed") == "completed"
    ]
    failed_count = sum(
        item.get("evaluation_status") == "failed" for item in scoring_dataset
    )
    wrong_answer_count = sum(
        item.get("evaluation_status", "completed") == "completed"
        and float(item.get("accuracy", 0.0)) != 1.0
        for item in scoring_dataset
    )
    details, extra_metrics, problem_observations = _build_problem_level_metrics(
        scoring_dataset
    )
    complete_problem_count = sum(1 for problem in details if problem["complete"])
    metrics = {"accuracy": accuracy, **extra_metrics}
    return ScorerResult(
        metrics=metrics,
        observations={
            "accuracy": observations,
            **problem_observations,
        },
        records=[dict(item) for item in scoring_dataset],
        details={
            "problem_level": details,
            "complete_problem_count": complete_problem_count,
            "incomplete_problem_count": len(details) - complete_problem_count,
            "wrong_answer_count": wrong_answer_count,
            "excluded_problem_doc_ids": [
                problem["doc_id"] for problem in details if not problem["complete"]
            ],
        },
        sample_count=len(scoring_dataset),
        effective_sample_count=len(scoring_dataset) - failed_count,
        failed_count=failed_count,
    )


def _normalize_math_samples(
    eval_dataset: list[dict[str, Any]], label_key: str, response_key: str
) -> list[dict[str, Any]]:
    """Validate and normalize one math generation per input row.

    Repeated rows remain independent samples, including rows with identical
    responses. Rows that repeat a ``doc_id`` with a conflicting gold answer
    or prompt raise ``ValueError``.
    """
    return normalize_single_generation_samples(
        eval_dataset,
        response_key,
        problem_identity=_problem_identity,
        conflict_keys=(label_key, "prompt"),
        record_kind="math document",
    )


def _problem_identity(item: dict[str, Any], row_index: int) -> str:
    """Return a stable grouping key without merging equal prompt text."""
    document_id = item.get("doc_id")
    if document_id is not None and str(document_id).strip():
        return f"doc:{document_id}"
    return f"row:{row_index}"


def _majority_cluster(items: list[dict[str, Any]]) -> tuple[str | None, bool]:
    """Vote over math-equivalent answers, preserving deterministic tie order."""
    clusters: list[tuple[str, list[dict[str, Any]]]] = []
    for item in items:
        answer = item.get("extracted_answer")
        if answer in (None, ""):
            answer = INVALID_ANSWER
        answer_text = str(answer)
        if answer_text.startswith(("Error", "Format Error", "Timeout")):
            continue
        for position, (representative, members) in enumerate(clusters):
            if answer_text == representative or _math_text_equiv(
                representative, answer_text
            ):
                clusters[position] = (representative, [*members, item])
                break
        else:
            clusters.append((answer_text, [item]))
    if not clusters:
        return None, False
    representative, members = max(
        enumerate(clusters), key=lambda entry: (len(entry[1][1]), -entry[0])
    )[1]
    return representative, any(
        float(item.get("accuracy", 0.0)) == 1.0 for item in members
    )


def _build_problem_level_metrics(
    scored_dataset: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, list[float]]]:
    """Aggregate sample outcomes into pass@k and majority-vote metrics.

    The number of rows sharing a ``doc_id`` defines that problem's sampling
    depth. A problem is complete only when every row was scored successfully;
    problem-level metrics include complete problems only.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row_index, item in enumerate(scored_dataset):
        problem_id = _problem_identity(item, row_index)
        grouped.setdefault(problem_id, []).append(item)

    problems: list[dict[str, Any]] = []
    for problem_id, items in grouped.items():
        completed = [
            item
            for item in items
            if item.get("evaluation_status", "completed") == "completed"
        ]
        correct_samples = sum(
            float(item.get("accuracy", 0.0)) == 1.0 for item in completed
        )
        majority_prediction, majority_correct = _majority_cluster(completed)
        sample_count = len(items)
        observed_samples = len(completed)
        complete = observed_samples == sample_count
        problems.append(
            {
                "doc_id": problem_id,
                "correct_samples": correct_samples,
                "sample_count": sample_count,
                "observed_samples": observed_samples,
                "complete": complete,
                "correct_fraction": correct_samples / sample_count
                if sample_count
                else 0.0,
                "passed": correct_samples > 0,
                "majority_prediction": majority_prediction,
                "majority_correct": majority_correct,
            }
        )

    if not problems:
        return [], {}, {}
    metrics: dict[str, float] = {}
    observations: dict[str, list[float]] = {}
    for k in sorted({problem["sample_count"] for problem in problems}):
        cohort = [
            problem
            for problem in problems
            if problem["sample_count"] == k and problem["complete"]
        ]
        pass_key = f"problem_pass@{k}"
        majority_key = f"problem_majority@{k}"
        pass_values = [float(problem["passed"]) for problem in cohort]
        majority_values = [float(problem["majority_correct"]) for problem in cohort]
        metrics[pass_key] = sum(pass_values) / len(cohort) if cohort else 0.0
        metrics[majority_key] = sum(majority_values) / len(cohort) if cohort else 0.0
        observations[pass_key] = pass_values
        observations[majority_key] = majority_values
    return problems, metrics, observations


def _filter_artifacts(response: Any) -> dict[str, Any]:
    """Return compact task-pipeline metadata for an unprocessed response."""
    raw_response = response[0] if isinstance(response, list) and response else response
    _, trace = MATH_RESPONSE_PIPELINE.apply_with_trace(raw_response)
    return {"filter_trace": trace}
