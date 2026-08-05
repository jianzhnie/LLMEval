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
import re
from collections.abc import Iterator
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth
from llmeval.tasks.postprocess import (
    DEFAULT_FILTER_REGISTRY,
    TextFilterPipeline,
)
from llmeval.tasks.results import ScorerResult
from llmeval.utils.log import init_logger

# Configure a dedicated logger for the math scoring module
logger = init_logger("math_score")


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
    "math_response", "1", "strip_reasoning"
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


@dataclass
class ProcessingStats:
    """Container for tracking processing statistics."""

    total: int = 0
    correct: int = 0
    timeout: int = 0
    error: int = 0
    skipped: int = 0

    @property
    def effective(self) -> int:
        """Number of samples eligible for the accuracy denominator."""
        return max(self.total - self.timeout - self.error - self.skipped, 0)

    @property
    def correct_rate(self) -> float:
        """Calculate percentage of correct answers."""
        return (self.correct / self.effective * 100) if self.effective > 0 else 0.0

    @property
    def timeout_rate(self) -> float:
        """Calculate percentage of timeouts."""
        return (self.timeout / self.total * 100) if self.total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate percentage of errors."""
        return (self.error / self.total * 100) if self.total > 0 else 0.0


@dataclass(frozen=True)
class MathAnswerResult:
    """Worker result with a non-breaking marker for successful fallback matching."""

    index: int
    grade: float
    predicted: str | None
    gold: str | None
    fallback_matched: bool = False

    def __iter__(self) -> Iterator[int | float | str | None]:
        yield self.index
        yield self.grade
        yield self.predicted
        yield self.gold


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


def process_answers(
    args: tuple[int, dict[str, Any], str, str],
) -> tuple[int, float, str | None, str | None] | MathAnswerResult | None:
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

    # Family-only names are valid registry inputs and use generic gold parsing.
    task_name = input_data.get("task", "")
    if not _is_math_task_name(task_name):
        logger.warning(f"⚠️ Invalid task format for job {index}")
        return index, 0.0, None, None
    _, _, data_name = task_name.partition("/")

    # Parse the ground truth answer from the input data
    # The first return value (cot_answer) is unused for this metric
    try:
        # The first return value (cot_answer) is unused for this metric.
        _, gold_answer_text = parse_ground_truth(input_data, data_name, label_key)
    except (ValueError, NotImplementedError, KeyError) as e:
        logger.error(f"❌ [Error] Parsing gold truth for job {index} failed: {e}")
        return index, 0.0, f"Error: {e}", None

    # Get the generated text. Handles cases where response might be missing or empty.
    generated_text = input_data.get(response_key, [])
    if not generated_text:
        logger.warning(f"⚠️ No generated text found for job {index}")
        return index, 0.0, None, None
    generated_text = (
        generated_text[0] if isinstance(generated_text, list) else str(generated_text)
    )
    generated_text = MATH_RESPONSE_PIPELINE.apply(generated_text)

    try:
        grade, extracted_answers = _verify_func([gold_answer_text], [generated_text])

        if not extracted_answers:
            if _math_text_equiv(gold_answer_text, generated_text):
                logger.debug("Fallback normalization matched job %d", index)
                return MathAnswerResult(
                    index, 1.0, generated_text, gold_answer_text, fallback_matched=True
                )
            logger.warning(
                "No answers could be extracted for job %d; fallback did not match",
                index,
            )
            return index, 0.0, None, None

        # Extract answers with validation
        try:
            gold_ans = extracted_answers[0] if len(extracted_answers) > 0 else None
            pred_ans = extracted_answers[1] if len(extracted_answers) > 1 else None
        except IndexError:
            logger.error(f"❌ [Error] Invalid extraction format for job {index}")
            return index, 0.0, None, None

        # Validate grade value
        if not (isinstance(grade, int | float) and 0 <= grade <= 1):
            logger.error(f"❌ [Error] Invalid grade value {grade} for job {index}")
            return index, 0.0, pred_ans, gold_ans

        return index, float(grade), pred_ans, gold_ans

    # Note: Pebble enforces timeouts at the pool level (terminating subprocess),
    # so TimeoutError here is a safety net for timeouts from math_verify internals.
    except TimeoutError:
        logger.warning(f"⏰ [Timeout] Job {index} timed out")
        return index, 0.0, "Timeout", "Timeout"
    except ValueError as ve:
        if _math_text_equiv(gold_answer_text, generated_text):
            logger.debug("Fallback normalization matched job %d after: %s", index, ve)
            return MathAnswerResult(
                index, 1.0, generated_text, gold_answer_text, fallback_matched=True
            )
        logger.warning("Math verification value error for job %d: %s", index, ve)
        return index, 0.0, f"Format Error: {ve}", None
    except Exception as e:
        if _math_text_equiv(gold_answer_text, generated_text):
            logger.debug("Fallback normalization matched job %d after %s", index, e)
            return MathAnswerResult(
                index, 1.0, generated_text, gold_answer_text, fallback_matched=True
            )
        logger.error(
            f"❌ [Error] An unexpected error occurred for job {index}: {e}",
            exc_info=True,
        )
        return index, 0.0, f"Error: {e}", f"Error: {e}"


def _math_record_status(
    item: dict[str, Any],
    label_key: str,
    response_key: str,
    extracted_answer: Any,
) -> str:
    """Classify a math record for the shared denominator contract.

    An unparseable model answer is a completed incorrect observation. Missing
    input fields are skipped, while verifier/worker errors and timeouts are
    excluded from the metric denominator.
    """
    if extracted_answer == "Timeout":
        return "timeout"
    if isinstance(extracted_answer, str) and extracted_answer.startswith(
        ("Error", "Format Error")
    ):
        return "failed"
    response = item.get(response_key)
    label = item.get(label_key)
    task_name = item.get("task")
    if (
        not response
        or label is None
        or (isinstance(label, str) and not label.strip())
        or not isinstance(task_name, str)
        or not _is_math_task_name(task_name)
    ):
        return "skipped"
    return "completed"


def compute_scores(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str,
    max_workers: int,
    timeout: int,
) -> float:
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
        logger.info("No jobs to process. Returning 0.0 accuracy.")
        return 0.0

    total = len(eval_dataset)
    stats = ProcessingStats(total=total)
    processed_indices = set()

    # Optimize worker count based on system resources.
    # Use min(total, max_workers, cpu_count-1) to avoid over-provisioning
    # for small datasets (e.g., AIME24 has only 30 items).
    cpu_count = os.cpu_count() or 1
    optimal_workers = min(total, max_workers, max(1, cpu_count - 1))

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
            except TimeoutError:
                # Handle timeout for individual task — skip and continue
                logger.warning("Individual task timed out, skipping and continuing")
                pbar.update(1)
                continue
            except Exception as e:
                # Catch exceptions from the iterator, e.g., if a worker fails.
                logger.error(f"❌ An error occurred while retrieving a result: {e}")
                # We can't identify the specific job, so we continue.
                pbar.update(1)
                continue

            pbar.update(1)
            if result is not None:
                fallback_matched = bool(getattr(result, "fallback_matched", False))
                idx, is_correct, extracted_answer, extracted_gold = result
                status = _math_record_status(
                    eval_dataset[idx], label_key, response_key, extracted_answer
                )

                # Update results atomically
                eval_dataset[idx].update(
                    {
                        "accuracy": is_correct,
                        "extracted_gold": extracted_gold,
                        "extracted_answer": extracted_answer,
                        "evaluation_status": status,
                        "fallback_matched": fallback_matched,
                        **_filter_artifacts(eval_dataset[idx].get(response_key)),
                    }
                )
                processed_indices.add(idx)

                # Update statistics
                if is_correct == 1.0:
                    stats.correct += 1
                if status == "timeout":
                    stats.timeout += 1
                elif status == "failed":
                    stats.error += 1
                elif status == "skipped":
                    stats.skipped += 1

    # Handle any jobs that were not processed. Workers convert their own
    # errors into sentinel results, so a missing index means the pool-level
    # timeout killed the worker (or an unrecoverable crash).
    for idx in range(total):
        if idx not in processed_indices:
            eval_dataset[idx].update(
                {
                    "accuracy": 0.0,
                    "extracted_gold": "Timeout",
                    "extracted_answer": "Timeout",
                    "evaluation_status": "timeout",
                    **_filter_artifacts(eval_dataset[idx].get(response_key)),
                }
            )
            stats.timeout += 1

    logger.info(f"Summary: {total} eval_dataset processed.")

    # Log performance summary
    logger.info(f"""
    Performance Summary:
    -------------------
    Total Jobs: {stats.total}
    Correct: {stats.correct} ({stats.correct_rate:.1f}%)
    Timeouts: {stats.timeout} ({stats.timeout_rate:.1f}%)
    Errors: {stats.error} ({stats.error_rate:.1f}%)
    Workers Used: {optimal_workers}
    """)

    # Add metadata and save results
    metadata = {
        "total_jobs": stats.total,
        "correct_count": stats.correct,
        "timeout_count": stats.timeout,
        "error_count": stats.error,
        "skipped_count": stats.skipped,
        "sample_count": stats.total,
        "effective_sample_count": stats.effective,
        "workers_used": optimal_workers,
        "timeout_setting": timeout,
    }

    logger.debug(f"Processing metadata: {metadata}")
    # Save the results to the cache file
    save_cache(eval_dataset, cache_path)

    # Calculate and return the average accuracy
    eligible_accuracy = [
        float(data["accuracy"])
        for data in eval_dataset
        if data.get("evaluation_status", "completed") == "completed"
    ]
    accuracy = mean(eligible_accuracy) if eligible_accuracy else 0.0
    save_summary(
        accuracy=accuracy,
        metadata=metadata,
        cache_path=cache_path,
    )
    logger.info(f"Final Accuracy: {accuracy:.4f}")
    return accuracy


def compute_score_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str,
    max_workers: int,
    timeout: int,
) -> ScorerResult:
    """Return math scores through the registry's structured scorer contract.

    :func:`compute_scores` remains the compatibility API and performs the
    established JSONL/summary persistence. This entry point builds the registry
    result from the in-memory annotated records, without reading those files
    back from disk.
    """
    accuracy = compute_scores(
        eval_dataset=eval_dataset,
        label_key=label_key,
        response_key=response_key,
        cache_path=cache_path,
        max_workers=max_workers,
        timeout=timeout,
    )
    observations = [
        float(item.get("accuracy", 0.0))
        for item in eval_dataset
        if item.get("evaluation_status", "completed") == "completed"
    ]
    timeout_count = sum(
        item.get("evaluation_status") == "timeout" for item in eval_dataset
    )
    failed_count = sum(
        item.get("evaluation_status") == "failed" for item in eval_dataset
    )
    skipped_count = sum(
        item.get("evaluation_status") == "skipped" for item in eval_dataset
    )
    return ScorerResult(
        metrics={"accuracy": accuracy},
        observations={"accuracy": observations},
        per_item=[dict(item) for item in eval_dataset],
        sample_count=len(eval_dataset),
        effective_sample_count=len(observations),
        failed_count=failed_count,
        skipped_count=skipped_count,
        timeout_count=timeout_count,
    )


def _filter_artifacts(response: Any) -> dict[str, Any]:
    """Return the raw response, filtered response, and task pipeline trace."""
    raw_response = response[0] if isinstance(response, list) and response else response
    filtered, trace = MATH_RESPONSE_PIPELINE.apply_with_trace(raw_response)
    return {
        "raw_gen": "" if raw_response is None else str(raw_response),
        "filtered_gen": filtered,
        "filter_trace": trace,
    }


def save_cache(eval_dataset: list[dict[str, Any]], cache_path: str) -> None:
    """
    Save evaluation results and metadata to a JSONL file.

    Args:
        eval_dataset: Evaluation results to save
        cache_path: Output file path for JSONL data

    Raises:
        IOError: If the cache file cannot be written
    """
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for dataset in eval_dataset:
                f.write(json.dumps(dataset, ensure_ascii=False) + "\n")
        logger.info(f"✅ Results saved to {cache_path}")
    except OSError as e:
        logger.error(f"❌ Failed to save cache: {e}")
        raise


def save_summary(
    accuracy: float,
    metadata: dict[str, Any],
    cache_path: str,
) -> None:
    """Save aggregated math metrics next to the JSONL cache."""
    summary_path = Path(cache_path).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": round(accuracy, 6),
                **metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
