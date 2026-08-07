"""
This module provides functionality to evaluate the accuracy of model-generated mathematical answers
against their ground truth using the `math-verify` library. It leverages multiprocessing to speed
up the evaluation process and includes robust error handling and caching mechanisms.

The module implements a parallel processing architecture to efficiently handle large batches of
mathematical evaluation tasks while providing detailed logging and progress tracking.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth
from llmeval.tasks.persistence import (
    atomic_write_jsonl,
    persist_results,
)
from llmeval.tasks.postprocess import (
    DEFAULT_FILTER_REGISTRY,
    TextFilterPipeline,
    build_filter_artifacts,
    expand_single_generation_samples,
    resolve_max_workers,
)
from llmeval.tasks.registry import ScorerResult
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
    """Worker result with a non-breaking marker for successful fallback matching.

    ``filter_trace`` carries the full pipeline trace (including the raw input
    under ``filter_trace["raw"]``), so ``raw_gen`` is not shipped separately
    over the pool IPC.
    """

    index: int
    grade: float
    predicted: str | None
    gold: str | None
    fallback_matched: bool = False
    failure_stage: str = "none"
    failure_reason: str | None = None
    filtered_gen: str = ""
    filter_trace: dict[str, Any] = field(default_factory=dict)

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
        return MathAnswerResult(
            index, 0.0, None, None, failure_stage="input", failure_reason="invalid task"
        )
    _, _, data_name = task_name.partition("/")

    # Parse the ground truth answer from the input data
    # The first return value (cot_answer) is unused for this metric
    try:
        # The first return value (cot_answer) is unused for this metric.
        _, gold_answer_text = parse_ground_truth(input_data, data_name, label_key)
    except (ValueError, NotImplementedError, KeyError) as e:
        logger.error(f"❌ [Error] Parsing gold truth for job {index} failed: {e}")
        return MathAnswerResult(
            index,
            0.0,
            None,
            None,
            failure_stage="input",
            failure_reason=str(e),
        )

    # Get the generated text. Handles cases where response might be missing or empty.
    generated_text = input_data.get(response_key, [])
    if not generated_text:
        logger.warning(f"⚠️ No generated text found for job {index}")
        return MathAnswerResult(
            index,
            0.0,
            None,
            None,
            failure_stage="inference",
            failure_reason="missing generation",
        )
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
        failure_stage: str = "none",
        failure_reason: str | None = None,
    ) -> MathAnswerResult:
        return MathAnswerResult(
            index,
            grade,
            predicted,
            gold,
            fallback_matched=fallback_matched,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            filtered_gen=generated_text,
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
            return result(
                0.0,
                None,
                gold_answer_text,
                failure_stage="extraction",
                failure_reason="no answer extracted",
            )

        # Extract answers with validation
        try:
            gold_ans = extracted_answers[0] if len(extracted_answers) > 0 else None
            pred_ans = extracted_answers[1] if len(extracted_answers) > 1 else None
        except IndexError:
            logger.error(f"❌ [Error] Invalid extraction format for job {index}")
            return result(
                0.0,
                None,
                gold_answer_text,
                failure_stage="extraction",
                failure_reason="invalid extraction format",
            )

        # Validate grade value
        if not (isinstance(grade, int | float) and 0 <= grade <= 1):
            logger.error(f"❌ [Error] Invalid grade value {grade} for job {index}")
            return result(
                0.0,
                pred_ans,
                gold_ans,
                failure_stage="verification",
                failure_reason=f"invalid grade: {grade!r}",
            )

        return result(float(grade), pred_ans, gold_ans)

    # Note: Pebble enforces timeouts at the pool level (terminating subprocess),
    # so TimeoutError here is a safety net for timeouts from math_verify internals.
    except TimeoutError:
        logger.warning(f"⏰ [Timeout] Job {index} timed out")
        return result(
            0.0,
            "Timeout",
            "Timeout",
            failure_stage="verification",
            failure_reason="timeout",
        )
    except Exception as e:
        if _math_text_equiv(gold_answer_text, generated_text):
            logger.debug("Fallback normalization matched job %d after %s", index, e)
            return result(1.0, generated_text, gold_answer_text, fallback_matched=True)
        logger.warning("Math verification failed for job %d: %s", index, e)
        return result(
            0.0,
            None,
            gold_answer_text,
            failure_stage="verification",
            failure_reason=str(e),
        )


def _math_record_status(
    item: dict[str, Any],
    label_key: str,
    response_key: str,
    extracted_answer: Any,
    failure_stage: str = "none",
) -> str:
    """Classify a math record for the shared denominator contract.

    Missing or unparseable input fields are skipped (a dataset problem, not a
    model failure). Answer extraction, verification, worker errors, and
    timeouts are classified separately and excluded from the model-accuracy
    denominator; only successfully scored grade-0 answers count as wrong.
    """
    if extracted_answer == "Timeout":
        return "timeout"
    if failure_stage == "input":
        # Gold-truth parsing failed: the dataset row itself is unusable.
        return "skipped"
    if failure_stage in {"inference", "extraction", "verification"}:
        return "failed"
    response = item.get(response_key)
    label = item.get(label_key)
    task_name = item.get("task")
    if (
        label is None
        or (isinstance(label, str) and not label.strip())
        or not isinstance(task_name, str)
        or not _is_math_task_name(task_name)
    ):
        return "skipped"
    if not response:
        return "failed"
    return "completed"


def compute_scores(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str,
    max_workers: int,
    timeout: int,
    persist_legacy: bool = True,
) -> float:
    """
    Computes accuracy scores for a batch of mathematical evaluation jobs using parallel processing.

    This function orchestrates the parallel evaluation process:
    1. Validates input parameters and dataset
    2. Optimizes worker count based on system resources and workload
    3. Processes jobs in parallel with timeout protection
    4. Tracks statistics (correct answers, timeouts, errors)
    5. Saves detailed results to JSONL
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
        IOError: When the result file cannot be written
        RuntimeError: On critical parallel processing failures

    Note:
        Results are saved in JSONL format with additional metadata including:
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
    failure_counts: dict[str, int] = Counter()

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
                failure_stage = str(getattr(result, "failure_stage", "none"))
                failure_reason = getattr(result, "failure_reason", None)
                idx, is_correct, extracted_answer, extracted_gold = result
                status = _math_record_status(
                    eval_dataset[idx],
                    label_key,
                    response_key,
                    extracted_answer,
                    failure_stage,
                )

                # Update results atomically
                eval_dataset[idx].update(
                    {
                        "accuracy": is_correct,
                        "extracted_gold": extracted_gold,
                        "extracted_answer": extracted_answer,
                        "evaluation_status": status,
                        "failure_stage": failure_stage,
                        "failure_reason": failure_reason,
                        "fallback_matched": fallback_matched,
                        **build_filter_artifacts(
                            result.filter_trace.get("raw", ""),
                            result.filtered_gen,
                            result.filter_trace,
                        ),
                    }
                )
                processed_indices.add(idx)

                # Update statistics
                if is_correct == 1.0:
                    stats.correct += 1
                if status == "timeout":
                    stats.timeout += 1
                    failure_counts["verification_failed"] += 1
                elif status == "failed":
                    stats.error += 1
                    failure_counts[f"{failure_stage}_failed"] += 1
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
                    "failure_stage": "verification",
                    "failure_reason": "pool timeout",
                    **_filter_artifacts(eval_dataset[idx].get(response_key)),
                }
            )
            stats.timeout += 1
            failure_counts["verification_failed"] += 1

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
        "effective_sample_count": stats.effective,
        "workers_used": optimal_workers,
        "timeout_setting": timeout,
        "failure_counts": dict(failure_counts),
    }

    logger.debug(f"Processing metadata: {metadata}")
    # Calculate and return the average accuracy
    eligible_accuracy = [
        float(data["accuracy"])
        for data in eval_dataset
        if data.get("evaluation_status", "completed") == "completed"
    ]
    accuracy = mean(eligible_accuracy) if eligible_accuracy else 0.0
    if persist_legacy:
        persist_results(
            cache_path,
            eval_dataset,
            {"accuracy": round(accuracy, 6), **metadata},
        )
        logger.info("Results saved to %s", cache_path)
    logger.info(f"Final Accuracy: {accuracy:.4f}")
    return accuracy


def score_math_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str,
    max_workers: int,
    timeout: int,
    expected_samples: int | None = None,
    persist_legacy: bool = True,
) -> ScorerResult:
    """Return math scores through the registry's structured scorer contract.

    :func:`compute_scores` remains the compatibility API and performs the
    established JSONL/summary persistence. This entry point builds the registry
    result from the in-memory annotated records, without reading those files
    back from disk.
    """
    # Each inference row carries one generation; normalize its representation
    # before scoring and problem-level aggregation.
    scoring_dataset = _normalize_math_samples(eval_dataset, response_key)
    accuracy = compute_scores(
        eval_dataset=scoring_dataset,
        label_key=label_key,
        response_key=response_key,
        cache_path=cache_path,
        max_workers=max_workers,
        timeout=timeout,
        persist_legacy=persist_legacy,
    )
    observations = [
        float(item.get("accuracy", 0.0))
        for item in scoring_dataset
        if item.get("evaluation_status", "completed") == "completed"
    ]
    timeout_count = sum(
        item.get("evaluation_status") == "timeout" for item in scoring_dataset
    )
    failed_count = sum(
        item.get("evaluation_status") == "failed" for item in scoring_dataset
    )
    skipped_count = sum(
        item.get("evaluation_status") == "skipped" for item in scoring_dataset
    )
    failure_counts: dict[str, int] = Counter()
    for item in scoring_dataset:
        stage = str(item.get("failure_stage", "none"))
        if item.get("evaluation_status") == "failed":
            failure_counts[f"{stage}_failed"] += 1
        elif item.get("evaluation_status") == "timeout":
            failure_counts["verification_failed"] += 1
    failure_counts["wrong_answer"] = sum(
        item.get("evaluation_status", "completed") == "completed"
        and float(item.get("accuracy", 0.0)) != 1.0
        for item in scoring_dataset
    )
    details, extra_metrics, problem_observations = _build_problem_level_metrics(
        scoring_dataset, expected_samples=expected_samples
    )
    complete_problem_count = sum(1 for problem in details if problem["complete"])
    metrics = {"accuracy": accuracy, "sample_accuracy": accuracy, **extra_metrics}
    return ScorerResult(
        metrics=metrics,
        observations={
            "accuracy": observations,
            "sample_accuracy": observations,
            **problem_observations,
        },
        per_item=[dict(item) for item in scoring_dataset],
        details={
            "problem_level": details,
            "complete_problem_count": complete_problem_count,
            "incomplete_problem_count": len(details) - complete_problem_count,
            "excluded_problem_doc_ids": [
                problem["doc_id"] for problem in details if not problem["complete"]
            ],
        },
        sample_count=len(scoring_dataset),
        effective_sample_count=len(observations),
        failed_count=failed_count,
        skipped_count=skipped_count,
        timeout_count=timeout_count,
        failure_counts=failure_counts,
    )


def _normalize_math_samples(
    eval_dataset: list[dict[str, Any]], response_key: str
) -> list[dict[str, Any]]:
    """Validate and normalize one math generation per input row."""
    return expand_single_generation_samples(
        eval_dataset, response_key, problem_identity=_problem_identity
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
            continue
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
    expected_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, list[float]]]:
    """Aggregate sample outcomes into pass@k and majority-vote metrics.

    A problem is complete when its observed row count reaches the requested
    generation count. Problem-level metrics include complete problems only.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row_index, item in enumerate(scored_dataset):
        document_id = item.get("doc_id")
        item_expected = item.get("expected_samples")
        if (document_id is None or not str(document_id).strip()) and (
            (expected_samples is not None and expected_samples > 1)
            or (isinstance(item_expected, int) and item_expected > 1)
        ):
            raise ValueError(
                "Math problem-level metrics require a non-empty 'doc_id' for "
                "multi-sample records"
            )
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
        item_expected = max(
            [
                int(item["expected_samples"])
                for item in items
                if isinstance(item.get("expected_samples"), int)
                and item["expected_samples"] > 0
            ],
            default=0,
        )
        target_count = (
            expected_samples
            if expected_samples and expected_samples > 0
            else item_expected
        )
        target_count = target_count or sample_count
        complete = len(items) == target_count
        problems.append(
            {
                "doc_id": problem_id,
                "correct_samples": correct_samples,
                "sample_count": sample_count,
                "observed_samples": sample_count,
                "expected_samples": target_count,
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
    for k in sorted({problem["expected_samples"] for problem in problems}):
        cohort = [
            problem
            for problem in problems
            if problem["expected_samples"] == k and problem["complete"]
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
    """Return the raw response, filtered response, and task pipeline trace."""
    raw_response = response[0] if isinstance(response, list) and response else response
    filtered, trace = MATH_RESPONSE_PIPELINE.apply_with_trace(raw_response)
    return build_filter_artifacts(
        "" if raw_response is None else str(raw_response), filtered, trace
    )


def save_cache(eval_dataset: list[dict[str, Any]], cache_path: str) -> None:
    """
    Save evaluation results and metadata to a JSONL file.

    Args:
        eval_dataset: Evaluation results to save
        cache_path: Output file path for JSONL data

    Raises:
        IOError: If the result file cannot be written
    """
    try:
        atomic_write_jsonl(cache_path, eval_dataset)
        logger.info(f"✅ Results saved to {cache_path}")
    except OSError as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise
