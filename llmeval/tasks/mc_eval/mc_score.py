"""Multiple-choice answer-token and generation scoring.

Metrics
-------
acc — argmax accuracy for answer-token logprobs, or extracted answer labels

Entry points
------------
score_loglikelihood_result — score MC inference loglikelihood output
score_generate_result      — score MC inference generation output

Pipeline
--------
Both entry points share one pipeline:

    score_items        — serial / process-pool dispatcher (order-preserving)
    process_item       — pool worker, unpacks (index, item, ...) tuples
    score_*_item       — per-item scorers (total functions, never raise)
    extract_answer     — answer-letter extraction for generate mode
    build_result       — aggregate records into MCScoreResult

Only lightweight dependencies (pebble / tqdm) are used — the module stays
independent of the inference environment (no openai / torch).

Per-item record schemas
-----------------------
Loglikelihood:  {"gold": int, "pred": int, "correct": bool}
Generate:       {"gold": str, "pred": str, "correct": bool}
"""

from __future__ import annotations

import math
import re
from collections import Counter
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from typing import Any, Literal

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.postprocess import (
    TextFilterPipeline,
    normalize_single_generation_samples,
    resolve_max_workers,
    resolve_single_generation,
    sample_order_indices,
    strip_reasoning_wrappers,
)
from llmeval.tasks.registry import ScorerResult
from llmeval.utils.log import init_logger

__all__ = [
    "MCScoreResult",
    "extract_answer",
    "merge_generate_records",
    "score_generate_result",
    "score_loglikelihood_item",
    "score_loglikelihood_result",
]

logger = init_logger("mc_score")

_MC_AGGREGATIONS = frozenset({"first", "majority_vote", "any_correct", "per_sample"})

# Precompiled answer-extraction regexes.
_ANSWER_MARKER_RE: re.Pattern[str] = re.compile(
    r"(?:Answer|答案)\s*[:：]\s*([A-J])\s*[.。]?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_LAST_LETTER_RE: re.Pattern[str] = re.compile(r"\b([A-Ja-j])\b")

# Standalone "I" (pronoun) and "a" (article) are English words, not answer
# letters — they must never win the fallback extraction.
_FALLBACK_STOPWORDS = frozenset({"I", "a"})


def _normalize_generate_gold(value: Any) -> str:
    """Normalize a generate-mode label; missing labels remain empty."""
    return "" if value is None else str(value).strip().upper()


def _resolve_generate_gold(gold: str, item: dict[str, Any]) -> str | None:
    """Resolve a generate-mode gold label to a supported answer letter.

    Accepts:
    - a single letter ``A``-``J``,
    - a 1-based option index (``"1"`` → first choice), when ``choices`` is
      present,
    - the exact text of one of the ``choices``.

    Returns ``None`` when the label cannot be resolved to a supported choice
    (data error — the row is marked failed rather than scored wrong).
    """
    choices = item.get("choices")
    if choices is None:
        return gold if len(gold) == 1 and "A" <= gold <= "J" else None
    if not isinstance(choices, list) or not choices:
        return None
    supported_choice_count = min(len(choices), 10)
    if len(gold) == 1 and "A" <= gold <= "J":
        index = ord(gold) - ord("A")
        return gold if index < supported_choice_count else None
    if gold.isdigit():
        index = int(gold) - 1
        if 0 <= index < supported_choice_count:
            return chr(ord("A") + index)
        return None
    for index, choice in enumerate(choices[:supported_choice_count]):
        if str(choice).strip().upper() == gold:
            return chr(ord("A") + index)
    return None


def _mc_problem_identity(item: dict[str, Any], row_index: int) -> str:
    """Return the stable question identity, falling back to the row position."""
    document_id = item.get("doc_id")
    if document_id is None or (
        isinstance(document_id, str) and not document_id.strip()
    ):
        return f"row:{row_index}"
    return f"doc:{document_id}"


@dataclass
class MCScoreResult:
    """Aggregate multiple-choice accuracy.

    Attributes
    ----------
    acc:
        Accuracy via argmax of raw answer-token logprobs (loglikelihood), or via answer-letter
        extraction from generated text (generate mode).
    total:
        Number of items scored.
    correct:
        Number of items answered correctly under *acc*.
    records:
        Internal scoring records used to build aggregate metrics.
    """

    acc: float = 0.0
    total: int = 0
    correct: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)


# ===========================================================================
# Entry points
# ===========================================================================


def score_loglikelihood_result(
    eval_dataset: list[dict[str, Any]],
    max_workers: int = 8,
    timeout: int = 60,
) -> ScorerResult:
    """Score loglikelihood-based MC results and return structured metrics.

    Parameters
    ----------
    eval_dataset:
        Items with ``gold`` and answer-token ``logprobs`` fields.
    max_workers:
        Maximum process-pool workers (capped by dataset size and CPU count).
    timeout:
        Per-item scoring timeout in seconds.

    """
    records = score_items(
        eval_dataset,
        mode="loglikelihood",
        label_key="",
        response_key="",
        max_workers=max_workers,
        timeout=timeout,
    )
    metrics = build_result(records)
    return _to_scorer_result(metrics)


def merge_generate_records(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    n_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Validate sample rows and group them by stable MC question identity.

    Input remains strictly one sample per row. Grouping is an internal scoring
    detail used by question-level aggregation modes. MC additionally validates
    choices, gold indices, and choice tokens because those fields are part of
    the question definition; math and code tasks have no corresponding fields.
    """
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    samples_by_position: list[list[tuple[dict[str, Any], str]]] = []

    for row_index, source in enumerate(eval_dataset):
        item = source.copy()
        document_id = item.get("doc_id")
        identity = _mc_problem_identity(item, row_index)
        position = positions.get(identity)
        if position is None:
            position = len(merged)
            positions[identity] = position
            merged.append(item)
            samples_by_position.append([])
        else:
            target = merged[position]
            for key in (
                label_key,
                "gold",
                "prompt",
                "query",
                "choices",
                "choice_tokens",
            ):
                if key in item and key in target and item[key] != target[key]:
                    raise ValueError(
                        f"Conflicting {key!r} for resumed MC document {document_id!r}"
                    )
                if key in item and key not in target:
                    target[key] = item[key]
        target_samples = samples_by_position[position]
        generation = resolve_single_generation(item, response_key)
        target_samples.append((item, generation or ""))

    for item, samples in zip(merged, samples_by_position, strict=True):
        problem_id = str(item.get("doc_id") or "unknown")
        sample_order = sample_order_indices(
            [sample for sample, _ in samples],
            problem_id=problem_id,
            n_samples=n_samples,
        )
        ordered_samples = [samples[index] for index in sample_order]
        item["sample_group_complete"] = ordered_samples[0][0]["sample_group_complete"]
        item["n_samples"] = ordered_samples[0][0]["n_samples"]
        item["sample_index"] = ordered_samples[0][0]["sample_index"]
        sample_errors = [
            bool(sample.get("error"))
            or resolve_single_generation(sample, response_key) is None
            for sample, _ in ordered_samples
        ]
        item[response_key] = [generation for _, generation in ordered_samples]
        item.pop("error", None)
        if any(sample_errors):
            # Internal scorer metadata. It is consumed by score_generate_item
            # and never included in the returned score record.
            item["_mc_sample_errors"] = sample_errors
    return merged


def score_generate_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    max_workers: int = 8,
    timeout: int = 60,
    aggregation: str = "first",
    n_samples: int | None = None,
) -> ScorerResult:
    """Score generation-based MC results and return structured metrics.

    The ``aggregation`` argument controls how multiple generations are
    evaluated.  ``first`` preserves the historical behavior; ``majority_vote``
    and ``any_correct`` aggregate at question level, while ``per_sample``
    reports accuracy over every generated sample.

    Parameters
    ----------
    eval_dataset:
        Items with a gold-answer field and a generation-list field.
    label_key:
        Name of the gold-answer field (e.g. ``"answer"``).
    response_key:
        Name of the generation-list field (e.g. ``"gen"``).
    max_workers:
        Maximum process-pool workers (capped by dataset size and CPU count).
    timeout:
        Per-item scoring timeout in seconds.
    aggregation:
        Multiple-generation aggregation strategy.

    """
    if aggregation == "per_sample":
        merged_dataset = normalize_single_generation_samples(
            eval_dataset,
            response_key,
            problem_identity=_mc_problem_identity,
            conflict_keys=(
                label_key,
                "gold",
                "prompt",
                "query",
                "choices",
                "choice_tokens",
            ),
            record_kind="MC document",
            n_samples=n_samples,
        )
    else:
        merged_dataset = merge_generate_records(
            eval_dataset,
            label_key,
            response_key,
            n_samples=n_samples,
        )
    records = score_items(
        merged_dataset,
        mode="generate",
        label_key=label_key,
        response_key=response_key,
        max_workers=max_workers,
        timeout=timeout,
        aggregation=aggregation,
    )
    metrics = build_result(records)
    return _to_scorer_result(metrics)


# ===========================================================================
# Parallel driver
# ===========================================================================


def score_items(
    eval_dataset: list[dict[str, Any]],
    mode: Literal["loglikelihood", "generate"],
    label_key: str,
    response_key: str,
    max_workers: int,
    timeout: int,
    aggregation: str = "first",
) -> list[dict[str, Any]]:
    """Score every item, preserving input order.

    When the dataset is small or workers are limited, scoring runs serially
    to avoid pool overhead.  Otherwise a :class:`~pebble.ProcessPool` is used
    so that large benchmarks (e.g. MMLU ~14k items) finish quickly.

    Timed-out or crashed worker tasks are replaced with records carrying an
    explicit status.  They remain in the output for diagnostics, but are not
    treated as ordinary incorrect model answers by the structured scorer.
    """
    total = len(eval_dataset)
    if total == 0:
        return []
    if max_workers <= 1 or total == 1:
        # Serial path — avoids pool startup cost for tiny workloads.
        return [
            process_item((i, item, mode, label_key, response_key, aggregation))[1]
            for i, item in enumerate(eval_dataset)
        ]

    optimal_workers = resolve_max_workers(total, max_workers)
    results_by_index: dict[int, dict[str, Any]] = {}

    with (
        tqdm(total=total, desc="Scoring items", unit="item") as pbar,
        ProcessPool(max_workers=optimal_workers) as pool,
    ):
        iterable = [
            (i, item, mode, label_key, response_key, aggregation)
            for i, item in enumerate(eval_dataset)
        ]
        future = pool.map(process_item, iterable, timeout=timeout)
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError:
                logger.warning("Individual scoring task timed out; marked as failed")
                pbar.update(1)
                continue
            except Exception as exc:
                logger.warning("MC scoring worker result failed: %s", exc)
                pbar.update(1)
                continue

            if result is not None:
                idx, record = result
                results_by_index[idx] = record
            pbar.update(1)

    # Replace missing entries (pool timeouts, worker crashes) with explicit
    # failure records so they never inflate accuracy or uncertainty.
    records: list[dict[str, Any]] = []
    for i, item in enumerate(eval_dataset):
        record = results_by_index.get(i)
        if record is None:
            record = _error_record(item, mode, label_key, response_key, aggregation)
        records.append(record)
    return records


def _error_record(
    item: dict[str, Any],
    mode: Literal["loglikelihood", "generate"],
    label_key: str,
    response_key: str,
    aggregation: str,
) -> dict[str, Any]:
    """Build a failed record for an item that could not be scored."""
    if mode == "loglikelihood":
        try:
            gold: int | str = int(item.get("gold", -1))
        except (TypeError, ValueError):
            gold = -1
        pred: int | str = -1
    else:
        gold = _normalize_generate_gold(item.get(label_key))
        pred = ""
    record = {
        **{
            key: item[key]
            for key in (
                "doc_id",
                "sample_index",
                "sample_group_complete",
                "n_samples",
            )
            if key in item and item[key] is not None
        },
        "gold": gold,
        "pred": pred,
        "correct": False,
        "evaluation_status": "failed",
        "aggregation": aggregation,
        "scoring_mode": item.get(
            "scoring_mode",
            "unknown_legacy" if mode == "loglikelihood" else aggregation,
        ),
    }
    if mode == "generate":
        # Keep the generation count visible so per_sample weighting does not
        # drop the item from every count when its worker was lost.
        response = item.get(response_key)
        record["sample_total"] = (
            max(len(response), 1) if isinstance(response, list) else 1
        )
    return record


def process_item(
    args: tuple[
        int,
        dict[str, Any],
        Literal["loglikelihood", "generate"],
        str,
        str,
        str,
    ],
) -> tuple[int, dict[str, Any]]:
    """Pool-worker entry point — **must** be module-level for pickling.

    Takes an ``(index, item, mode, label_key, response_key, aggregation)`` tuple and returns
    ``(original_index, scored_record)`` so results can be re-ordered after
    parallel execution. Worker errors are returned as ``failed`` records so a
    missing result in the collector unambiguously means a pool-level timeout.
    """
    idx, item, mode, label_key, response_key, aggregation = args
    try:
        if mode == "loglikelihood":
            record = score_loglikelihood_item(item)
        else:
            record = score_generate_item(item, label_key, response_key, aggregation)
        return idx, _attach_item_metadata(record, item, mode, aggregation)
    except Exception as exc:
        logger.warning("MC scoring worker failed for item %d: %s", idx, exc)
        return idx, _error_record(item, mode, label_key, response_key, aggregation)


def _attach_item_metadata(
    record: dict[str, Any],
    item: dict[str, Any],
    mode: Literal["loglikelihood", "generate"],
    aggregation: str,
) -> dict[str, Any]:
    """Carry stable identity and scoring mode into the persisted score row."""
    metadata = {
        key: item[key]
        for key in ("doc_id", "sample_index", "sample_group_complete", "n_samples")
        if key in item and item[key] is not None
    }
    metadata["scoring_mode"] = item.get(
        "scoring_mode",
        "unknown_legacy" if mode == "loglikelihood" else aggregation,
    )
    return {**metadata, **record}


def _sample_weight(record: dict[str, Any]) -> int:
    """Number of scored observations a record represents.

    ``per_sample`` aggregation scores every generation independently, so the
    record counts ``sample_total`` observations; other aggregations collapse
    to one scored observation per record.
    """
    if record.get("aggregation") == "per_sample":
        sample_total = record.get("sample_total")
        # Pool-level timeout/crash records represent the input item whose
        # generations could not be scored and therefore still count once.
        return 1 if sample_total is None else max(int(sample_total), 0)
    return 1


def build_result(records: list[dict[str, Any]]) -> MCScoreResult:
    """Aggregate per-item records into :class:`MCScoreResult`."""
    per_sample = bool(records) and all(
        r.get("aggregation") == "per_sample" for r in records
    )

    eligible_records = [
        record
        for record in records
        if record.get("evaluation_status", "completed") == "completed"
        and _record_is_metric_eligible(record)
    ]
    if per_sample:
        total = sum(int(r.get("sample_total", 0)) for r in records)
        n_correct = sum(int(r.get("sample_correct_count", 0)) for r in eligible_records)
    else:
        total = len(records)
        n_correct = sum(1 for r in eligible_records if r["correct"])

    effective_total = sum(_sample_weight(record) for record in eligible_records)
    safe_total = max(effective_total, 1)
    return MCScoreResult(
        acc=n_correct / safe_total,
        total=total,
        correct=n_correct,
        records=records,
    )


def _to_scorer_result(result: MCScoreResult) -> ScorerResult:
    """Convert MC task output to the registry's scorer contract."""
    observations: dict[str, list[float]] = {"acc": []}

    for record in result.records:
        if record.get("evaluation_status", "completed") != "completed":
            continue
        if not _record_is_metric_eligible(record):
            continue
        if record.get("aggregation") == "per_sample" and isinstance(
            record.get("sample_correct"), list
        ):
            observations["acc"].extend(
                float(bool(value)) for value in record["sample_correct"]
            )
            continue
        observations["acc"].append(float(bool(record.get("correct", False))))

    sample_count = sum(_sample_weight(record) for record in result.records)
    failed_count = sum(
        _sample_weight(record)
        for record in result.records
        if record.get("evaluation_status") == "failed"
    )
    excluded_count = sum(
        _sample_weight(record)
        for record in result.records
        if record.get("evaluation_status", "completed") == "completed"
        and not _record_is_metric_eligible(record)
    )
    problem_completeness: dict[str, bool] = {}
    for index, record in enumerate(result.records):
        problem_id = str(record.get("doc_id", f"row:{index}"))
        problem_completeness[problem_id] = problem_completeness.get(
            problem_id, True
        ) and bool(record.get("sample_group_complete", True))
    incomplete_problem_ids = [
        problem_id
        for problem_id, complete in problem_completeness.items()
        if not complete
    ]
    return ScorerResult(
        metrics={"acc": result.acc},
        observations=observations,
        records=result.records,
        details={
            "complete_problem_count": len(problem_completeness)
            - len(incomplete_problem_ids),
            "incomplete_problem_count": len(incomplete_problem_ids),
            "incomplete_problem_doc_ids": incomplete_problem_ids,
        },
        sample_count=sample_count,
        effective_sample_count=sample_count - failed_count - excluded_count,
        failed_count=failed_count,
        excluded_count=excluded_count,
    )


def _record_is_metric_eligible(record: dict[str, Any]) -> bool:
    """Exclude incomplete groups only from aggregations that require all samples."""
    aggregation = record.get("aggregation")
    if aggregation == "first":
        return record.get("sample_index", 0) == 0
    return aggregation not in {"majority_vote", "any_correct"} or bool(
        record.get("sample_group_complete", True)
    )


# ===========================================================================
# Per-item scorers
# ===========================================================================


def score_loglikelihood_item(item: dict[str, Any]) -> dict[str, Any]:
    """Score a single loglikelihood item.

    Unscorable items (missing / empty logprobs, invalid gold index, or every
    logprob set to ``-inf`` after a failed API call) receive ``pred = -1``.
    Invalid data and inference failures are represented uniformly as failed.
    """
    raw_gold = item.get("gold", -1)
    if type(raw_gold) is int:
        gold = raw_gold
    else:
        logger.warning(
            f"Invalid gold index {item.get('gold')!r} — treating as -1 (always wrong)."
        )
        gold = -1
    raw_logprobs = item.get("logprobs", [])
    logprobs: list[float] = []
    valid_logprobs = isinstance(raw_logprobs, list)
    if isinstance(raw_logprobs, list):
        try:
            for value in raw_logprobs:
                if value is None:
                    logprobs.append(float("-inf"))
                elif isinstance(value, int | float) and not isinstance(value, bool):
                    score = float(value)
                    if math.isnan(score) or score == float("inf"):
                        raise ValueError("logprob must be finite or -inf")
                    logprobs.append(score)
                else:
                    raise TypeError("non-numeric logprob")
        except (OverflowError, TypeError, ValueError):
            logger.warning("Invalid aggregate logprobs; marking item failed")
            logprobs = []
            valid_logprobs = False
    choice_fields = [
        item[name]
        for name in ("choice_tokens", "choices", "choice_texts")
        if name in item
    ]
    valid_choice_fields = not choice_fields or all(
        isinstance(choices, list) and bool(choices) and len(choices) == len(logprobs)
        for choices in choice_fields
    )
    # Invalid gold, malformed scores, and missing inference output cannot be
    # scored and therefore share the same failed status.
    invalid_shape = valid_logprobs and bool(logprobs) and not valid_choice_fields
    invalid_gold = gold < 0 or (bool(logprobs) and gold >= len(logprobs))
    if (
        not logprobs
        or invalid_shape
        or invalid_gold
        or all(lp == float("-inf") for lp in logprobs)
    ):
        return {
            "gold": gold,
            "pred": -1,
            "correct": False,
            "evaluation_status": "failed",
        }

    # acc — argmax over raw logprobs (ties broken by smallest index).
    pred = max(range(len(logprobs)), key=logprobs.__getitem__)
    is_correct = pred == gold

    return {
        "gold": gold,
        "pred": pred,
        "correct": is_correct,
    }


def score_generate_item(
    item: dict[str, Any],
    label_key: str,
    response_key: str,
    aggregation: str = "first",
) -> dict[str, Any]:
    """Score a single generate-mode item by extracting the answer letter.

    Edge cases handled defensively: empty gold or unparseable generation is
    always counted wrong (otherwise ``"" == ""`` would inflate accuracy);
    bare-string ``gen`` fields are tolerated (the schema expects a list).
    """
    if aggregation not in _MC_AGGREGATIONS:
        raise ValueError(f"Unsupported MC aggregation: {aggregation}")

    gold = _resolve_generate_gold(_normalize_generate_gold(item.get(label_key)), item)
    # A missing response key is a structural schema failure, not a completed
    # empty answer: only an explicit empty list/string is one empty answer.
    if response_key not in item:
        response_valid = False
        generations: Any = None
    else:
        generations = item.get(response_key)
        response_valid = isinstance(generations, str) or (
            isinstance(generations, list)
            and (
                not generations or all(isinstance(value, str) for value in generations)
            )
        )

    if isinstance(generations, str):
        generation_texts = [generations]
    elif isinstance(generations, list):
        generation_texts = generations if response_valid and generations else [""]
    else:
        generation_texts = []

    raw_sample_errors = item.get("_mc_sample_errors")
    if isinstance(raw_sample_errors, list):
        sample_errors = [bool(error) for error in raw_sample_errors]
    else:
        sample_errors = [bool(item.get("error"))] * max(len(generation_texts), 1)
    relevant_errors = sample_errors[:1] if aggregation == "first" else sample_errors

    status = (
        "completed"
        if gold is not None and response_valid and not any(relevant_errors)
        else "failed"
    )
    gold = gold or ""

    filtered = [
        MC_GENERATION_PIPELINE.apply_with_trace(text) for text in generation_texts
    ]
    predictions = [prediction for prediction, _ in filtered]
    sample_correct = [bool(gold) and prediction == gold for prediction in predictions]

    if aggregation == "majority_vote":
        counts = Counter(predictions)
        if counts:
            max_count = max(counts.values())
            pred = next(
                prediction
                for prediction in predictions
                if counts[prediction] == max_count
            )
        else:
            pred = ""
        is_correct = bool(gold) and pred == gold
    elif aggregation == "any_correct":
        is_correct = any(sample_correct)
        pred = (
            gold
            if is_correct
            else next((prediction for prediction in predictions if prediction), "")
        )
    else:
        pred = predictions[0] if predictions else ""
        is_correct = bool(gold) and bool(pred) and pred == gold

    return {
        "gold": gold,
        "pred": pred,
        "correct": is_correct,
        "aggregation": aggregation,
        "predictions": predictions,
        "raw_gen": generation_texts,
        "filter_trace": [trace for _, trace in filtered],
        "evaluation_status": status,
        "sample_correct": sample_correct,
        "sample_total": max(len(predictions), 1)
        if status == "failed"
        else len(predictions),
        "sample_correct_count": sum(sample_correct),
    }


def extract_answer(text: str) -> str:
    """Extract an answer letter (``A``-``J``) from a model-generated string.

    Strategy (aligned with lm-evaluation-harness conventions)
    ----------------------------------------------------------
    1. **Marker match** — look for an explicit ``Answer: X`` or ``答案: X``
       token at the **end** of a line (``re.MULTILINE``, ``$`` anchor).
       Returns the **last** such match so an explicit correction wins.
    2. **Fallback** — find the **last** standalone letter ``A``-``J``
       (word-boundary delimited) anywhere in the text.  The English pronoun
       ``I`` and article ``a`` are ignored so trailing prose cannot hijack
       the extraction.

    Both paths normalise to uppercase.  Returns ``""`` when no letter can
    be extracted.
    """
    # 1. Explicit "Answer: X" / "答案：X" marker.
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper()

    # 2. Last standalone A-J letter in the text (excluding "I"/"a" stopwords).
    letters = [
        letter
        for letter in _LAST_LETTER_RE.findall(text)
        if letter not in _FALLBACK_STOPWORDS
    ]
    if letters:
        return letters[-1].upper()

    return ""


MC_GENERATION_PIPELINE = TextFilterPipeline(
    "mc_generation",
    (("strip_reasoning", strip_reasoning_wrappers), ("extract_answer", extract_answer)),
)
