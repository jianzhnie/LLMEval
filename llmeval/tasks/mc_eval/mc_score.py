"""Multiple-choice scoring: answer-token loglikelihood and generation-based evaluation.

Aligned with lm-evaluation-harness metric definitions.

Metrics
-------
acc         — accuracy (argmax of raw answer-token logprobs, or letter extraction in generate mode)
acc_norm    — accuracy with Unicode-character-normalized choice logprobs
acc_bytes   — accuracy with UTF-8-byte-normalized choice logprobs
exact_match — alias for acc in the MC context

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
Loglikelihood:  {"gold": int, "pred": int, "correct": bool, "correct_norm": bool}
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
    FilterRegistry,
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

MC_FILTER_REGISTRY = FilterRegistry()
MC_FILTER_REGISTRY.register("strip_reasoning", strip_reasoning_wrappers)
MC_GENERATION_PIPELINE: TextFilterPipeline

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
    """Aggregate MC evaluation metrics (lm-eval aligned).

    Attributes
    ----------
    acc:
        Accuracy via argmax of raw answer-token logprobs (loglikelihood), or via answer-letter
        extraction from generated text (generate mode).
    acc_norm:
        Accuracy via Unicode-character-length-normalized logprobs. In generate
        mode this always equals *acc* because no likelihood is available.
    acc_bytes:
        Accuracy via UTF-8-byte-length-normalized logprobs. In generate mode
        this always equals *acc*.
    exact_match:
        Convenience alias for *acc* in the multiple-choice context.
    total:
        Number of items scored.
    correct:
        Number of items answered correctly under *acc*.
    correct_norm:
        Number of items answered correctly under *acc_norm*.
    correct_bytes:
        Number of items answered correctly under *acc_bytes*.
    records:
        Internal scoring records used to build aggregate metrics.
    """

    acc: float = 0.0
    acc_norm: float = 0.0
    acc_bytes: float = 0.0
    exact_match: float = 0.0
    total: int = 0
    correct: int = 0
    correct_norm: int = 0
    correct_bytes: int = 0
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
        Items with ``gold`` / ``logprobs`` / ``choices`` fields (output of the
        MC inference loglikelihood mode). Newer inference outputs also include
        ``choice_tokens`` to record the actual answer tokens scored.
    max_workers:
        Maximum process-pool workers (capped by dataset size and CPU count).
    timeout:
        Per-item scoring timeout in seconds.

    """
    if any(
        not (
            item.get("choice_tokens") or item.get("choices") or item.get("choice_texts")
        )
        for item in eval_dataset
    ):
        logger.warning(
            "Some items have no 'choice_tokens'/'choices' field — acc_norm will fall back to acc."
        )

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
    eval_dataset: list[dict[str, Any]], label_key: str, response_key: str
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
        problem_id = str(document_id) if identity.startswith("doc:") else identity
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
        generation = resolve_single_generation(
            item, response_key, problem_id=problem_id
        )
        target_samples.append((item, generation or ""))

    for item, samples in zip(merged, samples_by_position, strict=True):
        problem_id = str(item.get("doc_id") or "unknown")
        sample_order = sample_order_indices(
            [sample for sample, _ in samples], problem_id=problem_id
        )
        ordered_samples = [samples[index] for index in sample_order]
        sample_errors = [
            bool(sample.get("error"))
            or response_key not in sample
            or sample.get(response_key) is None
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
    if aggregation not in _MC_AGGREGATIONS:
        raise ValueError(
            f"Unsupported MC aggregation {aggregation!r}; "
            f"expected one of {sorted(_MC_AGGREGATIONS)}"
        )

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
        )
    else:
        merged_dataset = merge_generate_records(eval_dataset, label_key, response_key)
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
            for key in ("doc_id",)
            if key in item and item[key] is not None
        },
        "gold": gold,
        "pred": pred,
        "correct": False,
        "correct_norm": False,
        "correct_bytes": False,
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
        key: item[key] for key in ("doc_id",) if key in item and item[key] is not None
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
    """Aggregate per-item records into :class:`MCScoreResult`.

    ``correct_norm`` is only present in loglikelihood records; generate-mode
    records fall back to ``correct`` so that ``acc_norm == acc``.
    """
    per_sample = bool(records) and all(
        r.get("aggregation") == "per_sample" for r in records
    )

    eligible_records = [
        record
        for record in records
        if record.get("evaluation_status", "completed") == "completed"
    ]
    if per_sample:
        total = sum(int(r.get("sample_total", 0)) for r in records)
        n_correct = sum(int(r.get("sample_correct_count", 0)) for r in eligible_records)
        n_correct_norm = sum(
            int(r.get("sample_correct_norm_count", 0)) for r in eligible_records
        )
        n_correct_bytes = sum(
            int(r.get("sample_correct_bytes_count", 0)) for r in eligible_records
        )
    else:
        total = len(records)
        n_correct = sum(1 for r in eligible_records if r["correct"])
        n_correct_norm = sum(
            1 for r in eligible_records if r.get("correct_norm", r["correct"])
        )
        n_correct_bytes = sum(
            1 for r in eligible_records if r.get("correct_bytes", r["correct"])
        )

    effective_total = sum(_sample_weight(record) for record in eligible_records)
    safe_total = max(effective_total, 1)
    return MCScoreResult(
        acc=n_correct / safe_total,
        acc_norm=n_correct_norm / safe_total,
        acc_bytes=n_correct_bytes / safe_total,
        exact_match=n_correct / safe_total,
        total=total,
        correct=n_correct,
        correct_norm=n_correct_norm,
        correct_bytes=n_correct_bytes,
        records=records,
    )


def _to_scorer_result(result: MCScoreResult) -> ScorerResult:
    """Convert MC task output to the registry's scorer contract."""
    observations: dict[str, list[float]] = {
        name: [] for name in ("acc", "acc_norm", "acc_bytes", "exact_match")
    }

    for record in result.records:
        if record.get("evaluation_status", "completed") != "completed":
            continue
        if record.get("aggregation") == "per_sample" and isinstance(
            record.get("sample_correct"), list
        ):
            samples = [float(bool(value)) for value in record["sample_correct"]]
            for values in observations.values():
                values.extend(samples)
            continue
        for name, key in (
            ("acc", "correct"),
            ("acc_norm", "correct_norm"),
            ("acc_bytes", "correct_bytes"),
            ("exact_match", "correct"),
        ):
            observations[name].append(float(bool(record.get(key, False))))

    sample_count = sum(_sample_weight(record) for record in result.records)
    failed_count = sum(
        _sample_weight(record)
        for record in result.records
        if record.get("evaluation_status") == "failed"
    )
    return ScorerResult(
        metrics={
            "acc": result.acc,
            "acc_norm": result.acc_norm,
            "acc_bytes": result.acc_bytes,
            "exact_match": result.exact_match,
        },
        observations=observations,
        records=result.records,
        sample_count=sample_count,
        effective_sample_count=sample_count - failed_count,
        failed_count=failed_count,
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
    try:
        if isinstance(raw_gold, bool):
            raise TypeError("boolean gold index")
        if isinstance(raw_gold, float) and not raw_gold.is_integer():
            raise ValueError("non-integral gold index")
        gold = int(raw_gold)
    except (TypeError, ValueError):
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
        except (TypeError, ValueError):
            logger.warning("Invalid aggregate logprobs; marking item failed")
            logprobs = []
            valid_logprobs = False
    choices = (
        item.get("choice_tokens") or item.get("choices") or item.get("choice_texts", [])
    )
    expected_choices = len(choices) if isinstance(choices, list) and choices else None
    # Invalid gold, malformed scores, and missing inference output cannot be
    # scored and therefore share the same failed status.
    invalid_shape = (
        valid_logprobs
        and bool(logprobs)
        and expected_choices is not None
        and expected_choices != len(logprobs)
    )
    invalid_gold = gold < 0 or (
        expected_choices is not None and gold >= expected_choices
    )
    invalid_gold = invalid_gold or (
        expected_choices is None and bool(logprobs) and gold >= len(logprobs)
    )
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
            "correct_norm": False,
            "correct_bytes": False,
            "evaluation_status": "failed",
        }

    # acc — argmax over raw logprobs (ties broken by smallest index).
    pred = max(range(len(logprobs)), key=logprobs.__getitem__)
    is_correct = pred == gold

    # lm-eval uses Unicode character length for acc_norm and UTF-8 byte length
    # for acc_bytes. Token counts are useful diagnostics but are not the
    # denominator of either harness metric.
    if choices and len(choices) == len(logprobs):
        choice_lens = [max(len(str(choice)), 1) for choice in choices]
        is_correct_norm = argmax_normalized(logprobs, choice_lens) == gold
    else:
        is_correct_norm = is_correct

    if choices and len(choices) == len(logprobs):
        choice_bytes = [max(len(str(choice).encode("utf-8")), 1) for choice in choices]
        is_correct_bytes = argmax_normalized(logprobs, choice_bytes) == gold
    else:
        is_correct_bytes = is_correct

    return {
        "gold": gold,
        "pred": pred,
        "correct": is_correct,
        "correct_norm": is_correct_norm,
        "correct_bytes": is_correct_bytes,
    }


def argmax_normalized(logprobs: list[float], lengths: list[int | float]) -> int:
    """Return the first argmax after dividing scores by positive lengths."""
    normalized = [
        score / float(length) for score, length in zip(logprobs, lengths, strict=True)
    ]
    return max(range(len(normalized)), key=normalized.__getitem__)


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
        generations: Any = []
    else:
        generations: Any = item.get(response_key)
        response_valid = isinstance(generations, str | list)

    if isinstance(generations, str):
        generation_texts = [generations]
    elif isinstance(generations, list):
        generation_texts = (
            [str(generation) for generation in generations] if generations else [""]
        )
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
        "correct_norm": is_correct,
        "correct_bytes": is_correct,
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
        "sample_correct_norm_count": sum(sample_correct),
        "sample_correct_bytes_count": sum(sample_correct),
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
    matches = [
        letter
        for letter in _LAST_LETTER_RE.findall(text)
        if letter not in _FALLBACK_STOPWORDS
    ]
    if matches:
        return matches[-1].upper()

    return ""


MC_FILTER_REGISTRY.register("extract_answer", extract_answer)
MC_GENERATION_PIPELINE = MC_FILTER_REGISTRY.build_pipeline(
    "mc_generation", "strip_reasoning", "extract_answer"
)
