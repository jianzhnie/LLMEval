"""Multiple-choice scoring: answer-token loglikelihood and generation-based evaluation.

Aligned with lm-evaluation-harness metric definitions.

Metrics
-------
acc         — accuracy (argmax of raw answer-token logprobs, or letter extraction in generate mode)
acc_norm    — accuracy with Unicode-character-normalized continuation logprobs
acc_bytes   — accuracy with UTF-8-byte-normalized continuation logprobs
exact_match — alias for acc in the MC context

Entry points
------------
score_loglikelihood  — score results produced by MC inference loglikelihood mode
score_generate       — score results produced by MC inference generate mode

Pipeline
--------
Both entry points share one pipeline:

    score_items        — serial / process-pool dispatcher (order-preserving)
    process_item       — pool worker, unpacks (index, item, ...) tuples
    score_*_item       — per-item scorers (total functions, never raise)
    extract_answer     — answer-letter extraction for generate mode
    build_result       — aggregate records into MCScoreResult
    write_cache        — persist per-item JSONL + summary JSON

Only lightweight dependencies (pebble / tqdm) are used — the module stays
independent of the inference environment (no openai / torch).

Per-item record schemas
-----------------------
Loglikelihood:  {"gold": int, "pred": int, "correct": bool, "correct_norm": bool}
Generate:       {"gold": str, "pred": str, "correct": bool}
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.persistence import persist_results
from llmeval.tasks.postprocess import (
    FilterRegistry,
    TextFilterPipeline,
    build_filter_artifacts,
    resolve_max_workers,
    resolve_single_generation,
    strip_reasoning_wrappers,
)
from llmeval.tasks.registry import ScorerResult
from llmeval.utils.log import init_logger

__all__ = [
    "MCScoreResult",
    "extract_answer",
    "merge_generate_records",
    "score_generate",
    "score_generate_result",
    "score_loglikelihood",
    "score_loglikelihood_item",
    "score_loglikelihood_result",
]

logger = init_logger("mc_score")

_MC_AGGREGATIONS = frozenset({"first", "majority_vote", "any_correct", "per_sample"})

MC_FILTER_REGISTRY = FilterRegistry()
MC_FILTER_REGISTRY.register("strip_reasoning", strip_reasoning_wrappers, version="1")
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
    """Normalize a generate-mode label; missing labels remain empty/skipped."""
    return "" if value is None else str(value).strip().upper()


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
    per_item:
        Per-item scoring records written to the JSONL result file.
    """

    acc: float = 0.0
    acc_norm: float = 0.0
    acc_bytes: float = 0.0
    exact_match: float = 0.0
    total: int = 0
    correct: int = 0
    correct_norm: int = 0
    correct_bytes: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


# ===========================================================================
# Entry points
# ===========================================================================


def score_loglikelihood_result(
    eval_dataset: list[dict[str, Any]],
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
    persist_legacy: bool = True,
) -> ScorerResult:
    """Score loglikelihood-based MC results and return structured metrics.

    Parameters
    ----------
    eval_dataset:
        Items with ``gold`` / ``logprobs`` / ``choices`` fields (output of the
        MC inference loglikelihood mode). Newer inference outputs also include
        ``choice_tokens`` to record the actual answer tokens scored.
    cache_path:
        Path for the per-item JSONL result file. A ``<stem>.summary.json`` metrics
        file is written alongside it.
    max_workers:
        Maximum process-pool workers (capped by dataset size and CPU count).
    timeout:
        Per-item scoring timeout in seconds.

    The legacy JSONL/summary artifacts are still written for CLI compatibility,
    but registry adapters consume the returned object directly.
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
    if persist_legacy:
        write_cache(metrics, cache_path)
    return _to_scorer_result(metrics)


def score_loglikelihood(
    eval_dataset: list[dict[str, Any]],
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
) -> float:
    """Compatibility wrapper returning only the primary ``acc`` metric."""
    return score_loglikelihood_result(
        eval_dataset,
        cache_path,
        max_workers=max_workers,
        timeout=timeout,
    ).metrics["acc"]


def merge_generate_records(
    eval_dataset: list[dict[str, Any]], label_key: str, response_key: str
) -> list[dict[str, Any]]:
    """Validate sample rows and group them by stable MC question identity.

    Input remains strictly one sample per row. Grouping is an internal scoring
    detail used by question-level aggregation modes.
    """
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    samples_by_position: list[list[str]] = []

    for row_index, source in enumerate(eval_dataset):
        item = source.copy()
        document_id = item.get("doc_id")
        if document_id is None or (
            isinstance(document_id, str) and not document_id.strip()
        ):
            identity = f"row:{row_index}"
            problem_id = identity
        else:
            identity = f"doc:{document_id}"
            problem_id = str(document_id)
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
        target_samples.append(generation or "")

    for item, samples in zip(merged, samples_by_position, strict=True):
        item[response_key] = samples
    return merged


def score_generate_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
    aggregation: str = "first",
    persist_legacy: bool = True,
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
    cache_path:
        Path for the per-item JSONL result file. A ``<stem>.summary.json`` metrics
        file is written alongside it.
    max_workers:
        Maximum process-pool workers (capped by dataset size and CPU count).
    timeout:
        Per-item scoring timeout in seconds.
    aggregation:
        Multiple-generation aggregation strategy.

    The legacy JSONL/summary artifacts are still written for CLI compatibility,
    but registry adapters consume the returned object directly.
    """
    if aggregation not in _MC_AGGREGATIONS:
        raise ValueError(
            f"Unsupported MC aggregation {aggregation!r}; "
            f"expected one of {sorted(_MC_AGGREGATIONS)}"
        )

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
    if persist_legacy:
        write_cache(metrics, cache_path)
    return _to_scorer_result(metrics)


def score_generate(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
    aggregation: str = "first",
) -> float:
    """Compatibility wrapper returning only the primary ``acc`` metric."""
    return score_generate_result(
        eval_dataset,
        label_key,
        response_key,
        cache_path,
        max_workers=max_workers,
        timeout=timeout,
        aggregation=aggregation,
    ).metrics["acc"]


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
                logger.warning("Individual scoring task timed out — marked as timeout.")
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
            record = _error_record(item, mode, label_key, aggregation, "timeout")
        records.append(record)
    return records


def _error_record(
    item: dict[str, Any],
    mode: Literal["loglikelihood", "generate"],
    label_key: str,
    aggregation: str,
    status: str,
) -> dict[str, Any]:
    """Build an explicit non-scored record for timed-out or crashed items."""
    if mode == "loglikelihood":
        try:
            gold: int | str = int(item.get("gold", -1))
        except (TypeError, ValueError):
            gold = -1
        pred: int | str = -1
    else:
        gold = _normalize_generate_gold(item.get(label_key))
        pred = ""
    return {
        **{
            key: item[key]
            for key in ("doc_id", "sample_total")
            if key in item and item[key] is not None
        },
        "gold": gold,
        "pred": pred,
        "correct": False,
        "correct_norm": False,
        "correct_bytes": False,
        "evaluation_status": status,
        "aggregation": aggregation,
        "scoring_mode": item.get(
            "scoring_mode",
            "unknown_legacy" if mode == "loglikelihood" else aggregation,
        ),
    }


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
        return idx, _error_record(item, mode, label_key, aggregation, "failed")


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


def _count_excluded(
    records: list[dict[str, Any]], weight: Callable[[dict[str, Any]], int]
) -> dict[str, int]:
    """Tally records by non-completed status using the given per-record weight."""
    return {
        status: sum(
            weight(record)
            for record in records
            if record.get("evaluation_status") == status
        )
        for status in ("failed", "skipped", "timeout")
    }


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
        per_item=records,
    )


def _to_scorer_result(result: MCScoreResult) -> ScorerResult:
    """Convert MC task output to the registry's scorer contract."""
    observations: dict[str, list[float]] = {
        name: [] for name in ("acc", "acc_norm", "acc_bytes", "exact_match")
    }

    for record in result.per_item:
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

    sample_count = sum(_sample_weight(record) for record in result.per_item)
    excluded = _count_excluded(result.per_item, _sample_weight)
    return ScorerResult(
        metrics={
            "acc": result.acc,
            "acc_norm": result.acc_norm,
            "acc_bytes": result.acc_bytes,
            "exact_match": result.exact_match,
        },
        observations=observations,
        per_item=result.per_item,
        sample_count=sample_count,
        effective_sample_count=max(
            sample_count
            - excluded["failed"]
            - excluded["skipped"]
            - excluded["timeout"],
            0,
        ),
        failed_count=excluded["failed"],
        skipped_count=excluded["skipped"],
        timeout_count=excluded["timeout"],
    )


# ===========================================================================
# Per-item scorers
# ===========================================================================


def score_loglikelihood_item(item: dict[str, Any]) -> dict[str, Any]:
    """Score a single loglikelihood item.

    Unscorable items (missing / empty logprobs, invalid gold index, or every
    logprob set to ``-inf`` after a failed API call) are marked as incorrect
    with ``pred = -1`` so they never contribute false positives.
    """
    try:
        gold = int(item.get("gold", -1))
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid gold index {item.get('gold')!r} — treating as -1 (always wrong)."
        )
        gold = -1
    logprobs: list[float] = item.get("logprobs", [])
    choices = (
        item.get("choice_tokens") or item.get("choices") or item.get("choice_texts", [])
    )
    choice_logprobs = item.get("choice_logprobs")
    expected_choices = len(choices) if isinstance(choices, list) and choices else None
    if (
        isinstance(choice_logprobs, list)
        and choice_logprobs
        and (expected_choices is None or len(choice_logprobs) == expected_choices)
    ):
        # Complete continuation scores are preferred when inference recorded
        # them. Empty per-choice lists remain -inf and cannot become a false
        # positive through argmax.
        try:
            logprobs = [
                sum(float(score) for score in scores) if scores else float("-inf")
                for scores in choice_logprobs
            ]
        except (TypeError, ValueError):
            logger.warning("Invalid choice_logprobs; using aggregate logprobs")
            choice_logprobs = None
    # Guard: unscorable data → forced-incorrect record.  An out-of-range gold
    # index is a dataset problem (skipped); empty/all--inf logprobs are an
    # inference failure (failed).
    invalid_gold = gold < 0 or (bool(logprobs) and gold >= len(logprobs))
    if not logprobs or invalid_gold or all(lp == float("-inf") for lp in logprobs):
        return {
            "gold": gold,
            "pred": -1,
            "correct": False,
            "correct_norm": False,
            "correct_bytes": False,
            "evaluation_status": "skipped" if invalid_gold else "failed",
        }

    # acc — argmax over raw logprobs (ties broken by smallest index).
    pred = max(range(len(logprobs)), key=logprobs.__getitem__)
    is_correct = pred == gold

    # lm-eval uses Unicode character length for acc_norm and UTF-8 byte length
    # for acc_bytes. Token counts are useful diagnostics but are not the
    # denominator of either harness metric.
    char_counts = item.get("choice_char_count")
    byte_counts = item.get("choice_byte_count")
    char_count_values = char_counts if isinstance(char_counts, list) else []
    byte_count_values = byte_counts if isinstance(byte_counts, list) else []
    has_char_counts = len(char_count_values) == len(logprobs) and all(
        isinstance(count, int | float) and count > 0 for count in char_count_values
    )
    has_byte_counts = len(byte_count_values) == len(logprobs) and all(
        isinstance(count, int | float) and count > 0 for count in byte_count_values
    )
    if has_char_counts:
        is_correct_norm = argmax_normalized(logprobs, char_count_values) == gold
    elif choices and len(choices) == len(logprobs):
        choice_lens = [max(len(str(choice)), 1) for choice in choices]
        is_correct_norm = argmax_normalized(logprobs, choice_lens) == gold
    else:
        is_correct_norm = is_correct

    if has_byte_counts:
        is_correct_bytes = argmax_normalized(logprobs, byte_count_values) == gold
    elif choices and len(choices) == len(logprobs):
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

    gold = _normalize_generate_gold(item.get(label_key))
    generations: Any = item.get(response_key, [])

    if isinstance(generations, str):
        generation_texts = [generations]
    elif isinstance(generations, list):
        generation_texts = [str(generation) for generation in generations]
    else:
        generation_texts = []

    if not gold:
        status = "skipped"
    elif not generation_texts or all(not text.strip() for text in generation_texts):
        status = "failed"
    else:
        status = "completed"

    filtered = [
        MC_GENERATION_PIPELINE.apply_with_trace(text) for text in generation_texts
    ]
    predictions = [prediction for prediction, _ in filtered]
    sample_correct = [bool(gold) and prediction == gold for prediction in predictions]

    if aggregation == "majority_vote":
        non_empty = [prediction for prediction in predictions if prediction]
        counts = Counter(non_empty)
        if counts:
            max_count = max(counts.values())
            pred = next(
                prediction
                for prediction in non_empty
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
        **build_filter_artifacts(
            generation_texts,
            predictions,
            [trace for _, trace in filtered],
        ),
        "evaluation_status": status,
        "sample_correct": sample_correct,
        "sample_total": len(predictions),
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
       Returns the **first** such match.
    2. **Fallback** — find the **last** standalone letter ``A``-``J``
       (word-boundary delimited) anywhere in the text.  The English pronoun
       ``I`` and article ``a`` are ignored so trailing prose cannot hijack
       the extraction.

    Both paths normalise to uppercase.  Returns ``""`` when no letter can
    be extracted.
    """
    # 1. Explicit "Answer: X" / "答案：X" marker.
    m = _ANSWER_MARKER_RE.search(text)
    if m:
        return m.group(1).upper()

    # 2. Last standalone A-J letter in the text (excluding "I"/"a" stopwords).
    matches = [
        letter
        for letter in _LAST_LETTER_RE.findall(text)
        if letter not in _FALLBACK_STOPWORDS
    ]
    if matches:
        return matches[-1].upper()

    return ""


MC_FILTER_REGISTRY.register("extract_answer", extract_answer, version="1")
MC_GENERATION_PIPELINE = MC_FILTER_REGISTRY.build_pipeline(
    "mc_generation", "1", "strip_reasoning", "extract_answer"
)


# ===========================================================================
# Result persistence
# ===========================================================================


def write_cache(result: MCScoreResult, cache_path: str | Path) -> None:
    """Persist per-item records (JSONL) and an aggregated metrics summary (JSON).

    The summary is written next to the JSONL result file.
    """
    question_total = len(result.per_item)

    def sample_count(record: dict[str, Any]) -> int:
        """Count generations when available, otherwise one scored observation."""
        if "sample_total" in record:
            return max(int(record["sample_total"]), 0)
        return 1

    sample_total = sum(sample_count(record) for record in result.per_item)

    excluded = _count_excluded(result.per_item, sample_count)
    persist_results(
        cache_path,
        result.per_item,
        {
            "acc": round(result.acc, 4),
            "acc_norm": round(result.acc_norm, 4),
            "acc_bytes": round(result.acc_bytes, 4),
            "exact_match": round(result.exact_match, 4),
            "total": result.total,
            "question_total": question_total,
            "sample_total": sample_total,
            "effective_sample_count": max(
                sample_total
                - excluded["failed"]
                - excluded["skipped"]
                - excluded["timeout"],
                0,
            ),
            "failed_count": excluded["failed"],
            "skipped_count": excluded["skipped"],
            "timeout_count": excluded["timeout"],
            "aggregation": (
                result.per_item[0].get("aggregation") if result.per_item else None
            ),
        },
    )
