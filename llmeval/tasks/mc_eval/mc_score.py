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

import json
import os
import re
from collections import Counter
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.postprocess import (
    FilterRegistry,
    TextFilterPipeline,
    strip_reasoning_wrappers,
)
from llmeval.tasks.results import ScorerResult
from llmeval.utils.log import init_logger

__all__ = [
    "MCScoreResult",
    "build_result",
    "extract_answer",
    "merge_generate_records",
    "process_item",
    "score_generate",
    "score_generate_item",
    "score_generate_result",
    "score_items",
    "score_loglikelihood",
    "score_loglikelihood_item",
    "score_loglikelihood_result",
    "write_cache",
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
        Per-item scoring records (written to the JSONL cache file).
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
) -> ScorerResult:
    """Score loglikelihood-based MC results and return structured metrics.

    Parameters
    ----------
    eval_dataset:
        Items with ``gold`` / ``logprobs`` / ``choices`` fields (output of the
        MC inference loglikelihood mode). Newer inference outputs also include
        ``choice_tokens`` to record the actual answer tokens scored.
    cache_path:
        Path for the per-item JSONL cache.  A ``<stem>.summary.json`` metrics
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
    """Merge resumed rows for the same stable MC question.

    New inference output records sample indices explicitly. Older stable-ID
    rows without indices are assigned the next unused positions in file order.
    Legacy rows without ``doc_id`` remain independent because prompt text is
    not a safe dataset identity.
    """
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    samples_by_position: list[dict[int, str] | None] = []

    for source in eval_dataset:
        item = source.copy()
        document_id = item.get("doc_id")
        prompt = item.get("prompt", item.get("query", ""))
        if not document_id:
            merged.append(item)
            samples_by_position.append(None)
            continue

        identity = (str(document_id), str(prompt))
        position = positions.get(identity)
        if position is None:
            position = len(merged)
            positions[identity] = position
            merged.append(item)
            samples_by_position.append({})
        else:
            target = merged[position]
            for key in (label_key, "gold", "choices", "choice_tokens"):
                if key in item and key in target and item[key] != target[key]:
                    raise ValueError(
                        f"Conflicting {key!r} for resumed MC document {document_id!r}"
                    )
                if key in item and key not in target:
                    target[key] = item[key]

        target_samples = samples_by_position[position]
        assert target_samples is not None
        raw_generations = item.get(response_key, [])
        if isinstance(raw_generations, str):
            generations = [raw_generations]
        elif isinstance(raw_generations, list):
            generations = [str(value) for value in raw_generations]
        else:
            generations = []

        raw_indices = item.get("_llmeval_sample_indices")
        if (
            isinstance(raw_indices, list)
            and len(raw_indices) == len(generations)
            and all(isinstance(value, int) and value >= 0 for value in raw_indices)
        ):
            sample_indices = raw_indices
        else:
            sample_indices = []
            next_index = 0
            for _ in generations:
                while next_index in target_samples:
                    next_index += 1
                sample_indices.append(next_index)
                next_index += 1

        for sample_index, generation in zip(sample_indices, generations, strict=True):
            existing = target_samples.get(sample_index)
            if existing is not None and existing != generation:
                raise ValueError(
                    "Conflicting generation for resumed MC document "
                    f"{document_id!r}, sample {sample_index}"
                )
            target_samples[sample_index] = generation

    for item, samples in zip(merged, samples_by_position, strict=True):
        if samples is None:
            continue
        ordered_indices = sorted(samples)
        item[response_key] = [samples[index] for index in ordered_indices]
        item["_llmeval_sample_indices"] = ordered_indices
    return merged


def score_generate_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
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
    cache_path:
        Path for the per-item JSONL cache.  A ``<stem>.summary.json`` metrics
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

    # Clamp worker count: never more than items, available CPUs, or requested max.
    cpu_count = os.cpu_count() or 1
    optimal_workers = min(total, max_workers, max(1, cpu_count - 1))
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
            except Exception:
                logger.exception("Unexpected error retrieving scoring result.")
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
        gold = str(item.get(label_key, "")).strip().upper()
        pred = ""
    return {
        "gold": gold,
        "pred": pred,
        "correct": False,
        "correct_norm": False,
        "correct_bytes": False,
        "evaluation_status": status,
        "aggregation": aggregation,
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
            return idx, score_loglikelihood_item(item)
        return idx, score_generate_item(item, label_key, response_key, aggregation)
    except Exception:
        logger.exception("Scoring worker failed for item %d", idx)
        return idx, _error_record(item, mode, label_key, aggregation, "failed")


def build_result(records: list[dict[str, Any]]) -> MCScoreResult:
    """Aggregate per-item records into :class:`MCScoreResult`.

    ``correct_norm`` is only present in loglikelihood records; generate-mode
    records fall back to ``correct`` so that ``acc_norm == acc``.
    """
    per_sample = bool(records) and all(
        r.get("aggregation") == "per_sample" for r in records
    )

    def record_weight(record: dict[str, Any]) -> int:
        return max(int(record.get("sample_total", 0)), 0) if per_sample else 1

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

    effective_total = sum(record_weight(record) for record in eligible_records)
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

    def weight(record: dict[str, Any]) -> int:
        if record.get("aggregation") == "per_sample":
            return max(int(record.get("sample_total", 0)), 0)
        return 1

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

    sample_count = sum(weight(record) for record in result.per_item)
    failed_count = sum(
        weight(record)
        for record in result.per_item
        if record.get("evaluation_status") == "failed"
    )
    skipped_count = sum(
        weight(record)
        for record in result.per_item
        if record.get("evaluation_status") == "skipped"
    )
    timeout_count = sum(
        weight(record)
        for record in result.per_item
        if record.get("evaluation_status") == "timeout"
    )
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
            sample_count - failed_count - skipped_count - timeout_count, 0
        ),
        failed_count=failed_count,
        skipped_count=skipped_count,
        timeout_count=timeout_count,
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
    choice_logprobs = item.get("choice_logprobs")
    if isinstance(choice_logprobs, list) and len(choice_logprobs) == len(logprobs):
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
    choices = (
        item.get("choice_tokens") or item.get("choices") or item.get("choice_texts", [])
    )

    # Guard: unscorable data → forced-incorrect record.
    if not logprobs or gold < 0 or all(lp == float("-inf") for lp in logprobs):
        return {
            "gold": gold,
            "pred": -1,
            "correct": False,
            "correct_norm": False,
            "correct_bytes": False,
            "evaluation_status": "skipped" if gold < 0 else "failed",
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
        normed = [
            lp / float(count)
            for lp, count in zip(logprobs, char_count_values, strict=True)
        ]
        is_correct_norm = max(range(len(normed)), key=normed.__getitem__) == gold
    elif choices and len(choices) == len(logprobs):
        choice_lens = [max(len(str(choice)), 1) for choice in choices]
        normed = [lp / length for lp, length in zip(logprobs, choice_lens, strict=True)]
        is_correct_norm = max(range(len(normed)), key=normed.__getitem__) == gold
    else:
        is_correct_norm = is_correct

    if has_byte_counts:
        normed_bytes = [
            lp / float(count)
            for lp, count in zip(logprobs, byte_count_values, strict=True)
        ]
        is_correct_bytes = (
            max(range(len(normed_bytes)), key=normed_bytes.__getitem__) == gold
        )
    elif choices and len(choices) == len(logprobs):
        choice_bytes = [max(len(str(choice).encode("utf-8")), 1) for choice in choices]
        normed_bytes = [
            lp / length for lp, length in zip(logprobs, choice_bytes, strict=True)
        ]
        is_correct_bytes = (
            max(range(len(normed_bytes)), key=normed_bytes.__getitem__) == gold
        )
    else:
        is_correct_bytes = is_correct

    return {
        "gold": gold,
        "pred": pred,
        "correct": is_correct,
        "correct_norm": is_correct_norm,
        "correct_bytes": is_correct_bytes,
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

    gold = str(item.get(label_key, "")).strip().upper()
    generations: Any = item.get(response_key, [])

    if isinstance(generations, str):
        generation_texts = [generations]
    elif isinstance(generations, list):
        generation_texts = [str(generation) for generation in generations]
    else:
        generation_texts = []

    if not gold:
        status = "skipped"
    elif not generation_texts:
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
        "raw_gen": generation_texts,
        "filtered_gen": predictions,
        "filter_trace": [trace for _, trace in filtered],
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
       (word-boundary delimited) anywhere in the text.

    Both paths normalise to uppercase.  Returns ``""`` when no letter can
    be extracted.
    """
    # 1. Explicit "Answer: X" / "答案：X" marker.
    m = _ANSWER_MARKER_RE.search(text)
    if m:
        return m.group(1).upper()

    # 2. Last standalone A-J letter in the text.
    matches = _LAST_LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()

    return ""


MC_FILTER_REGISTRY.register("extract_answer", extract_answer, version="1")
MC_GENERATION_PIPELINE = MC_FILTER_REGISTRY.build_pipeline(
    "mc_generation", "1", "strip_reasoning", "extract_answer"
)


# ===========================================================================
# Cache persistence
# ===========================================================================


def write_cache(result: MCScoreResult, cache_path: str | Path) -> None:
    """Persist per-item records (JSONL) and an aggregated metrics summary (JSON).

    The summary is written to ``<cache_path>.summary.json`` alongside the
    JSONL file.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-item records (JSONL).
    with open(cache_path, "w", encoding="utf-8") as fh:
        for record in result.per_item:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Aggregated metrics summary (JSON).
    summary_path = cache_path.with_suffix(".summary.json")
    question_total = len(result.per_item)
    sample_total = sum(int(record.get("sample_total", 1)) for record in result.per_item)

    def weight(record: dict[str, Any]) -> int:
        return (
            max(int(record.get("sample_total", 0)), 0)
            if record.get("aggregation") == "per_sample"
            else 1
        )

    failed_count = sum(
        weight(record)
        for record in result.per_item
        if record.get("evaluation_status") == "failed"
    )
    skipped_count = sum(
        weight(record)
        for record in result.per_item
        if record.get("evaluation_status") == "skipped"
    )
    timeout_count = sum(
        weight(record)
        for record in result.per_item
        if record.get("evaluation_status") == "timeout"
    )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "acc": round(result.acc, 4),
                "acc_norm": round(result.acc_norm, 4),
                "acc_bytes": round(result.acc_bytes, 4),
                "exact_match": round(result.exact_match, 4),
                "total": result.total,
                "question_total": question_total,
                "sample_total": sample_total,
                "sample_count": sample_total,
                "effective_sample_count": max(
                    sample_total - failed_count - skipped_count - timeout_count, 0
                ),
                "failed_count": failed_count,
                "skipped_count": skipped_count,
                "timeout_count": timeout_count,
                "aggregation": (
                    result.per_item[0].get("aggregation") if result.per_item else None
                ),
            },
            fh,
            indent=2,
        )
