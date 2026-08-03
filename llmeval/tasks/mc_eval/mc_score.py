"""Multiple-choice scoring: loglikelihood and generation-based evaluation.

Aligned with lm-evaluation-harness metric definitions.

Metrics
-------
acc         — accuracy (argmax of raw logprobs, or letter extraction in generate mode)
acc_norm    — accuracy with length-normalized logprobs (loglikelihood only; equals acc in generate)
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
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.utils.log import init_logger

__all__ = [
    "MCScoreResult",
    "build_result",
    "extract_answer",
    "process_item",
    "score_generate",
    "score_generate_item",
    "score_items",
    "score_loglikelihood",
    "score_loglikelihood_item",
    "write_cache",
]

logger = init_logger("mc_score")

# Precompiled answer-extraction regexes.
_ANSWER_MARKER_RE: re.Pattern[str] = re.compile(
    r"(?:Answer|答案)\s*[:：]\s*([A-J])\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_LAST_LETTER_RE: re.Pattern[str] = re.compile(r"\b([A-Ja-j])\b")


@dataclass
class MCScoreResult:
    """Aggregate MC evaluation metrics (lm-eval aligned).

    Attributes
    ----------
    acc:
        Accuracy via argmax of raw logprobs (loglikelihood), or via answer-letter
        extraction from generated text (generate mode).
    acc_norm:
        Accuracy via length-normalized logprobs.  In generate mode this always
        equals *acc* because generate records carry no ``correct_norm`` field.
    exact_match:
        Convenience alias for *acc* in the multiple-choice context.
    total:
        Number of items scored.
    correct:
        Number of items answered correctly under *acc*.
    correct_norm:
        Number of items answered correctly under *acc_norm*.
    per_item:
        Per-item scoring records (written to the JSONL cache file).
    """

    acc: float = 0.0
    acc_norm: float = 0.0
    exact_match: float = 0.0
    total: int = 0
    correct: int = 0
    correct_norm: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


# ===========================================================================
# Entry points
# ===========================================================================


def score_loglikelihood(
    eval_dataset: list[dict[str, Any]],
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
) -> float:
    """Score loglikelihood-based MC results and persist cache files.

    Parameters
    ----------
    eval_dataset:
        Items with ``gold`` / ``logprobs`` / ``choices`` fields (output of the
        MC inference loglikelihood mode).
    cache_path:
        Path for the per-item JSONL cache.  A ``<stem>.summary.json`` metrics
        file is written alongside it.
    max_workers:
        Maximum process-pool workers (capped by dataset size and CPU count).
    timeout:
        Per-item scoring timeout in seconds.

    Returns
    -------
    float
        Accuracy (*acc* metric).
    """
    if any(
        not (item.get("choices") or item.get("choice_texts")) for item in eval_dataset
    ):
        logger.warning(
            "Some items have no 'choices' field — acc_norm will fall back to acc."
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
    return metrics.acc


def score_generate(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
) -> float:
    """Score generation-based MC results by extracting the answer letter.

    Only the **first** sample of each generation list is evaluated.

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

    Returns
    -------
    float
        Accuracy (also reported as *acc_norm* / *exact_match* in the summary).
    """
    records = score_items(
        eval_dataset,
        mode="generate",
        label_key=label_key,
        response_key=response_key,
        max_workers=max_workers,
        timeout=timeout,
    )
    metrics = build_result(records)
    write_cache(metrics, cache_path)
    return metrics.acc


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
) -> list[dict[str, Any]]:
    """Score every item, preserving input order.

    When the dataset is small or workers are limited, scoring runs serially
    to avoid pool overhead.  Otherwise a :class:`~pebble.ProcessPool` is used
    so that large benchmarks (e.g. MMLU ~14k items) finish quickly.

    Timed-out or crashed worker tasks are replaced with forced-incorrect
    records so that a single bad item never aborts the whole run.
    """
    total = len(eval_dataset)
    if total == 0:
        return []
    if max_workers <= 1 or total == 1:
        # Serial path — avoids pool startup cost for tiny workloads.
        return [
            process_item((i, item, mode, label_key, response_key))[1]
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
            (i, item, mode, label_key, response_key)
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
                logger.warning("Individual scoring task timed out — marked as failed.")
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

    # Replace missing entries (pool timeouts, worker crashes) with
    # forced-incorrect records so they never inflate accuracy.
    records: list[dict[str, Any]] = []
    for i, item in enumerate(eval_dataset):
        record = results_by_index.get(i)
        if record is None:
            if mode == "loglikelihood":
                try:
                    gold = int(item.get("gold", -1))
                except (TypeError, ValueError):
                    gold = -1
                record = {
                    "gold": gold,
                    "pred": -1,
                    "correct": False,
                    "correct_norm": False,
                }
            else:
                record = {
                    "gold": str(item.get(label_key, "")).strip().upper(),
                    "pred": "",
                    "correct": False,
                }
        records.append(record)
    return records


def process_item(
    args: tuple[int, dict[str, Any], Literal["loglikelihood", "generate"], str, str],
) -> tuple[int, dict[str, Any]]:
    """Pool-worker entry point — **must** be module-level for pickling.

    Takes an ``(index, item, mode, label_key, response_key)`` tuple and returns
    ``(original_index, scored_record)`` so results can be re-ordered after
    parallel execution.
    """
    idx, item, mode, label_key, response_key = args
    if mode == "loglikelihood":
        return idx, score_loglikelihood_item(item)
    return idx, score_generate_item(item, label_key, response_key)


def build_result(records: list[dict[str, Any]]) -> MCScoreResult:
    """Aggregate per-item records into :class:`MCScoreResult`.

    ``correct_norm`` is only present in loglikelihood records; generate-mode
    records fall back to ``correct`` so that ``acc_norm == acc``.
    """
    total = len(records)
    n_correct = sum(1 for r in records if r["correct"])
    n_correct_norm = sum(1 for r in records if r.get("correct_norm", r["correct"]))

    safe_total = max(total, 1)
    return MCScoreResult(
        acc=n_correct / safe_total,
        acc_norm=n_correct_norm / safe_total,
        exact_match=n_correct / safe_total,
        total=total,
        correct=n_correct,
        correct_norm=n_correct_norm,
        per_item=records,
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
    choices = item.get("choices") or item.get("choice_texts", [])

    # Guard: unscorable data → forced-incorrect record.
    if not logprobs or gold < 0 or all(lp == float("-inf") for lp in logprobs):
        return {
            "gold": gold,
            "pred": -1,
            "correct": False,
            "correct_norm": False,
        }

    # acc — argmax over raw logprobs (ties broken by smallest index).
    pred = max(range(len(logprobs)), key=logprobs.__getitem__)
    is_correct = pred == gold

    # acc_norm — length-normalised logprobs (lm-eval convention).
    if choices and len(choices) == len(logprobs):
        choice_lens = [max(len(str(c)), 1) for c in choices]
        normed = [lp / cl for lp, cl in zip(logprobs, choice_lens, strict=False)]
        is_correct_norm = max(range(len(normed)), key=normed.__getitem__) == gold
    else:
        is_correct_norm = is_correct

    return {
        "gold": gold,
        "pred": pred,
        "correct": is_correct,
        "correct_norm": is_correct_norm,
    }


def score_generate_item(
    item: dict[str, Any],
    label_key: str,
    response_key: str,
) -> dict[str, Any]:
    """Score a single generate-mode item by extracting the answer letter.

    Edge cases handled defensively: empty gold or unparseable generation is
    always counted wrong (otherwise ``"" == ""`` would inflate accuracy);
    bare-string ``gen`` fields are tolerated (the schema expects a list).
    """
    gold = str(item.get(label_key, "")).strip().upper()
    generations: Any = item.get(response_key, [])

    if isinstance(generations, str):
        pred_text = generations
    else:
        pred_text = str(generations[0]) if generations else ""

    pred = extract_answer(pred_text)
    # Both must be non-empty for a match — prevents "" == "".
    is_correct = bool(gold) and bool(pred) and pred == gold

    return {"gold": gold, "pred": pred, "correct": is_correct}


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
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "acc": round(result.acc, 4),
                "acc_norm": round(result.acc_norm, 4),
                "exact_match": round(result.exact_match, 4),
                "total": result.total,
            },
            fh,
            indent=2,
        )
