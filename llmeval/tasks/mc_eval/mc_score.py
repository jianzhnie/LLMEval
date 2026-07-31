"""Multiple-choice scoring: loglikelihood and generation-based.

Aligned with lm-evaluation-harness.
Metrics: acc, acc_norm (length-normalized logprobs), exact_match.

Public API:
- score_loglikelihood: score results produced by mc_infer loglikelihood mode
- score_generate:      score results produced by mc_infer generate mode
- MCScoreResult:       metrics container (also written to the cache files)

Per-item scoring runs in a pebble ProcessPool (same architecture as
math_eval/math_score.py) so large datasets score in parallel; small inputs
fall back to serial execution without pool overhead. Only light project
dependencies (pebble/tqdm) are used, so scoring stays independent of the
inference environment (no openai/torch).
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.utils.logger import init_logger

__all__ = ["MCScoreResult", "score_generate", "score_loglikelihood"]

logger = init_logger("mc_score")

# Input field keys (mc_infer output schema)
GOLD_KEY = "gold"
LOGPROBS_KEY = "logprobs"
CHOICES_KEY = "choices"

# Answer extraction patterns (precompiled; see _extract_answer)
_ANSWER_MARKER_RE = re.compile(
    r"(?:Answer|答案)\s*[:：]\s*([A-J])\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_LETTER_FALLBACK_RE = re.compile(r"\b([A-Ja-j])\b")

# Scoring modes supported by _process_item
MODE_LOGLIKELIHOOD = "loglikelihood"
MODE_GENERATE = "generate"


@dataclass
class MCScoreResult:
    """MC evaluation metrics (lm-eval aligned).

    Attributes:
        acc: Accuracy via argmax of raw logprobs (or extraction in generate mode)
        acc_norm: Accuracy via length-normalized logprobs (loglikelihood only;
            equals acc in generate mode)
        exact_match: Alias for acc in the MC context
        total: Number of scored items
        correct: Count correct under acc
        correct_norm: Count correct under acc_norm
        per_item: Per-item records written to the cache file
    """

    acc: float = 0.0
    acc_norm: float = 0.0
    exact_match: float = 0.0
    total: int = 0
    correct: int = 0
    correct_norm: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


# ===========================================================================
# Loglikelihood scoring
# ===========================================================================


def score_loglikelihood(
    eval_dataset: list[dict[str, Any]],
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
) -> float:
    """Score loglikelihood MC results and write the cache files.

    Args:
        eval_dataset: Items with gold / logprobs / choices (mc_infer output)
        cache_path: Per-item records JSONL path; a `<name>.summary.json`
            metrics file is written next to it
        max_workers: Process-pool size (capped by dataset size and CPUs)
        timeout: Per-item scoring timeout in seconds

    Returns:
        acc (the primary metric)
    """
    if any(
        not (item.get(CHOICES_KEY) or item.get("choice_texts")) for item in eval_dataset
    ):
        logger.warning(
            "Items have no 'choices' field; acc_norm falls back to acc. "
            "(mc_infer >= 2026-07-30 writes choices into results)"
        )
    records = _score_items(
        eval_dataset,
        mode=MODE_LOGLIKELIHOOD,
        label_key="",
        response_key="",
        max_workers=max_workers,
        timeout=timeout,
    )
    metrics = _build_result(records)
    _write_cache(metrics, cache_path)
    return metrics.acc


def _compute_loglikelihood_metrics(
    eval_dataset: list[dict[str, Any]],
) -> MCScoreResult:
    """Compute acc + acc_norm serially (used by tests).

    acc:      argmax of raw logprobs
    acc_norm: argmax of logprob / len(choice_text) for each choice
    """
    records = [_score_loglikelihood_item(item) for item in eval_dataset]
    return _build_result(records)


# ===========================================================================
# Generate scoring
# ===========================================================================


def score_generate(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 60,
) -> float:
    """Score generation-based MC results by extracting the answer letter.

    Args:
        eval_dataset: Items with a label field and a generation list field
        label_key: Field name of the gold answer letter (e.g. "answer")
        response_key: Field name of the generations list (e.g. "gen");
            only the first sample is scored
        cache_path: Per-item records JSONL path (summary written next to it)
        max_workers: Process-pool size (capped by dataset size and CPUs)
        timeout: Per-item scoring timeout in seconds

    Returns:
        Accuracy (also reported as acc/acc_norm/exact_match in the summary)
    """
    records = _score_items(
        eval_dataset,
        mode=MODE_GENERATE,
        label_key=label_key,
        response_key=response_key,
        max_workers=max_workers,
        timeout=timeout,
    )
    metrics = _build_result(records)
    _write_cache(metrics, cache_path)
    return metrics.acc


# ===========================================================================
# Parallel driver (mirrors math_eval/math_score.py)
# ===========================================================================


def _process_item(args: tuple[int, dict[str, Any], str, str, str]) -> tuple[int, dict[str, Any]]:
    """Pool worker: score one item. Must stay module-level (picklable).

    Args:
        args: (index, item, mode, label_key, response_key)

    Returns:
        (index, per-item record)
    """
    idx, item, mode, label_key, response_key = args
    if mode == MODE_LOGLIKELIHOOD:
        return idx, _score_loglikelihood_item(item)
    return idx, _score_generate_item(item, label_key, response_key)


def _score_items(
    eval_dataset: list[dict[str, Any]],
    mode: str,
    label_key: str,
    response_key: str,
    max_workers: int,
    timeout: int,
) -> list[dict[str, Any]]:
    """Score all items, preserving input order.

    Uses a pebble ProcessPool when there is enough work; tiny inputs and
    max_workers<=1 take a serial path. Timed-out or crashed items are
    recorded as failed so one bad item never aborts the run.
    """
    total = len(eval_dataset)
    if total == 0:
        return []
    if max_workers <= 1 or total == 1:
        return [
            _process_item((i, item, mode, label_key, response_key))[1]
            for i, item in enumerate(eval_dataset)
        ]

    # Optimize worker count based on system resources.
    # Use min(total, max_workers, cpu_count-1) to avoid over-provisioning
    # for small datasets.
    cpu_count = os.cpu_count() or 1
    optimal_workers = min(total, max_workers, max(1, cpu_count - 1))
    records: dict[int, dict[str, Any]] = {}

    with (
        tqdm(total=total, desc="Scoring items", unit="item") as pbar,
        ProcessPool(max_workers=optimal_workers) as pool,
    ):
        future = pool.map(
            _process_item,
            [
                (i, item, mode, label_key, response_key)
                for i, item in enumerate(eval_dataset)
            ],
            timeout=timeout,
        )
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError:
                logger.warning("Individual scoring task timed out; marked failed")
                pbar.update(1)
                continue
            except Exception as e:
                logger.error(f"Error retrieving scoring result: {e}")
                pbar.update(1)
                continue

            if result is not None:
                idx, record = result
                records[idx] = record
            pbar.update(1)

    # Fill gaps (timeouts / worker crashes) as failed records
    return [
        records.get(i) or _failure_record(item, mode, label_key)
        for i, item in enumerate(eval_dataset)
    ]


def _build_result(records: list[dict[str, Any]]) -> MCScoreResult:
    """Aggregate per-item records into the final metrics container."""
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    # Generate-mode records have no correct_norm; it falls back to correct
    # there, so acc_norm == acc for generate mode.
    correct_norm = sum(1 for r in records if r.get("correct_norm", r["correct"]))
    total_f = max(total, 1)
    return MCScoreResult(
        acc=correct / total_f,
        acc_norm=correct_norm / total_f,
        exact_match=correct / total_f,
        total=total,
        correct=correct,
        correct_norm=correct_norm,
        per_item=records,
    )


# ===========================================================================
# Per-item scoring
# ===========================================================================


def _score_loglikelihood_item(item: dict[str, Any]) -> dict[str, Any]:
    """Score one loglikelihood item; returns its per-item record.

    Unscorable items (missing logprobs, invalid gold, or all -inf logprobs —
    the residue of a failed inference) are marked incorrect with pred=-1.
    """
    gold = _parse_gold(item.get(GOLD_KEY, -1))
    logprobs: list[float] = item.get(LOGPROBS_KEY, [])
    choices = item.get(CHOICES_KEY, []) or item.get("choice_texts", [])

    # 全 -inf 是推理失败的产物（不应出现于此，防御性按错误处理）
    if not logprobs or gold < 0 or all(lp == float("-inf") for lp in logprobs):
        return {"gold": gold, "pred": -1, "correct": False, "correct_norm": False}

    # acc: argmax of raw logprobs
    pred = _argmax(logprobs)
    is_correct = pred == gold

    # acc_norm: length-normalized (lm-eval: logprob / len(choice));
    # falls back to acc when choices are unavailable
    if choices and len(choices) == len(logprobs):
        choice_lens = [max(len(str(c)), 1) for c in choices]
        norm_lp = [lp / cl for lp, cl in zip(logprobs, choice_lens, strict=False)]
        is_correct_norm = _argmax(norm_lp) == gold
    else:
        is_correct_norm = is_correct

    return {
        "gold": gold,
        "pred": pred,
        "correct": is_correct,
        "correct_norm": is_correct_norm,
    }


def _score_generate_item(
    item: dict[str, Any], label_key: str, response_key: str
) -> dict[str, Any]:
    """Score one generate-mode item; returns its per-item record.

    An empty gold or an unparseable generation never counts as correct
    ("" == "" would otherwise inflate accuracy).
    """
    gold = str(item.get(label_key, "")).strip().upper()
    generations: Any = item.get(response_key, [])
    # Tolerate a bare-string generation (schema expects a list)
    if isinstance(generations, str):
        pred_text = generations
    else:
        pred_text = str(generations[0]) if generations else ""

    pred = _extract_answer(pred_text)
    # 空 gold 或空 pred 不能判对：两者皆空时 == 成立会虚增准确率
    is_correct = bool(gold) and bool(pred) and pred == gold
    return {"gold": gold, "pred": pred, "correct": is_correct}


def _failure_record(item: dict[str, Any], mode: str, label_key: str) -> dict[str, Any]:
    """Build a failed per-item record for timed-out / crashed pool tasks."""
    if mode == MODE_LOGLIKELIHOOD:
        return {
            "gold": _parse_gold(item.get(GOLD_KEY, -1)),
            "pred": -1,
            "correct": False,
            "correct_norm": False,
        }
    return {
        "gold": str(item.get(label_key, "")).strip().upper(),
        "pred": "",
        "correct": False,
    }


# ===========================================================================
# Helpers
# ===========================================================================


def _argmax(xs: list[float]) -> int:
    """Return index of maximum value."""
    return max(enumerate(xs), key=lambda x: x[1])[0]


def _parse_gold(value: Any) -> int:
    """Parse a gold index defensively; non-numeric values become -1 (invalid)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid gold value {value!r}; treating as -1")
        return -1


def _write_cache(result: MCScoreResult, cache_path: str | Path) -> None:
    """Write per-item records (JSONL) and a metrics summary (JSON) to disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for record in result.per_item:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Metrics summary
    summary_path = cache_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "acc": round(result.acc, 4),
                "acc_norm": round(result.acc_norm, 4),
                "exact_match": round(result.exact_match, 4),
                "total": result.total,
            },
            f,
            indent=2,
        )


def _extract_answer(text: str) -> str:
    """Extract answer letter from model output, aligned with lm-eval patterns.

    Prefers an explicit "Answer: X" / "答案: X" marker; otherwise falls back to
    the LAST standalone letter A-J (case-insensitive, normalized to upper).
    Returns "" when no letter can be extracted.
    """
    m = _ANSWER_MARKER_RE.search(text)
    if m:
        return m.group(1).upper()

    matches = _LETTER_FALLBACK_RE.findall(text)
    if matches:
        return matches[-1].upper()

    return ""
