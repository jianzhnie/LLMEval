"""Multiple-choice scoring: loglikelihood and generation-based.

Aligned with lm-evaluation-harness.
Metrics: acc, acc_norm (length-normalized logprobs), exact_match.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCScoreResult:
    """MC evaluation metrics (lm-eval aligned)."""

    acc: float = 0.0  # argmax of raw logprobs
    acc_norm: float = 0.0  # argmax of length-normalized logprobs
    exact_match: float = 0.0  # alias for acc in MC context
    total: int = 0
    correct: int = 0
    correct_norm: int = 0
    per_item: list[dict] = field(default_factory=list)


# ===========================================================================
# Loglikelihood scoring
# ===========================================================================


def score_loglikelihood(
    eval_dataset: list[dict[str, Any]],
    cache_path: str | Path,
) -> float:
    """Score loglikelihood MC results. Returns acc (primary metric)."""
    metrics = _compute_loglikelihood_metrics(eval_dataset)
    _write_cache(metrics, cache_path)
    return metrics.acc


def _compute_loglikelihood_metrics(
    eval_dataset: list[dict[str, Any]],
) -> MCScoreResult:
    """Compute acc + acc_norm for loglikelihood results (lm-eval style).

    acc:      argmax of raw logprobs
    acc_norm: argmax of logprob / len(choice_text) for each choice
    """
    correct = 0
    correct_norm = 0
    total = 0
    per_item: list[dict] = []

    for item in eval_dataset:
        total += 1
        gold = int(item.get("gold", -1))
        logprobs = item.get("logprobs", [])
        choices = item.get("choices", []) or item.get("choice_texts", [])

        if not logprobs or gold < 0:
            per_item.append(
                {"gold": gold, "pred": -1, "correct": False, "correct_norm": False}
            )
            continue

        # acc: argmax of raw logprobs
        pred = _argmax(logprobs)
        is_correct = pred == gold
        if is_correct:
            correct += 1

        # acc_norm: length-normalized (lm-eval: logprob / len(choice))
        if choices and len(choices) == len(logprobs):
            choice_lens = [max(len(str(c)), 1) for c in choices]
            norm_lp = [lp / cl for lp, cl in zip(logprobs, choice_lens, strict=False)]
            pred_norm = _argmax(norm_lp)
            is_correct_norm = pred_norm == gold
        else:
            is_correct_norm = is_correct
        if is_correct_norm:
            correct_norm += 1

        per_item.append(
            {
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "correct_norm": is_correct_norm,
            }
        )

    total_f = max(total, 1)
    return MCScoreResult(
        acc=correct / total_f,
        acc_norm=correct_norm / total_f,
        exact_match=correct / total_f,
        total=total,
        correct=correct,
        correct_norm=correct_norm,
        per_item=per_item,
    )


# ===========================================================================
# Generate scoring
# ===========================================================================


def score_generate(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
) -> float:
    """Score generation-based MC results by extracting the answer letter."""
    correct = 0
    total = 0
    per_item: list[dict] = []

    for item in eval_dataset:
        total += 1
        gold = str(item.get(label_key, "")).strip().upper()
        gen_list = item.get(response_key, [])
        pred_text = str(gen_list[0]) if gen_list else ""

        pred = _extract_answer(pred_text)
        is_correct = pred == gold
        if is_correct:
            correct += 1

        per_item.append({"gold": gold, "pred": pred, "correct": is_correct})

    accuracy = correct / total if total > 0 else 0.0
    result = MCScoreResult(
        acc=accuracy,
        acc_norm=accuracy,
        exact_match=accuracy,
        total=total,
        correct=correct,
        correct_norm=correct,
        per_item=per_item,
    )
    _write_cache(result, cache_path)
    return accuracy


# ===========================================================================
# Helpers
# ===========================================================================


def _argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def _write_cache(result: MCScoreResult, cache_path: str | Path) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for r in result.per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
    """Extract answer letter from model output, aligned with lm-eval patterns."""
    m = re.search(
        r"(?:Answer|答案)\s*[:：]\s*([A-J])\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    matches = re.findall(r"\b([A-J])\b", text)
    if matches:
        return matches[-1]

    return ""
