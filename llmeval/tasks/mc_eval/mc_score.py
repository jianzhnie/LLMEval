"""Multiple-choice scoring: loglikelihood and generation-based.

Aligned with lm-evaluation-harness approach.
- loglikelihood: compare log-prob of each choice, pick argmax
- generate: extract answer letter from generated text, match
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def score_loglikelihood(
    eval_dataset: list[dict[str, Any]],
    cache_path: str | Path,
) -> float:
    """Score loglikelihood MC results.

    Each item should have:
        gold: ground truth answer index (int)
        logprobs: list of log-probabilities for each choice

    Predicted = argmax(logprobs). Correct = pred == gold.
    """
    correct = 0
    total = 0
    results = []

    for item in eval_dataset:
        total += 1
        gold = int(item.get("gold", -1))
        logprobs = item.get("logprobs", [])

        if not logprobs or gold < 0:
            results.append({"gold": gold, "pred": -1, "correct": False})
            continue

        pred = max(range(len(logprobs)), key=lambda i: logprobs[i])
        is_correct = pred == gold
        if is_correct:
            correct += 1
        results.append({"gold": gold, "pred": pred, "correct": is_correct})

    accuracy = correct / total if total > 0 else 0.0

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return accuracy


def score_generate(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
) -> float:
    """Score generation-based MC results by extracting the answer letter.

    Extraction (aligned with lm-eval):
    1. Match "Answer: X" or "答案：X" pattern at end of response
    2. Last standalone capital letter A-J in the response
    """
    correct = 0
    total = 0
    results = []

    for item in eval_dataset:
        total += 1
        gold = str(item.get(label_key, "")).strip().upper()
        gen_list = item.get(response_key, [])
        pred_text = str(gen_list[0]) if gen_list else ""

        pred = _extract_answer(pred_text)
        is_correct = pred == gold
        if is_correct:
            correct += 1

        results.append({"gold": gold, "pred": pred, "correct": is_correct})

    accuracy = correct / total if total > 0 else 0.0

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return accuracy


def _extract_answer(text: str) -> str:
    """Extract answer letter from model output, aligned with lm-eval patterns."""
    # Strategy 1: "Answer: X" or "答案：X" at end of line
    m = re.search(
        r'(?:Answer|答案)\s*[:：]\s*([A-J])\s*$',
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Strategy 2: Last standalone capital letter A-J
    matches = re.findall(r'\b([A-J])\b', text)
    if matches:
        return matches[-1]

    return ""
