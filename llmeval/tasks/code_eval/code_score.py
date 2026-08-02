"""Scoring driver for code-generation evaluation tasks (HumanEval / Pass@k).

This module follows the established pattern from ``mc_score.py``: a module-level
picklable worker, a serial / parallel dispatcher, an aggregate result dataclass,
and JSONL + summary cache persistence.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.code_eval.execute import check_correctness
from llmeval.utils.logger import init_logger

logger = init_logger("code_score")

__all__ = [
    "CodeScoreResult",
    "estimate_pass_at_k",
    "extract_code",
    "score_code",
    "write_cache",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EXEC_TIMEOUT: float = 3.0

# Precompiled regex — fenced code block extraction
_FENCE_RE: re.Pattern[str] = re.compile(
    r"```(?:\w+)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)

# Line where actual Python code starts.
_CODE_START_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:def |from |import |class |@)",
    re.MULTILINE,
)

# Lines that suggest the model output continues past the actual code body.
# Only matches at column 0 (top-level) — indented def/class inside a function
# body are *not* stop markers.
_STOP_MARKERS: re.Pattern[str] = re.compile(
    r"^(?:class |def |if __name__|print\b|#)",
    re.MULTILINE,
)

# Think-tag stripping for reasoning-model outputs (deepseek_r1 / openr1).
_THINK_RE: re.Pattern[str] = re.compile(
    r"<think[^>]*>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_END_RE: re.Pattern[str] = re.compile(
    r"</think\s*>",
    re.IGNORECASE,
)
_ANSWER_TAG_RE: re.Pattern[str] = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str:
    """Extract runnable Python code from a model-generated string.

    Cascades through three strategies (first match wins):

    1. **Fenced code block** — `` ```python ... ``` `` or bare `` ``` ... ``` ``.
    2. **Code-start heuristic** — text from the first ``def`` / ``from`` /
       ``import`` / ``class`` line onward, truncated at the **next** stop
       marker (``class``, ``def``, ``if __name__``, ``print``, ``#``) that
       appears after the initial code block.
    3. **Raw fallback** — the original text, stripped.

    Returns ``""`` for empty / non-string inputs.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strategy 1 — fenced block
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).rstrip()

    # Strategy 2 — first code line to first stop marker
    start_m = _CODE_START_RE.search(text)
    if start_m:
        body = text[start_m.start() :]
        # Skip past the initial code-start line.
        first_line_end = body.find("\n")
        if first_line_end == -1:
            return body.rstrip()
        after_first_line = body[first_line_end + 1 :]
        # Find the next stop marker at column 0 — these are top-level
        # constructs, not indented body lines.
        stop_m = _STOP_MARKERS.search(after_first_line)
        if stop_m:
            body = body[: first_line_end + 1 + stop_m.start()]
        return body.rstrip()

    # Strategy 3 — raw fallback
    return text.rstrip()


# ---------------------------------------------------------------------------
# Pass@k estimation
# ---------------------------------------------------------------------------


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    ``pass@k := 1 - C(n-c, k) / C(n, k)``, computed in a numerically-stable
    product form.

    When *k* = 1 this simplifies to ``num_correct / num_samples``.
    """
    n, c = int(num_samples), int(num_correct)
    if n < 1:
        return 0.0
    if c < 0:
        c = 0
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(n - c + 1, n + 1):
        result *= 1.0 - k / i
    return 1.0 - result


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CodeScoreResult:
    """Aggregate code evaluation metrics."""

    pass_at_1: float = 0.0
    total: int = 0
    correct: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------


def _failure_code_record(item: dict[str, Any]) -> dict[str, Any]:
    """Build a placeholder record for items that could not be scored."""
    return {
        "task_id": item.get("task_id", ""),
        "passed": False,
        "result": "scoring error",
        "stderr": "",
    }


def _strip_think_tags(text: str) -> str:
    """Remove reasoning-model output wrappers from *text*.

    - Prefer content inside ``<answer>...</answer>``.
    - Fall back to text after ``</think>``.
    - Otherwise return *text* unchanged.
    """
    _am = _ANSWER_TAG_RE.search(text)
    if _am:
        return _am.group(1)
    _tm = _THINK_END_RE.search(text)
    if _tm:
        return text[_tm.end() :]
    return text


# ---------------------------------------------------------------------------
# Per-item worker (module-level → picklable)
# ---------------------------------------------------------------------------


def _process_code_item(
    args: tuple[int, dict[str, Any], str, str, float],
) -> tuple[int, dict[str, Any]]:
    """Pool-worker entry point — **must** be module-level for pickling.

    Parameters
    ----------
    args:
        ``(index, item, label_key, response_key, exec_timeout)``.

    Returns
    -------
    (original_index, scored_record)
        *record* is a dict with keys ``task_id``, ``passed``, ``result``,
        ``stderr``.  The caller re-sorts by *original_index*.
    """
    idx, item, label_key, response_key, exec_timeout = args

    task_id = item.get("task_id", f"task_{idx}")
    prompt: str = str(item.get("prompt", ""))
    test_code: str = str(item.get(label_key, ""))

    # --- extract model output ---------------------------------------------------
    gen_raw = item.get(response_key)
    if isinstance(gen_raw, list):
        gen_str = str(gen_raw[0]) if gen_raw else ""
    elif isinstance(gen_raw, str):
        gen_str = gen_raw
    else:
        gen_str = ""

    if not gen_str.strip():
        return idx, {
            "task_id": task_id,
            "passed": False,
            "result": "failed: empty generation",
            "stderr": "",
        }

    # Strip reasoning-model output wrappers before code extraction.
    # Prefer <answer>...</answer> content; fall back to text after </think>;
    # otherwise keep the original string unchanged.
    gen_str = _strip_think_tags(gen_str)

    code = extract_code(gen_str)

    if not code.strip():
        return idx, {
            "task_id": task_id,
            "passed": False,
            "result": "failed: no code extracted",
            "stderr": "",
        }

    # --- construct the check program --------------------------------------------
    # extract_code() now uses .rstrip() so indentation is preserved for
    # HumanEval-style bare function bodies (e.g. "    return a + b").
    candidate = prompt.rstrip() + "\n" + code
    check_program = candidate + "\n" + test_code

    # --- execute -----------------------------------------------------------------
    exec_result = check_correctness(check_program, exec_timeout, task_id)
    exec_result.setdefault("task_id", task_id)
    return idx, exec_result


# ---------------------------------------------------------------------------
# Dispatcher (serial / parallel)
# ---------------------------------------------------------------------------


def _score_items(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    exec_timeout: float,
    max_workers: int,
    timeout: int,
) -> list[dict[str, Any]]:
    """Score every item, preserving input order.

    Parameters
    ----------
    eval_dataset:
        List of item dicts (must have *label_key* and *response_key*).
    label_key:
        Dict key for the ground-truth test code.
    response_key:
        Dict key for the model outputs (list or string).
    exec_timeout:
        Per-item code execution timeout (seconds).
    max_workers:
        Maximum parallel workers; ≤1 forces a serial run.
    timeout:
        Pebble pool-level timeout per task.

    Returns
    -------
    list[dict[str, Any]]
        One scored record per item, in the same order as *eval_dataset*.
        Failed items are represented by :func:`_failure_code_record`.
    """
    total = len(eval_dataset)
    if total == 0:
        return []

    # --- serial fast-path -------------------------------------------------------
    if max_workers <= 1 or total == 1:
        records: list[dict[str, Any]] = []
        for i, item in enumerate(eval_dataset):
            _idx, rec = _process_code_item(
                (i, item, label_key, response_key, exec_timeout),
            )
            records.append(rec)
        return records

    # --- parallel path ----------------------------------------------------------
    cpu_count = os.cpu_count() or 1
    optimal_workers = min(total, max_workers, max(1, cpu_count - 1))
    results_by_index: dict[int, dict[str, Any]] = {}

    iterable = [
        (i, item, label_key, response_key, exec_timeout)
        for i, item in enumerate(eval_dataset)
    ]

    with (
        tqdm(total=total, desc="Scoring code", unit="item") as pbar,
        ProcessPool(max_workers=optimal_workers) as pool,
    ):
        future = pool.map(_process_code_item, iterable, timeout=timeout)
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

    return [
        results_by_index.get(i) or _failure_code_record(item)
        for i, item in enumerate(eval_dataset)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_code(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 20,
    exec_timeout: float = _DEFAULT_EXEC_TIMEOUT,
) -> float:
    """Score a code-generation dataset and return the Pass@1 accuracy.

    Parameters
    ----------
    eval_dataset:
        List of dicts.  Each dict must contain *label_key* (test code) and
        *response_key* (model output, list-of-strings or bare string).
        Only the **first** sample of multi-sample outputs is evaluated.
    label_key:
        Dict key for the ground-truth test harness (e.g. ``"answer"``).
    response_key:
        Dict key for the model output (e.g. ``"gen"``).  List or string.
    cache_path:
        File path for the per-item JSONL cache.  A ``.summary.json`` is also
        written alongside it.
    max_workers:
        Maximum number of :class:`pebble.ProcessPool` workers.  Set ≤1 for
        serial execution.
    timeout:
        Pool-level timeout per worker task (seconds).
    exec_timeout:
        Per-item code execution timeout (seconds).  Default 3.0.

    Returns
    -------
    float
        Pass@1 score in [0.0, 1.0].
    """
    total = len(eval_dataset)
    if total == 0:
        logger.warning("Empty dataset — returning 0.0")
        return 0.0

    logger.info(
        "Scoring %d item(s) with max_workers=%d, exec_timeout=%.1fs",
        total,
        max_workers,
        exec_timeout,
    )

    records = _score_items(
        eval_dataset,
        label_key,
        response_key,
        exec_timeout,
        max_workers,
        timeout,
    )

    correct = sum(1 for r in records if r.get("passed"))
    pass_at_1 = correct / max(total, 1)

    # Log failures for diagnostics.
    failures = [r for r in records if not r.get("passed")]
    if failures:
        logger.info(
            "%d failure(s): %s",
            len(failures),
            ", ".join(r.get("result", "?") for r in failures[:10]),
        )

    result = CodeScoreResult(
        pass_at_1=pass_at_1,
        total=total,
        correct=correct,
        per_item=records,
    )
    write_cache(result, cache_path)

    logger.info("Pass@1: %.2f%% (%d/%d)", pass_at_1 * 100, correct, total)
    return pass_at_1


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


def write_cache(result: CodeScoreResult, cache_path: str | Path) -> None:
    """Write per-item JSONL and a ``.summary.json`` metrics file.

    Matches the pattern established by :func:`llmeval.tasks.mc_eval.mc_score.write_cache`.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as fh:
        for record in result.per_item:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = cache_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pass_at_1": round(result.pass_at_1, 6),
                "total": result.total,
                "correct": result.correct,
            },
            fh,
            indent=2,
        )
