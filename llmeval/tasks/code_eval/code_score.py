"""Scoring driver for code-generation evaluation (HumanEval / MBPP / Pass@k).

Architecture matches ``mc_score.py``:
    * Module-level picklable worker (``_process_code_item``)
    * Serial / parallel dispatcher (``_score_items``)
    * Aggregate result dataclass (``CodeScoreResult``)
    * Structured in-memory result returned to the task registry
"""

from __future__ import annotations

import ast
import math
import re
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.code_eval.execute import (
    PROCESS_JOIN_MARGIN_SECONDS,
    PROCESS_KILL_MARGIN,
    check_correctness,
)
from llmeval.tasks.postprocess import (
    TextFilterPipeline,
    normalize_single_generation_samples,
    resolve_max_workers,
    resolve_single_generation,
    strip_reasoning_wrappers,
)
from llmeval.tasks.registry import ScorerResult
from llmeval.utils.log import init_logger

logger = init_logger("code_score")

__all__ = [
    "CodeScoreResult",
    "estimate_pass_at_k",
    "extract_code",
    "score_code_result",
]

# ===========================================================================
# Constants
# ===========================================================================

_DEFAULT_EXEC_TIMEOUT: float = 3.0
"""Default per-item code execution timeout in seconds."""

_MAX_CHECK_PROGRAMS = 2
_POOL_COORDINATOR_MARGIN_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Precompiled regular expressions
# ---------------------------------------------------------------------------

_FENCE_RE: re.Pattern[str] = re.compile(
    r"```(?:\w+)?\s*\n(.*?)\n?\s*```",
    re.DOTALL,
)
"""Fenced code block: opening fence + code + closing fence. The newline
before the closing fence is optional so compact outputs still match."""

_CODE_START_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:def |from |import |class |@)",
    re.MULTILINE,
)
"""First line that looks like the start of actual Python code."""

_STOP_MARKERS: re.Pattern[str] = re.compile(
    r"^(?:if __name__|print\b)",
    re.MULTILINE,
)
"""Top-level executable markers after which the code block is considered done."""

_TOP_LEVEL_PREFIXES: tuple[str, ...] = ("def ", "from ", "import ", "class ", "@")
"""Python prefixes that can start a standalone candidate program."""

_MAX_PREFIX_PARSE_LINES: int = 500
"""Line cap for the quadratic longest-valid-prefix parse search."""

_HUMANEVAL_PROMPT_MODES: tuple[str, ...] = (
    "human_eval",
    "humaneval",
    "human_eval_plus",
    "humaneval_plus",
    "human_eval_prefix",
)
_MBPP_PROMPT_MODES: tuple[str, ...] = (
    "mbpp",
    "mbpp_plus",
    "instruction",
)

# ===========================================================================
# Code extraction
# ===========================================================================


def extract_code(text: object) -> str:
    """Extract runnable Python code from a model-generated string.

    Cascades through three strategies (first match wins):

    1. **Fenced block** — `` ```python ... ``` `` or bare `` ``` ... ``` ``.
    2. **Code-start heuristic** — text from the first ``def`` / ``from`` /
       ``import`` / ``class`` line onward, truncated at the next top-level
       stop marker.
    3. **Raw fallback** — the original text with trailing whitespace removed.

    Returns ``""`` for empty / ``None`` / non-string inputs.

    .. note::
        All strategies use ``.rstrip()`` rather than ``.strip()`` so that
        leading indentation is preserved for HumanEval-style bare function
        bodies (e.g. ``"    return a + b"``).
    """
    if not text or not isinstance(text, str):
        return ""

    # Strategy 1 — fenced block
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).rstrip()

    # Strategy 2 — first code line → longest syntactically-valid Python prefix.
    # This preserves helper functions/imports while dropping trailing prose.
    start_m = _CODE_START_RE.search(text)
    if start_m:
        body = text[start_m.start() :]
        stop_m = _STOP_MARKERS.search(body)
        if stop_m:
            body = body[: stop_m.start()]
        body = _longest_valid_python_prefix(body)
        return body.rstrip()

    # Strategy 3 — raw fallback (preserve leading whitespace for indentation)
    return text.rstrip()


CODE_GENERATION_PIPELINE = TextFilterPipeline(
    "code_generation",
    (("strip_reasoning", strip_reasoning_wrappers), ("extract_code", extract_code)),
)


def _longest_valid_python_prefix(code: str) -> str:
    """Return the longest prefix of *code* that parses as Python."""
    lines = code.rstrip().splitlines()
    # Cap the O(n^2) prefix search: pathologically long outputs (e.g.
    # repetition loops) would otherwise trigger one ast.parse per line.
    if len(lines) > _MAX_PREFIX_PARSE_LINES:
        lines = lines[:_MAX_PREFIX_PARSE_LINES]
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate
    return code.rstrip()


# ===========================================================================
# Pass@k estimation
# ===========================================================================


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    .. math::
        \\text{pass@k} = 1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}}

    Computed in a numerically-stable product form.  When *k* = 1 this
    simplifies to ``num_correct / num_samples``.
    """
    if any(type(value) is not int for value in (num_samples, num_correct, k)):
        raise TypeError("num_samples, num_correct, and k must be integers")
    n, c = num_samples, num_correct
    if n < 1:
        raise ValueError(f"num_samples must be positive, got {n}")
    if c < 0 or c > n:
        raise ValueError(f"num_correct must satisfy 0 <= c <= n, got c={c}, n={n}")
    if k < 1 or k > n:
        raise ValueError(f"pass@k requires 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(n - c + 1, n + 1):
        result *= 1.0 - k / i
    return 1.0 - result


# ===========================================================================
# Result dataclass
# ===========================================================================


@dataclass
class CodeScoreResult:
    """Aggregate code evaluation metrics."""

    pass_at_1: float = 0.0
    """Pass@1 accuracy in [0.0, 1.0]."""

    pass_at_k: dict[str, float] = field(default_factory=dict)
    """Pass@k metrics keyed as ``pass@1``, ``pass@10``, ..."""

    total: int = 0
    """Total number of evaluated samples."""

    correct: int = 0
    """Number of samples that passed all tests."""

    problems: int = 0
    """Number of complete benchmark problems included in Pass@k."""

    records: list[dict[str, Any]] = field(default_factory=list)
    """Per-item execution records (``task_id``, ``passed``, ``result``, ``stderr``)."""


# ===========================================================================
# Internal helpers
# ===========================================================================


def _failure_record(
    task_id: str,
    result: str,
    *,
    evaluation_status: str = "completed",
) -> dict[str, Any]:
    """Return a uniform failure record for a single item.

    ``evaluation_status="completed"`` means the failure was caused by the
    generated answer (an incorrect model observation); ``"failed"`` marks
    scorer/infrastructure failures excluded from Pass@k metrics.
    """
    return {
        "task_id": task_id,
        "passed": False,
        "result": result,
        "evaluation_status": evaluation_status,
        "stderr": "",
    }


def _failure_code_record(item: dict[str, Any]) -> dict[str, Any]:
    """Build a placeholder record for items whose worker raised."""
    task_id = str(item.get("task_id", ""))
    return _failure_record(
        task_id,
        "scoring error",
        evaluation_status="failed",
    )


def _first_meaningful_line(text: str) -> str:
    """Return the first non-empty, non-comment line from ``text``."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return line
    return ""


def _starts_with_top_level_code(text: str) -> bool:
    """Return whether ``text`` appears to start with standalone Python code."""
    first_line = _first_meaningful_line(text)
    return first_line.startswith(_TOP_LEVEL_PREFIXES)


def _build_check_programs(
    prompt: str, code: str, test_code: str, prompt_mode: str | None = None
) -> list[tuple[str, str]]:
    """Build candidate programs to execute for a generated code sample.

    HumanEval prompts are Python prefixes, so body completions need
    ``prompt + code``.  MBPP prompts are natural language instructions, so the
    prompt must not be executed.  If a HumanEval-style model returns a full
    top-level ``def`` instead of only a body, the standalone ``code`` candidate
    is tried as a fallback.
    """
    programs: list[tuple[str, str]] = []
    prompt_mode_norm = prompt_mode.strip().lower() if prompt_mode else ""

    if prompt_mode_norm in _HUMANEVAL_PROMPT_MODES:
        programs.append(("prompt_plus_code", f"{prompt.rstrip()}\n{code}\n{test_code}"))
        if _starts_with_top_level_code(code):
            programs.append(("code_only", f"{code.rstrip()}\n{test_code}"))
    elif prompt_mode_norm in _MBPP_PROMPT_MODES:
        programs.append(("code_only", f"{code.rstrip()}\n{test_code}"))
    elif _starts_with_top_level_code(prompt):
        programs.append(("prompt_plus_code", f"{prompt.rstrip()}\n{code}\n{test_code}"))
        if _starts_with_top_level_code(code):
            programs.append(("code_only", f"{code.rstrip()}\n{test_code}"))
    else:
        programs.append(("code_only", f"{code.rstrip()}\n{test_code}"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for variant, program in programs:
        if program not in seen:
            deduped.append((variant, program))
            seen.add(program)
    return deduped


def _resolve_prompt_mode(item: dict[str, Any]) -> str:
    """Resolve an explicit prompt mode from the prepared record."""
    prompt_mode = item.get("prompt_mode", "")
    return str(prompt_mode).strip().lower() if prompt_mode else ""


# ===========================================================================
# Per-item worker (module-level → picklable by Pebble)
# ===========================================================================


def _process_code_item_impl(
    args: tuple[int, dict[str, Any], str, str, float, bool],
) -> tuple[int, dict[str, Any]]:
    """Score a single code-generation item.

    Parameters
    ----------
    args : tuple
        ``(index, item_dict, label_key, response_key, exec_timeout, allow_unsafe_code)``.

        * **index** — position in the original dataset (for result ordering).
        * **item_dict** — must contain ``"prompt"``, *label_key*, and
          *response_key*.
        * **label_key** — key for the ground-truth test harness.
        * **response_key** — key for the model output (``str`` or ``list[str]``).
        * **exec_timeout** — per-item execution timeout in seconds.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(index, record)`` where *record* has keys ``task_id``, ``passed``,
        ``result``, ``stderr``.
    """
    idx, item, label_key, response_key, exec_timeout, allow_unsafe_code = args

    # -- resolve identifiers ----------------------------------------------------
    task_id: str = str(item["task_id"])
    prompt: str = str(item.get("prompt", ""))
    test_code: str = str(item.get(label_key, ""))
    prompt_mode: str = _resolve_prompt_mode(item)

    # -- extract model output ---------------------------------------------------
    generation = resolve_single_generation(item, response_key)
    gen_str = generation or ""

    code, filter_trace = CODE_GENERATION_PIPELINE.apply_with_trace(gen_str)
    filter_artifacts = {
        "filtered_gen": code,
        "filter_trace": filter_trace,
    }

    # -- guard: missing test harness ---------------------------------------------
    # Without a test harness any executable candidate would appear to pass,
    # so the item cannot be scored.
    if not test_code.strip():
        record = _failure_record(
            task_id,
            f"failed: empty test harness in label field {label_key!r}",
            evaluation_status="failed",
        )
        record.update(filter_artifacts)
        return idx, record

    inference_error = item.get("error")
    if inference_error:
        record = _failure_record(
            task_id,
            f"failed: inference error: {inference_error}",
            evaluation_status="failed",
        )
        record.update(filter_artifacts)
        return idx, record

    if generation is None:
        record = _failure_record(
            task_id,
            "failed: missing or malformed generation",
            evaluation_status="failed",
        )
        record.update(filter_artifacts)
        return idx, record

    if not gen_str.strip():
        record = _failure_record(task_id, "failed: empty generation")
        record.update(filter_artifacts)
        return idx, record

    if not code.strip():
        record = _failure_record(task_id, "failed: no code extracted")
        record.update(filter_artifacts)
        return idx, record

    # -- construct and execute --------------------------------------------------
    # extract_code() preserves leading indentation (uses .rstrip()), so bare
    # HumanEval-style function bodies (``"    return a + b"``) remain valid.
    exec_result: dict[str, Any] | None = None
    for _, check_program in _build_check_programs(
        prompt, code, test_code, prompt_mode=prompt_mode
    ):
        exec_result = check_correctness(
            check_program,
            exec_timeout,
            task_id,
            allow_unsafe_code=allow_unsafe_code,
        )
        if exec_result.get("passed"):
            break
        if exec_result.get("evaluation_status") == "failed":
            break
    if exec_result is None:
        record = _failure_record(task_id, "failed: no executable candidate")
        record.update(filter_artifacts)
        return idx, record
    exec_result.setdefault("task_id", task_id)
    if exec_result.get("evaluation_status") not in {"completed", "failed"}:
        raise ValueError("code execution result is missing a valid evaluation_status")
    exec_result.update(filter_artifacts)
    return idx, exec_result


def _process_code_item(
    args: tuple[int, dict[str, Any], str, str, float, bool],
) -> tuple[int, dict[str, Any]]:
    """Convert every worker exception into an indexed infrastructure failure."""
    idx, item, *_ = args
    try:
        return _process_code_item_impl(args)
    except Exception as exc:
        logger.warning("Code scoring worker failed for item %d: %s", idx, exc)
        return idx, _failure_code_record(item)


# ===========================================================================
# Dispatcher
# ===========================================================================


def _score_items(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    exec_timeout: float,
    max_workers: int,
    timeout: int,
    allow_unsafe_code: bool,
) -> list[dict[str, Any]]:
    """Score every item, preserving input order.

    Parameters
    ----------
    eval_dataset : list[dict]
        Items to score.  Each must contain *label_key* and *response_key*.
    label_key : str
        Dict key for the ground-truth test code.
    response_key : str
        Dict key for the model output (``str`` or ``list[str]``).
    exec_timeout : float
        Per-item code execution timeout in seconds.
    max_workers : int
        Maximum Pebble ``ProcessPool`` workers.  ≤ 1 forces serial execution.
    timeout : int
        Pebble pool-level timeout per worker task in seconds.

    Returns
    -------
    list[dict[str, Any]]
        One scored record per input item, in the same order.  Items whose
        worker raised are represented by :func:`_failure_code_record`; items
        whose result never arrived are represented by
        :func:`_failure_code_record`.
    """
    total = len(eval_dataset)
    if total == 0:
        return []

    # -- serial fast-path -----------------------------------------------------
    if max_workers <= 1 or total == 1:
        records: list[dict[str, Any]] = []
        for i, item in enumerate(eval_dataset):
            try:
                _, rec = _process_code_item(
                    (i, item, label_key, response_key, exec_timeout, allow_unsafe_code),
                )
            except Exception as exc:
                logger.warning("Code scoring failed for item %d: %s", i, exc)
                rec = _failure_code_record(item)
            records.append(rec)
        return records

    # -- parallel path ---------------------------------------------------------
    optimal_workers = resolve_max_workers(total, max_workers)
    results_by_index: dict[int, dict[str, Any]] = {}

    iterable = [
        (i, item, label_key, response_key, exec_timeout, allow_unsafe_code)
        for i, item in enumerate(eval_dataset)
    ]

    worker_timeout = _code_worker_timeout(timeout, exec_timeout)
    if worker_timeout > timeout:
        logger.info(
            "Expanded code worker timeout from %ss to %ss for exec_timeout=%.1fs",
            timeout,
            worker_timeout,
            exec_timeout,
        )

    with (
        tqdm(total=total, desc="Scoring code", unit="item") as pbar,
        ProcessPool(max_workers=optimal_workers) as pool,
    ):
        future = pool.map(_process_code_item, iterable, timeout=worker_timeout)
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
            except Exception as exc:
                logger.warning("Code scoring worker result failed: %s", exc)
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


def _code_worker_timeout(configured_timeout: int, exec_timeout: float) -> int:
    """Cover every nested candidate execution plus coordinator overhead.

    Each ``check_correctness`` call costs at most ``exec_timeout`` (signal
    timeout) + ``PROCESS_JOIN_MARGIN_SECONDS`` (join) + ``PROCESS_KILL_MARGIN``
    (kill + rejoin) when the worker hangs, so the per-item budget must use the
    hung-worker path — not just the signal timeout — for every candidate
    program the worker may try.
    """
    per_check = exec_timeout + PROCESS_JOIN_MARGIN_SECONDS + PROCESS_KILL_MARGIN
    nested_budget = _MAX_CHECK_PROGRAMS * per_check
    return max(
        configured_timeout,
        math.ceil(nested_budget + _POOL_COORDINATOR_MARGIN_SECONDS),
    )


def _code_identity(item: dict[str, Any], row_index: int) -> str:
    """Return the stable code task identity, validating a non-empty task_id."""
    task_id = item.get("task_id")
    if task_id is None or not str(task_id).strip():
        raise ValueError(
            f"Code evaluation record at index {row_index} is missing required "
            "non-empty 'task_id'"
        )
    return str(task_id)


def _normalize_code_samples(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    n_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Validate and normalize one code generation per input row.

    Repeated rows remain independent samples, including rows with identical
    responses. Rows that repeat a ``task_id`` with a conflicting prompt or
    test harness raise ``ValueError``.
    """
    return normalize_single_generation_samples(
        eval_dataset,
        response_key,
        problem_identity=_code_identity,
        conflict_keys=(label_key, "prompt"),
        record_kind="code task",
        n_samples=n_samples,
    )


def _pass_at_k_scores(grouped: dict[str, list[dict[str, Any]]], k: int) -> list[float]:
    """Return one unbiased pass@k estimate per eligible completed group."""
    return [
        estimate_pass_at_k(
            len(records), sum(1 for record in records if record.get("passed")), k
        )
        for records in grouped.values()
        if len(records) >= k
    ]


def _compute_pass_at_k(
    grouped: dict[str, list[dict[str, Any]]], k_values: tuple[int, ...]
) -> dict[str, float]:
    """Aggregate completed problem groups into problem-level pass@k metrics."""
    metrics: dict[str, float] = {}
    for k in sorted(set(k_values)):
        scores = _pass_at_k_scores(grouped, k)
        if scores:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)
    return metrics


def _completed_code_groups(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return problem groups whose samples all completed successfully.

    Pass@k is a problem-level estimate. Keeping the successful subset of an
    incomplete problem can bias that estimate, so any scorer, worker, or
    inference failure excludes the whole problem from problem-level metrics.
    The individual failures remain in ``records`` and in the sample counters.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task_id = str(record["task_id"])
        grouped.setdefault(task_id, []).append(record)
    return {
        task_id: problem_records
        for task_id, problem_records in grouped.items()
        if all(
            record.get("evaluation_status", "completed") == "completed"
            for record in problem_records
        )
    }


# ===========================================================================
# Public API
# ===========================================================================


def _score_code_task_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    max_workers: int = 8,
    timeout: int = 20,
    exec_timeout: float = _DEFAULT_EXEC_TIMEOUT,
    k_values: tuple[int, ...] = (1, 10, 64),
    allow_unsafe_code: bool = False,
    n_samples: int | None = None,
) -> tuple[CodeScoreResult, dict[str, list[dict[str, Any]]]]:
    """Score a code-generation dataset and return task-native details.

    Returns the aggregate result together with the completed problem groups,
    so metric and observation aggregation share one grouping pass.

    Parameters
    ----------
    eval_dataset : list[dict]
        Items to score.  Each must contain *label_key* (test harness) and
        *response_key* (model output). Each row must contain exactly one
        generation; repeated rows for one ``task_id`` contribute to pass@k.
    label_key : str
        Dict key for the ground-truth test harness (e.g. ``"answer"``).
    response_key : str
        Dict key for the model output (e.g. ``"gen"``).
    max_workers : int
        Maximum Pebble ``ProcessPool`` workers (≤ 1 = serial).
    timeout : int
        Pool-level timeout per worker task in seconds.
    exec_timeout : float
        Per-item code execution timeout in seconds (default 3.0).
    k_values : tuple[int, ...]
        pass@k values to include in the summary when enough samples exist.
    allow_unsafe_code : bool
        Explicitly allow execution of generated code.  The CLI defaults to
        ``False`` and should only enable this in a trusted or isolated runtime.

    """
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers!r}")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    if (
        not isinstance(exec_timeout, int | float)
        or isinstance(exec_timeout, bool)
        or not math.isfinite(exec_timeout)
        or exec_timeout <= 0
    ):
        raise ValueError(f"exec_timeout must be positive, got {exec_timeout!r}")
    # ``k_values`` is a scorer-level option rather than an inference config
    # field, so validate it at the public scoring boundary.
    if (
        not isinstance(k_values, tuple)
        or not k_values
        or any(type(k) is not int or k <= 0 for k in k_values)
    ):
        raise ValueError(
            f"k_values must contain only positive integers, got {k_values}"
        )
    if not eval_dataset:
        logger.warning("Empty dataset — returning 0.0")
        return CodeScoreResult(), {}
    if not allow_unsafe_code:
        raise PermissionError(
            "Code evaluation executes generated code. Pass "
            "allow_unsafe_code=True (or the CLI --allow_unsafe_code flag) "
            "only when the execution environment is trusted."
        )
    expanded_dataset = _normalize_code_samples(
        eval_dataset,
        label_key,
        response_key,
        n_samples=n_samples,
    )
    total = len(expanded_dataset)

    logger.info(
        "Scoring %d sample(s) from %d record(s) with max_workers=%d, exec_timeout=%.1fs",
        total,
        len(eval_dataset),
        max_workers,
        exec_timeout,
    )

    records = _score_items(
        expanded_dataset,
        label_key,
        response_key,
        exec_timeout,
        max_workers,
        timeout,
        allow_unsafe_code,
    )

    correct = sum(1 for r in records if r.get("passed"))
    completed_groups = _completed_code_groups(records)
    pass_at_k = _compute_pass_at_k(completed_groups, k_values)
    problems = len(completed_groups)
    # pass@1 is always computable (every completed group has n >= 1), so
    # derive it independently of ``k_values`` rather than defaulting to 0.0
    # when the caller did not request k=1.
    pass_at_1 = pass_at_k.get("pass@1")
    if pass_at_1 is None:
        pass_at_1 = _compute_pass_at_k(completed_groups, (1,)).get("pass@1", 0.0)

    # Log failures for diagnostics (up to 10).
    failures = [r for r in records if not r.get("passed")]
    if failures:
        logger.info(
            "%d failure(s): %s",
            len(failures),
            ", ".join(r.get("result", "?") for r in failures[:10]),
        )

    result = CodeScoreResult(
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        total=total,
        correct=correct,
        problems=problems,
        records=records,
    )
    logger.info(
        "Pass@1 (problem macro): %.2f%% across %d complete problem(s); "
        "sample outcomes: %d/%d passed",
        pass_at_1 * 100,
        problems,
        correct,
        total,
    )
    return result, completed_groups


def _code_observations(
    result: CodeScoreResult, grouped: dict[str, list[dict[str, Any]]]
) -> dict[str, list[float]]:
    """Return problem-level observations for every available pass@k metric."""
    observations: dict[str, list[float]] = {}
    # pass@1 is always observable (n >= 1 for every group), even when the
    # caller's ``k_values`` did not include it.
    for name in dict.fromkeys(["pass@1", *result.pass_at_k]):
        try:
            k = int(name.removeprefix("pass@"))
        except ValueError:
            continue
        observations[name] = _pass_at_k_scores(grouped, k)
    return observations


def score_code_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    max_workers: int = 8,
    timeout: int = 20,
    exec_timeout: float = _DEFAULT_EXEC_TIMEOUT,
    k_values: tuple[int, ...] = (1, 10, 64),
    allow_unsafe_code: bool = False,
    n_samples: int | None = None,
) -> ScorerResult:
    """Score code and return the registry's structured scorer contract."""
    result, completed_groups = _score_code_task_result(
        eval_dataset,
        label_key,
        response_key,
        max_workers=max_workers,
        timeout=timeout,
        exec_timeout=exec_timeout,
        k_values=k_values,
        allow_unsafe_code=allow_unsafe_code,
        n_samples=n_samples,
    )
    metrics = dict(result.pass_at_k)
    metrics.setdefault("pass@1", result.pass_at_1)
    observations = _code_observations(result, completed_groups)
    observations.setdefault("pass@1", [])
    # Every record from _score_items carries its evaluation_status (set by
    # _process_code_item / the failure builders), so classify from the stored
    # field rather than re-deriving it from the result text.
    statuses = [
        record.get("evaluation_status", "completed") for record in result.records
    ]
    failed_count = statuses.count("failed")
    all_problem_ids = list(
        dict.fromkeys(str(record["task_id"]) for record in result.records)
    )
    excluded_problem_ids = [
        task_id for task_id in all_problem_ids if task_id not in completed_groups
    ]
    return ScorerResult(
        metrics=metrics,
        observations=observations,
        records=result.records,
        details={
            "complete_problem_count": len(completed_groups),
            "incomplete_problem_count": len(excluded_problem_ids),
            "excluded_problem_task_ids": excluded_problem_ids,
        },
        sample_count=result.total,
        effective_sample_count=result.total - failed_count,
        failed_count=failed_count,
    )
