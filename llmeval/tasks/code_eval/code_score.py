"""Scoring driver for code-generation evaluation (HumanEval / MBPP / Pass@k).

Architecture matches ``mc_score.py``:
    * Module-level picklable worker (``_process_code_item``)
    * Serial / parallel dispatcher (``_score_items``)
    * Aggregate result dataclass (``CodeScoreResult``)
    * JSONL + summary result persistence (``write_cache``)
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pebble import ProcessPool
from tqdm import tqdm

from llmeval.tasks.code_eval.execute import check_correctness
from llmeval.tasks.persistence import atomic_write_json, atomic_write_jsonl
from llmeval.tasks.postprocess import (
    FilterRegistry,
    TextFilterPipeline,
    strip_reasoning_wrappers,
)
from llmeval.tasks.results import ScorerResult
from llmeval.tasks.sample_index import duplicate_sample_error, resolve_sample_indices
from llmeval.utils.log import init_logger

logger = init_logger("code_score")

__all__ = [
    "CodeScoreResult",
    "estimate_pass_at_k",
    "extract_code",
    "score_code",
    "score_code_result",
    "write_cache",
]

# ===========================================================================
# Constants
# ===========================================================================

_DEFAULT_EXEC_TIMEOUT: float = 3.0
"""Default per-item code execution timeout in seconds."""

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

CODE_FILTER_REGISTRY = FilterRegistry()
CODE_FILTER_REGISTRY.register("strip_reasoning", strip_reasoning_wrappers, version="1")
CODE_GENERATION_PIPELINE: TextFilterPipeline


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


def _extract_code_filter(text: str) -> str:
    """Text-filter adapter for :func:`extract_code`."""
    return extract_code(text)


CODE_FILTER_REGISTRY.register("extract_code", _extract_code_filter, version="1")
CODE_GENERATION_PIPELINE = CODE_FILTER_REGISTRY.build_pipeline(
    "code_generation", "1", "strip_reasoning", "extract_code"
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
    n, c = int(num_samples), int(num_correct)
    if n < 1:
        return 0.0
    if k < 1 or k > n:
        raise ValueError(f"pass@k requires 1 <= k <= n, got k={k}, n={n}")
    if c < 0:
        c = 0
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
    """Number of distinct benchmark problems."""

    per_item: list[dict[str, Any]] = field(default_factory=list)
    """Per-item execution records (``task_id``, ``passed``, ``result``, ``stderr``)."""


# ===========================================================================
# Internal helpers
# ===========================================================================


def _failure_record(
    task_id: str,
    result: str,
    *,
    group_id: str | None = None,
    sample_index: int = 0,
    evaluation_status: str = "completed",
) -> dict[str, Any]:
    """Return a uniform failure record for a single item.

    ``evaluation_status="completed"`` means the failure was caused by the
    generated answer (an incorrect model observation); ``"failed"`` marks
    scorer/infrastructure failures excluded from Pass@k metrics.
    """
    return {
        "task_id": task_id,
        "group_id": group_id or task_id,
        "sample_index": sample_index,
        "passed": False,
        "result": result,
        "evaluation_status": evaluation_status,
        "stderr": "",
    }


def _failure(
    task_id: str,
    reason: str,
    group_id: str | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Return a failure record where the generated answer caused the failure."""
    return _failure_record(
        task_id,
        reason,
        group_id=group_id,
        sample_index=sample_index,
        evaluation_status="completed",
    )


def _failure_code_record(item: dict[str, Any]) -> dict[str, Any]:
    """Build a placeholder record for items that could not be scored."""
    task_id = str(item.get("task_id", ""))
    return _failure_record(
        task_id,
        "scoring error",
        group_id=task_id,
        sample_index=item.get("sample_index", 0),
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


def _process_code_item(
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
    group_id: str = str(item.get("task_id", f"task_{idx}"))
    sample_index: int = int(item.get("sample_index", 0))
    task_id: str = str(item.get("task_id") or group_id)
    prompt: str = str(item.get("prompt", ""))
    test_code: str = str(item.get(label_key, ""))
    prompt_mode: str = _resolve_prompt_mode(item)

    # -- extract model output ---------------------------------------------------
    gen_raw = item.get(response_key)
    if isinstance(gen_raw, list):
        gen_str: str = str(gen_raw[0]) if gen_raw else ""
    elif isinstance(gen_raw, str):
        gen_str = gen_raw
    else:
        gen_str = ""

    code, filter_trace = CODE_GENERATION_PIPELINE.apply_with_trace(gen_str)
    filter_artifacts = {
        "raw_gen": gen_str,
        "filtered_gen": code,
        "filter_trace": filter_trace,
    }

    # -- guard: missing test harness ---------------------------------------------
    # An empty test harness is a dataset problem, not a model failure: without
    # it ANY executable candidate would be scored "passed". Skip the item so
    # it stays out of the Pass@k denominator while retaining debug artifacts.
    if not test_code.strip():
        record = {
            "task_id": task_id,
            "group_id": group_id,
            "sample_index": sample_index,
            "passed": False,
            "result": f"skipped: empty test harness in label field {label_key!r}",
            "evaluation_status": "skipped",
            "stderr": "",
        }
        record.update(filter_artifacts)
        return idx, record

    if not gen_str.strip():
        record = _failure(task_id, "failed: empty generation", group_id, sample_index)
        record.update(filter_artifacts)
        return idx, record

    if not code.strip():
        record = _failure(task_id, "failed: no code extracted", group_id, sample_index)
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
    if exec_result is None:
        record = _failure(
            task_id, "failed: no executable candidate", group_id, sample_index
        )
        record.update(filter_artifacts)
        return idx, record
    exec_result.setdefault("task_id", task_id)
    exec_result.setdefault("group_id", group_id)
    exec_result.setdefault("sample_index", sample_index)
    exec_result["evaluation_status"] = _code_record_status(exec_result)
    exec_result.update(filter_artifacts)
    return idx, exec_result


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
        scoring failed are represented by :func:`_failure_code_record`.
    """
    total = len(eval_dataset)
    if total == 0:
        return []

    # -- serial fast-path -----------------------------------------------------
    if max_workers <= 1 or total == 1:
        records: list[dict[str, Any]] = []
        for i, item in enumerate(eval_dataset):
            _, rec = _process_code_item(
                (i, item, label_key, response_key, exec_timeout, allow_unsafe_code),
            )
            records.append(rec)
        return records

    # -- parallel path ---------------------------------------------------------
    cpu_count = os.cpu_count() or 1
    optimal_workers = min(total, max_workers, max(1, cpu_count - 1))
    results_by_index: dict[int, dict[str, Any]] = {}

    iterable = [
        (i, item, label_key, response_key, exec_timeout, allow_unsafe_code)
        for i, item in enumerate(eval_dataset)
    ]

    # NOTE: each pool worker calls ``check_correctness`` once per sample, and
    # that child process defaults to the ``fork`` start method (see
    # ``execute._resolve_mp_method``) — ``spawn``'s interpreter restart per
    # sample used to consume the pool-level ``timeout`` budget below and
    # caused spurious scoring timeouts.  Set ``LLMEVAL_MP_METHOD=spawn`` to
    # opt back in if fork-unsafe threads are ever introduced.
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


def _stable_problem_id(item: dict[str, Any], index: int) -> str:
    """Build a stable grouping id for pass@k aggregation."""
    if item.get("task_id") is not None:
        return str(item["task_id"])
    if item.get("prompt") is not None:
        digest = hashlib.sha1(
            str(item["prompt"]).encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        return f"prompt:{digest}"
    return f"task_{index}"


def _expand_code_samples(
    eval_dataset: list[dict[str, Any]], response_key: str
) -> list[dict[str, Any]]:
    """Expand each record into one scoring job per generated sample.

    Explicit ``sample_index``/``sample_indices`` fields are preserved per the
    shared protocol (invalid fields raise); only rows with no index fields at
    all receive the next unused per-problem indices.  Duplicate
    ``(problem, sample_index)`` pairs merge idempotently when the generation
    content matches and raise when it conflicts.
    """
    expanded: list[dict[str, Any]] = []
    used_by_problem: dict[str, set[int]] = {}
    seen_samples: dict[tuple[str, int], Any] = {}
    for item_idx, item in enumerate(eval_dataset):
        group_id = _stable_problem_id(item, item_idx)
        gen_raw = item.get(response_key)
        if isinstance(gen_raw, list):
            samples = gen_raw if gen_raw else [""]
        elif isinstance(gen_raw, str):
            samples = [gen_raw]
        else:
            samples = [""]

        used = used_by_problem.setdefault(group_id, set())
        if (
            isinstance(gen_raw, list)
            and not gen_raw
            and item.get("sample_indices") == []
            and "sample_index" not in item
        ):
            # Preserve an explicit zero-generation row as one unindexed
            # failure observation. The scorer can then report the inference
            # failure instead of aborting on a synthetic length mismatch.
            sample_item = item.copy()
            sample_item.pop("sample_indices", None)
            sample_item[response_key] = []
            sample_item.pop("sample_index", None)
            expanded.append(sample_item)
            continue
        sample_indices = resolve_sample_indices(
            item, len(samples), problem_id=group_id, used_indices=used
        )
        used.update(sample_indices)
        for sample_index, sample in zip(sample_indices, samples, strict=True):
            seen_key = (group_id, sample_index)
            if seen_key in seen_samples:
                if seen_samples[seen_key] != sample:
                    raise duplicate_sample_error(group_id, sample_index)
                continue  # idempotent duplicate — already scheduled
            seen_samples[seen_key] = sample
            sample_item = item.copy()
            sample_item.pop("sample_indices", None)
            sample_item[response_key] = [sample]
            sample_item["sample_index"] = sample_index
            expanded.append(sample_item)
    return expanded


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
    records: list[dict[str, Any]], k_values: tuple[int, ...]
) -> tuple[dict[str, float], int]:
    """Aggregate sample records into problem-level pass@k metrics."""
    grouped = _completed_code_groups(records)

    metrics: dict[str, float] = {}
    for k in sorted(set(k_values)):
        scores = _pass_at_k_scores(grouped, k)
        if scores:
            metrics[f"pass@{k}"] = sum(scores) / len(scores)

    return metrics, len(grouped)


def _completed_code_groups(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group completed code observations by their stable problem identity."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("evaluation_status", "completed") != "completed":
            continue
        group_id = str(record.get("group_id") or record.get("task_id") or "")
        grouped.setdefault(group_id, []).append(record)
    return grouped


# ===========================================================================
# Public API
# ===========================================================================


def _score_code_task_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 20,
    exec_timeout: float = _DEFAULT_EXEC_TIMEOUT,
    k_values: tuple[int, ...] = (1, 10, 64),
    allow_unsafe_code: bool = False,
    persist_legacy: bool = True,
) -> CodeScoreResult:
    """Score a code-generation dataset and return task-native details.

    Parameters
    ----------
    eval_dataset : list[dict]
        Items to score.  Each must contain *label_key* (test harness) and
        *response_key* (model output).  List-valued generations are expanded
        so every sample contributes to pass@k.
    label_key : str
        Dict key for the ground-truth test harness (e.g. ``"answer"``).
    response_key : str
        Dict key for the model output (e.g. ``"gen"``).
    cache_path : str | Path
        Path for the per-item JSONL result file. A ``.summary.json`` is written
        alongside it.
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
    if not eval_dataset:
        logger.warning("Empty dataset — returning 0.0")
        return CodeScoreResult()
    if not allow_unsafe_code:
        raise PermissionError(
            "Code evaluation executes generated code. Pass "
            "allow_unsafe_code=True (or the CLI --allow_unsafe_code flag) "
            "only when the execution environment is trusted."
        )
    # ``k_values`` is a scorer-level option rather than an inference config
    # field, so validate it at the public scoring boundary.
    if any(k <= 0 for k in k_values):
        raise ValueError(
            f"k_values must contain only positive integers, got {k_values}"
        )
    expanded_dataset = _expand_code_samples(eval_dataset, response_key)
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
    pass_at_k, problems = _compute_pass_at_k(records, k_values)
    # pass@1 is always computable (every completed group has n >= 1), so
    # derive it independently of ``k_values`` rather than defaulting to 0.0
    # when the caller did not request k=1.
    pass_at_1 = pass_at_k.get("pass@1")
    if pass_at_1 is None:
        pass_at_1 = _compute_pass_at_k(records, (1,))[0].get("pass@1", 0.0)

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
        per_item=records,
    )
    if persist_legacy:
        write_cache(result, cache_path)

    logger.info(
        "Pass@1: %.2f%% (%d/%d correct samples, %d problem(s))",
        pass_at_1 * 100,
        correct,
        total,
        problems,
    )
    return result


def _code_observations(result: CodeScoreResult) -> dict[str, list[float]]:
    """Return problem-level observations for every available pass@k metric."""
    grouped = _completed_code_groups(result.per_item)

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


def _is_code_infrastructure_failure(record: dict[str, Any]) -> bool:
    """Distinguish scorer/worker failures from ordinary incorrect programs."""
    result = str(record.get("result", "")).lower()
    # Note: "failed: killed by signal N" (e.g. RLIMIT_CPU/OOM killing a worker
    # running an infinite loop) is deliberately absent — the candidate code
    # caused it, so it counts as a completed (incorrect) model observation.
    return result in {
        "scoring error",
        "failed: could not start worker process",
        "failed: segmentationfault",
        "failed: worker did not produce a result",
    }


def _code_record_status(record: dict[str, Any]) -> str:
    """Classify execution outcomes for denominator accounting.

    Assertion failures, syntax errors, signal kills, and candidate-level
    timeouts are completed model observations: the candidate code ran and
    failed on its own.  Only worker/scorer failures — including a hung
    worker that had to be killed — are excluded from Pass@k metrics.
    """
    explicit_status = record.get("evaluation_status")
    if explicit_status in {"completed", "failed", "skipped", "timeout"}:
        return str(explicit_status)
    result = str(record.get("result", "")).lower()
    # "timed out" comes from the worker's result file: the candidate code
    # itself exceeded exec_timeout (e.g. an infinite loop) — an incorrect
    # model observation.  "timed out: worker killed" means the worker hung
    # past the inner timeout and had to be killed; that is an execution
    # anomaly, not evidence about the model.
    if result == "timed out: worker killed":
        return "timeout"
    if _is_code_infrastructure_failure(record):
        return "failed"
    return "completed"


def score_code_result(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 20,
    exec_timeout: float = _DEFAULT_EXEC_TIMEOUT,
    k_values: tuple[int, ...] = (1, 10, 64),
    allow_unsafe_code: bool = False,
    persist_legacy: bool = True,
) -> ScorerResult:
    """Score code and return the registry's structured scorer contract."""
    result = _score_code_task_result(
        eval_dataset,
        label_key,
        response_key,
        cache_path,
        max_workers=max_workers,
        timeout=timeout,
        exec_timeout=exec_timeout,
        k_values=k_values,
        allow_unsafe_code=allow_unsafe_code,
        persist_legacy=persist_legacy,
    )
    metrics = dict(result.pass_at_k)
    metrics.setdefault("pass@1", result.pass_at_1)
    observations = _code_observations(result)
    observations.setdefault("pass@1", [])
    # Every record from _score_items carries its evaluation_status (set by
    # _process_code_item / the failure builders), so classify from the stored
    # field rather than re-deriving it from the result text.
    statuses = [
        record.get("evaluation_status", "completed") for record in result.per_item
    ]
    timeout_count = statuses.count("timeout")
    failed_count = statuses.count("failed")
    skipped_count = statuses.count("skipped")
    return ScorerResult(
        metrics=metrics,
        observations=observations,
        per_item=result.per_item,
        sample_count=result.total,
        effective_sample_count=max(
            result.total - failed_count - skipped_count - timeout_count, 0
        ),
        failed_count=failed_count,
        skipped_count=skipped_count,
        timeout_count=timeout_count,
    )


def score_code(
    eval_dataset: list[dict[str, Any]],
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int = 8,
    timeout: int = 20,
    exec_timeout: float = _DEFAULT_EXEC_TIMEOUT,
    k_values: tuple[int, ...] = (1, 10, 64),
    allow_unsafe_code: bool = False,
) -> float:
    """Compatibility wrapper returning only the primary Pass@1 metric."""
    return score_code_result(
        eval_dataset,
        label_key,
        response_key,
        cache_path,
        max_workers=max_workers,
        timeout=timeout,
        exec_timeout=exec_timeout,
        k_values=k_values,
        allow_unsafe_code=allow_unsafe_code,
    ).metrics["pass@1"]


# ===========================================================================
# Result persistence
# ===========================================================================


def write_cache(result: CodeScoreResult, cache_path: str | Path) -> None:
    """Write per-item JSONL and a ``.summary.json`` metrics file.

    Pattern matches :func:`llmeval.tasks.mc_eval.mc_score.write_cache`.
    """
    cache_path = Path(cache_path)
    atomic_write_jsonl(cache_path, result.per_item)

    summary_path = cache_path.with_suffix(".summary.json")
    atomic_write_json(
        summary_path,
        {
            "pass_at_1": round(result.pass_at_1, 6),
            "pass_at_k": {
                key: round(value, 6) for key, value in result.pass_at_k.items()
            },
            "total": result.total,
            "correct": result.correct,
            "problems": result.problems,
        },
        indent=2,
    )
