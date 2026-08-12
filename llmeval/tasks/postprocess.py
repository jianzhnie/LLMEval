"""Shared text filtering, sample normalization, and atomic persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from typing import Any

__all__ = [
    "TextFilterPipeline",
    "atomic_write_json",
    "normalize_single_generation_samples",
    "resolve_max_workers",
    "resolve_single_generation",
    "sample_order_indices",
    "strip_reasoning_wrappers",
]

TextFilter = Callable[[str], str]

_ANSWER_TAG_RE: re.Pattern[str] = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_THINK_END_RE: re.Pattern[str] = re.compile(r"</think\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class TextFilterPipeline:
    """An ordered task-level text pipeline with a stable name."""

    name: str
    filters: tuple[tuple[str, TextFilter], ...]

    def apply_with_trace(self, value: Any) -> tuple[str, dict[str, Any]]:
        """Apply filters and return compact, JSON-compatible metadata."""
        raw_text = _coerce_text(value)
        text = raw_text
        steps: list[dict[str, str | int | bool]] = []
        for filter_name, filter_fn in self.filters:
            input_text = text
            text = _coerce_text(filter_fn(text))
            steps.append(
                {
                    "name": filter_name,
                    "changed": input_text != text,
                    "input_length": len(input_text),
                    "output_length": len(text),
                }
            )
        return text, {
            "pipeline": self.name,
            "filters": steps,
            "input_length": len(raw_text),
            "output_length": len(text),
        }


def _coerce_text(value: Any) -> str:
    """Convert arbitrary values to a string suitable for text filtering."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def strip_reasoning_wrappers(text: str) -> str:
    """Return the model's final tagged answer or post-reasoning text."""
    if not text:
        return ""

    for answer_match in reversed(list(_ANSWER_TAG_RE.finditer(text))):
        extracted = answer_match.group(1).strip()
        if extracted:
            return extracted

    think_matches = list(_THINK_END_RE.finditer(text))
    if think_matches:
        tail = text[think_matches[-1].end() :].strip()
        if tail:
            return tail

    return text


def resolve_single_generation(item: dict[str, Any], response_key: str) -> str | None:
    """Return one valid generation, or ``None`` for malformed/missing data."""
    if response_key not in item:
        return None
    value = item[response_key]
    if isinstance(value, list):
        # An explicit empty list represents one empty model answer.
        if not value:
            return ""
        if len(value) != 1 or not isinstance(value[0], str):
            return None
        return value[0]
    return value if isinstance(value, str) else None


def sample_order_indices(
    items: list[dict[str, Any]],
    *,
    problem_id: str,
    n_samples: int | None = None,
) -> list[int]:
    """Return stable sample positions ordered by validated ``sample_index``.

    Legacy rows without an index receive the lowest unused indices in file
    order. Explicit indices must be non-negative integers and unique within
    the problem.
    """
    if n_samples is not None and (type(n_samples) is not int or n_samples <= 0):
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    row_sample_counts = [item["n_samples"] for item in items if "n_samples" in item]
    if any(type(value) is not int or value <= 0 for value in row_sample_counts):
        raise ValueError(f"Invalid n_samples for problem {problem_id!r}")
    unique_sample_counts = set(row_sample_counts)
    if len(unique_sample_counts) > 1:
        raise ValueError(f"Conflicting n_samples for problem {problem_id!r}")
    if unique_sample_counts:
        row_n_samples = next(iter(unique_sample_counts))
        if n_samples is not None and row_n_samples != n_samples:
            raise ValueError(
                f"Problem {problem_id!r} records n_samples={row_n_samples}, "
                f"but n_samples={n_samples} was requested"
            )
        n_samples = row_n_samples

    explicit: dict[int, int] = {}
    used: dict[int, int] = {}
    for position, item in enumerate(items):
        sample_index = item.get("sample_index")
        if sample_index is None:
            continue
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError(
                f"Invalid sample_index {sample_index!r} for problem {problem_id!r}"
            )
        previous_position = used.get(sample_index)
        if previous_position is not None:
            raise ValueError(
                f"Duplicate sample_index {sample_index} for problem {problem_id!r}"
            )
        explicit[position] = sample_index
        used[sample_index] = position

    assigned: list[int] = []
    next_index = 0
    for position in range(len(items)):
        sample_index = explicit.get(position)
        if sample_index is None:
            while next_index in used:
                next_index += 1
            sample_index = next_index
            used[sample_index] = position
            next_index += 1
        assigned.append(sample_index)
    expected = len(items) if n_samples is None else n_samples
    actual_indices = set(assigned)
    required_indices = set(range(expected))
    if actual_indices != required_indices:
        missing = sorted(required_indices - actual_indices)
        unexpected = sorted(actual_indices - required_indices)
        raise ValueError(
            f"Incomplete samples for problem {problem_id!r}: expected indices "
            f"0..{expected - 1}, missing={missing}, unexpected={unexpected}"
        )
    return sorted(range(len(items)), key=assigned.__getitem__)


def normalize_single_generation_samples(
    eval_dataset: list[dict[str, Any]],
    response_key: str,
    *,
    problem_identity: Callable[[dict[str, Any], int], str],
    conflict_keys: tuple[str, ...] = (),
    record_kind: str = "document",
    n_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Validate repeated samples and normalize one generation per input row.

    Rows sharing one problem identity remain independent even when their
    responses are identical. Problem-level fields listed in *conflict_keys*
    must agree across those rows.
    """
    first_seen: dict[str, dict[str, Any]] = {}
    positions_by_problem: dict[str, list[int]] = {}
    normalized: list[dict[str, Any]] = []
    for row_index, item in enumerate(eval_dataset):
        problem_id = problem_identity(item, row_index)
        first = first_seen.get(problem_id)
        if first is None:
            first_seen[problem_id] = item
        else:
            for key in conflict_keys:
                if item.get(key) != first.get(key):
                    raise ValueError(
                        f"Conflicting {key!r} for {record_kind} {problem_id!r}"
                    )
        response = resolve_single_generation(item, response_key)
        record = dict(item)
        if response is None:
            record.pop(response_key, None)
        else:
            record[response_key] = [response]
        positions_by_problem.setdefault(problem_id, []).append(len(normalized))
        normalized.append(record)

    ordered = normalized.copy()
    for problem_id, positions in positions_by_problem.items():
        samples = [normalized[position] for position in positions]
        sample_order = sample_order_indices(
            samples,
            problem_id=problem_id,
            n_samples=n_samples,
        )
        for destination, source in zip(positions, sample_order, strict=True):
            ordered[destination] = samples[source]
    return ordered


def resolve_max_workers(total: int, requested: int) -> int:
    """Clamp process workers to the workload, request, and available CPUs."""
    if total < 1:
        raise ValueError("total must be positive")
    if requested < 1:
        raise ValueError("requested workers must be positive")
    cpu_count = os.cpu_count() or 1
    return min(total, requested, max(1, cpu_count - 1))


@contextmanager
def _atomic_text_writer(path: str | Path) -> Generator[TextIOBase, None, None]:
    """Yield a sibling temporary file and atomically publish it on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        # A failed write must never leave a partial file behind: the
        # destination keeps its previous (complete) content, and the
        # sibling temporary is removed so a later run starts clean.
        # BaseExceptions (KeyboardInterrupt/SystemExit) propagate untouched,
        # intentionally leaving the uniquely-named temp in place for recovery.
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: str | Path, value: Any, *, indent: int | None = None
) -> None:
    """Serialize JSON and persist it atomically."""
    content = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False)
    if indent is not None:
        content += "\n"
    with _atomic_text_writer(path) as handle:
        handle.write(content)
