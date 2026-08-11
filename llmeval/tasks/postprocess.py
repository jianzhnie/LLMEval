"""Shared text filtering, sample normalization, and atomic persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_FILTER_REGISTRY",
    "FilterRegistry",
    "TextFilter",
    "TextFilterPipeline",
    "atomic_write_json",
    "atomic_write_jsonl",
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
class RegisteredTextFilter:
    """A named, versioned text transformation."""

    name: str
    version: str
    function: TextFilter

    def __call__(self, text: str) -> str:
        return self.function(text)


class FilterRegistry:
    """Register named filters and build deterministic task pipelines."""

    def __init__(self) -> None:
        self._filters: dict[str, RegisteredTextFilter] = {}

    def register(self, name: str, function: TextFilter, *, version: str = "1") -> None:
        if not name:
            raise ValueError("filter name cannot be empty")
        if name in self._filters:
            raise ValueError(f"filter {name!r} is already registered")
        self._filters[name] = RegisteredTextFilter(name, version, function)

    def build_pipeline(
        self, name: str, version: str, *filter_names: str
    ) -> TextFilterPipeline:
        """Build a named, versioned pipeline from registered filters."""
        if not name:
            raise ValueError("pipeline name cannot be empty")
        if not version:
            raise ValueError("pipeline version cannot be empty")
        missing = [item for item in filter_names if item not in self._filters]
        if missing:
            available = ", ".join(sorted(self._filters)) or "<none>"
            raise ValueError(
                f"Unknown text filter(s) {missing!r}; registered filters: {available}"
            )
        return TextFilterPipeline(
            name,
            version,
            tuple(self._filters[item] for item in filter_names),
        )


@dataclass(frozen=True)
class TextFilterPipeline:
    """An ordered task-level text pipeline with a stable name and version."""

    name: str
    version: str
    filters: tuple[RegisteredTextFilter, ...]

    def apply_with_trace(self, value: Any) -> tuple[str, dict[str, Any]]:
        """Apply filters and return compact, JSON-compatible metadata."""
        raw_text = _coerce_text(value)
        text = raw_text
        steps: list[dict[str, str | int | bool]] = []
        for filter_fn in self.filters:
            input_text = text
            text = _coerce_text(filter_fn(text))
            steps.append(
                {
                    "name": filter_fn.name,
                    "version": filter_fn.version,
                    "changed": input_text != text,
                    "input_length": len(input_text),
                    "output_length": len(text),
                }
            )
        return text, {
            "pipeline": self.name,
            "pipeline_version": self.version,
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
    """Remove common reasoning wrappers while preserving raw text fallback."""
    if not text:
        return ""

    answer_match = _ANSWER_TAG_RE.search(text)
    if answer_match:
        extracted = answer_match.group(1).strip()
        if extracted:
            return extracted
        # Empty <answer></answer> tags carry no signal — fall through to the
        # think/raw-text fallbacks so a real answer outside the tag is kept.

    think_match = _THINK_END_RE.search(text)
    if think_match:
        tail = text[think_match.end() :].strip()
        if tail:
            return tail

    return text


DEFAULT_FILTER_REGISTRY = FilterRegistry()
DEFAULT_FILTER_REGISTRY.register(
    "strip_reasoning", strip_reasoning_wrappers, version="1"
)


def resolve_single_generation(
    item: dict[str, Any], response_key: str, *, problem_id: str
) -> str | None:
    """Return one generation, using ``None`` for a failed empty response."""
    value = item.get(response_key)
    if isinstance(value, list):
        if len(value) > 1:
            raise ValueError(
                f"Invalid {response_key!r} for problem {problem_id!r}: expected "
                "one generation per row"
            )
        value = value[0] if value else None
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"Invalid {response_key!r} for problem {problem_id!r}: expected a "
            "string or a list containing one string"
        )
    return value


def sample_order_indices(items: list[dict[str, Any]], *, problem_id: str) -> list[int]:
    """Return stable sample positions ordered by validated ``sample_index``.

    Legacy rows without an index receive the lowest unused indices in file
    order. Explicit indices must be non-negative integers and unique within
    the problem.
    """
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
    return sorted(range(len(items)), key=assigned.__getitem__)


def normalize_single_generation_samples(
    eval_dataset: list[dict[str, Any]],
    response_key: str,
    *,
    problem_identity: Callable[[dict[str, Any], int], str],
    conflict_keys: tuple[str, ...] = (),
    record_kind: str = "document",
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
        response = resolve_single_generation(item, response_key, problem_id=problem_id)
        record = dict(item)
        record[response_key] = [response] if response is not None else []
        positions_by_problem.setdefault(problem_id, []).append(len(normalized))
        normalized.append(record)

    ordered = normalized.copy()
    for problem_id, positions in positions_by_problem.items():
        samples = [normalized[position] for position in positions]
        sample_order = sample_order_indices(samples, problem_id=problem_id)
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


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Stream objects to an atomically replaced JSONL file."""
    with _atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
