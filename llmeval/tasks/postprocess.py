"""Shared input-record processing and atomic persistence helpers.

The task scorers use these helpers explicitly so the response-cleanup chain is
easy to inspect and test.  The helpers are intentionally small and composable:

* ``strip_reasoning_wrappers`` removes common ``<think>`` / ``<answer>``
  wrappers used by reasoning models.
* ``resolve_single_generation`` validates the one-generation-per-row protocol
  and extracts the single generation from an inference output record.
* ``normalize_single_generation_samples`` validates repeated sample rows while
  preserving every independently generated response.
* ``resolve_max_workers`` clamps the process-pool size to the workload,
  requested workers, and available CPUs.
* ``atomic_write_json`` / ``atomic_write_jsonl`` / ``persist_results`` persist
  scorer output via atomically replaced sibling temporary files.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Generator, Iterable, Sequence
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
    "apply_text_pipeline_with_trace",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "build_filter_artifacts",
    "normalize_single_generation_samples",
    "persist_results",
    "resolve_max_workers",
    "resolve_single_generation",
    "strip_reasoning_wrappers",
]

TextFilter = Callable[[str], str]


def build_filter_artifacts(
    raw_gen: Any, filtered_gen: Any, filter_trace: Any
) -> dict[str, Any]:
    """Build the common auditable output produced by task filter pipelines."""
    return {
        "raw_gen": raw_gen,
        "filtered_gen": filtered_gen,
        "filter_trace": filter_trace,
    }


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

    def resolve(self, name: str) -> RegisteredTextFilter:
        try:
            return self._filters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._filters)) or "<none>"
            raise ValueError(
                f"Unknown text filter {name!r}; registered filters: {available}"
            ) from exc

    def build(self, *names: str) -> tuple[RegisteredTextFilter, ...]:
        """Build an ordered pipeline from registered filter names."""
        return tuple(self.resolve(name) for name in names)

    def build_pipeline(
        self, name: str, version: str, *filter_names: str
    ) -> TextFilterPipeline:
        """Build a named, versioned pipeline from registered filters."""
        if not name:
            raise ValueError("pipeline name cannot be empty")
        if not version:
            raise ValueError("pipeline version cannot be empty")
        return TextFilterPipeline(name, version, self.build(*filter_names))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._filters))


@dataclass(frozen=True)
class TextFilterPipeline:
    """An ordered task-level text pipeline with a stable name and version."""

    name: str
    version: str
    filters: tuple[RegisteredTextFilter, ...]

    def apply_with_trace(self, value: Any) -> tuple[str, dict[str, Any]]:
        """Apply all filters and return a JSON-compatible step trace."""
        return apply_text_pipeline_with_trace(value, self)


_ANSWER_TAG_RE: re.Pattern[str] = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_THINK_END_RE: re.Pattern[str] = re.compile(r"</think\s*>", re.IGNORECASE)


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


def apply_text_pipeline_with_trace(
    value: Any,
    pipeline: Sequence[TextFilter | RegisteredTextFilter] | TextFilterPipeline | None,
    *,
    pipeline_name: str | None = None,
    pipeline_version: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply a pipeline and return an auditable JSON-compatible trace."""
    raw_text = _coerce_text(value)
    filters: Sequence[TextFilter | RegisteredTextFilter]
    if isinstance(pipeline, TextFilterPipeline):
        filters = pipeline.filters
        resolved_name = pipeline.name
        resolved_version = pipeline.version
    else:
        filters = tuple(pipeline or ())
        resolved_name = pipeline_name or "anonymous"
        resolved_version = pipeline_version or "unversioned"

    text = raw_text
    steps: list[dict[str, str]] = []
    for filter_fn in filters:
        input_text = text
        text = _coerce_text(filter_fn(text))
        steps.append(
            {
                "name": (
                    filter_fn.name
                    if isinstance(filter_fn, RegisteredTextFilter)
                    else getattr(filter_fn, "__name__", type(filter_fn).__name__)
                ),
                "version": (
                    filter_fn.version
                    if isinstance(filter_fn, RegisteredTextFilter)
                    else "unversioned"
                ),
                "input": input_text,
                "output": text,
            }
        )
    return text, {
        "pipeline": resolved_name,
        "pipeline_version": resolved_version,
        "filters": steps,
        "raw": raw_text,
        "output": text,
    }


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
        normalized.append(record)
    return normalized


def resolve_max_workers(total: int, requested: int) -> int:
    """Clamp process workers to the workload, request, and available CPUs."""
    if total < 1:
        raise ValueError("total must be positive")
    if requested < 1:
        raise ValueError("requested workers must be positive")
    cpu_count = os.cpu_count() or 1
    return min(total, requested, max(1, cpu_count - 1))


# ---------------------------------------------------------------------------
# Atomic persistence
# ---------------------------------------------------------------------------


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


def atomic_write_text(path: str | Path, content: str) -> None:
    """Replace ``path`` atomically after flushing a sibling temporary file."""
    with _atomic_text_writer(path) as handle:
        handle.write(content)


def atomic_write_json(
    path: str | Path, value: Any, *, indent: int | None = None
) -> None:
    """Serialize JSON and persist it atomically."""
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=indent)
        + ("\n" if indent is not None else ""),
    )


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Stream objects to an atomically replaced JSONL file."""
    with _atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def persist_results(
    cache_path: str | Path,
    records: Iterable[dict[str, Any]],
    summary: Any,
) -> Path:
    """Persist per-item JSONL and its adjacent summary using atomic writes."""
    destination = Path(cache_path)
    atomic_write_jsonl(destination, records)
    summary_path = destination.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary, indent=2)
    return summary_path
