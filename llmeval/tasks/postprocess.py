"""Shared input-record processing helpers for task-specific scoring.

The task scorers use these helpers explicitly so the response-cleanup chain is
easy to inspect and test.  The helpers are intentionally small and composable:

* ``strip_reasoning_wrappers`` removes common ``<think>`` / ``<answer>``
  wrappers used by reasoning models.
* ``resolve_single_generation`` validates the one-generation-per-row protocol
  and extracts the single generation from an inference output record.
* ``dedupe_repeated_samples`` skips exact duplicate rows from resumed runs
  and rejects rows that conflict on problem-level fields.
* ``resolve_max_workers`` clamps the process-pool size to the workload,
  requested workers, and available CPUs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_FILTER_REGISTRY",
    "FilterRegistry",
    "TextFilter",
    "TextFilterPipeline",
    "apply_text_pipeline_with_trace",
    "build_filter_artifacts",
    "dedupe_repeated_samples",
    "expand_single_generation_samples",
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


def dedupe_repeated_samples(
    eval_dataset: list[dict[str, Any]],
    response_key: str,
    *,
    problem_identity: Callable[[dict[str, Any], int], str],
    conflict_keys: tuple[str, ...] = (),
    record_kind: str = "document",
) -> list[dict[str, Any]]:
    """Drop exact duplicate rows while rejecting conflicting resumed rows.

    Rows that share one problem identity are independent samples, except
    exact duplicates (same response payload) produced when a resumed run
    re-appends an identical record — those are skipped idempotently.  Rows
    that repeat an identity while disagreeing on *conflict_keys* signal a
    corrupted resume and raise ``ValueError`` (mirroring the MC conflict
    detection in ``mc_score.merge_generate_records``).
    """
    first_seen: dict[str, dict[str, Any]] = {}
    seen_responses: dict[str, list[Any]] = {}
    deduped: list[dict[str, Any]] = []
    for row_index, item in enumerate(eval_dataset):
        identity = problem_identity(item, row_index)
        first = first_seen.get(identity)
        if first is None:
            first_seen[identity] = item
            seen_responses[identity] = [item.get(response_key)]
            deduped.append(item)
            continue
        for key in conflict_keys:
            if key in item and key in first and item[key] != first[key]:
                raise ValueError(
                    f"Conflicting {key!r} for resumed {record_kind} {identity!r}"
                )
        response = item.get(response_key)
        if any(response == seen for seen in seen_responses[identity]):
            continue
        seen_responses[identity].append(response)
        deduped.append(item)
    return deduped


def expand_single_generation_samples(
    eval_dataset: list[dict[str, Any]],
    response_key: str,
    *,
    problem_identity: Callable[[dict[str, Any], int], str],
) -> list[dict[str, Any]]:
    """Validate and normalize one generation per input row.

    Each output record carries exactly one generation under *response_key*
    (or an empty list when the response failed); *problem_identity* supplies
    the stable problem id used in validation error messages.
    """
    expanded: list[dict[str, Any]] = []
    for row_index, item in enumerate(eval_dataset):
        problem_id = problem_identity(item, row_index)
        response = resolve_single_generation(item, response_key, problem_id=problem_id)
        record = dict(item)
        record[response_key] = [response] if response is not None else []
        expanded.append(record)
    return expanded


def resolve_max_workers(total: int, requested: int) -> int:
    """Clamp process workers to the workload, request, and available CPUs."""
    if total < 1:
        raise ValueError("total must be positive")
    if requested < 1:
        raise ValueError("requested workers must be positive")
    cpu_count = os.cpu_count() or 1
    return min(total, requested, max(1, cpu_count - 1))
