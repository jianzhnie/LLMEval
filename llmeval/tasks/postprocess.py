"""Shared text post-processing helpers for task-specific scoring.

The task scorers use these helpers explicitly so the response-cleanup chain is
easy to inspect and test.  The helpers are intentionally small and composable:

* ``build_text_pipeline`` collects the ordered filters.
* ``apply_text_pipeline`` runs the pipeline on a response value.
* ``strip_reasoning_wrappers`` removes common ``<think>`` / ``<answer>``
  wrappers used by reasoning models.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_FILTER_REGISTRY",
    "FilterRegistry",
    "RegisteredTextFilter",
    "TextFilter",
    "TextFilterPipeline",
    "apply_text_pipeline",
    "apply_text_pipeline_with_trace",
    "build_text_pipeline",
    "strip_reasoning_wrappers",
]

TextFilter = Callable[[str], str]


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

    def apply(self, value: Any) -> str:
        """Apply all filters and return the final text."""
        return apply_text_pipeline(value, self)

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


def build_text_pipeline(*filters: TextFilter) -> tuple[TextFilter, ...]:
    """Collect an ordered tuple of text filters."""
    return tuple(filters)


def apply_text_pipeline(
    value: Any,
    pipeline: Sequence[TextFilter | RegisteredTextFilter]
    | TextFilterPipeline
    | None = None,
) -> str:
    """Apply a text-filter pipeline to a single response value."""
    text = _coerce_text(value)
    if not pipeline:
        return text

    filters = pipeline.filters if isinstance(pipeline, TextFilterPipeline) else pipeline
    for filter_fn in filters:
        text = filter_fn(text)
        if not isinstance(text, str):
            text = _coerce_text(text)
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
