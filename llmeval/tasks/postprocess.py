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
from typing import Any

__all__ = [
    "TextFilter",
    "apply_text_pipeline",
    "build_text_pipeline",
    "strip_reasoning_wrappers",
]

TextFilter = Callable[[str], str]

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
        return answer_match.group(1).strip()

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
    value: Any, pipeline: Sequence[TextFilter] | None = None
) -> str:
    """Apply a text-filter pipeline to a single response value."""
    text = _coerce_text(value)
    if not pipeline:
        return text

    for filter_fn in pipeline:
        text = filter_fn(text)
        if not isinstance(text, str):
            text = _coerce_text(text)
    return text
