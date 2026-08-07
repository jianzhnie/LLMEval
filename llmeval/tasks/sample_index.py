"""Validation helpers for the one-sample-per-row scorer protocol."""

from __future__ import annotations

from typing import Any

__all__ = [
    "duplicate_sample_error",
    "is_valid_index",
    "resolve_sample_index",
    "resolve_single_generation",
]


def is_valid_index(value: Any) -> bool:
    """Return whether *value* is a non-negative integer sample index."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def duplicate_sample_error(problem_id: str, sample_index: int) -> ValueError:
    """Build the uniform error for a conflicting repeated sample."""
    return ValueError(
        f"Conflicting duplicate sample {problem_id!r}/{sample_index}: "
        "same sample_index carries different content"
    )


def resolve_sample_index(item: dict[str, Any], *, problem_id: str) -> int:
    """Read and validate the required public sample identity."""
    value = item.get("sample_index")
    if not is_valid_index(value):
        raise ValueError(
            f"Invalid sample_index {value!r} for problem {problem_id!r}: "
            "expected a non-negative int"
        )
    return value


def resolve_single_generation(
    item: dict[str, Any], response_key: str, *, problem_id: str
) -> str | None:
    """Return one generation, using ``None`` for a failed empty response.

    Scorer input is deliberately not a batching protocol: every row describes
    exactly one sample. A list is accepted because inference JSONL stores a
    successful generation as a one-element list, while an empty list records
    an inference failure.
    """
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
