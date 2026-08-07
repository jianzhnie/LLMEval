"""Validation helpers for the one-generation-per-row scorer protocol."""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_single_generation"]


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
