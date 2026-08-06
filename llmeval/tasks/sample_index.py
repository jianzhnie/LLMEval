"""Shared sample-index protocol for scorer input rows.

Inference output rows carry sample identity in exactly one of two fields:

- ``sample_index``   — non-negative int, for single-generation rows;
- ``sample_indices`` — list of non-negative ints, one per generation.

The math, code, and MC scorers all expand or merge such rows before scoring.
This module resolves the two fields into a per-row list of sample indices so
every scorer interprets the protocol identically:

1. Single-generation rows prefer a valid scalar ``sample_index``.
2. Rows with an explicit ``sample_indices`` list must match the generation
   count exactly (length, element type, and non-negative range).
3. Explicit-but-invalid fields raise a schema ``ValueError`` naming the
   problem — rows are never silently renumbered.
4. Only rows without any index field fall back to legacy compatibility
   allocation: the next unused indices for the same problem, in order.

All three scorers also apply the same duplicate rule to a repeated
``(problem, sample_index)`` pair: identical raw generation content merges
idempotently (scored once, one per-item record), while different content
raises :func:`duplicate_sample_error`.

An empty-generation row may carry ``sample_indices: []``. A scalar
``sample_index`` cannot describe an empty or multi-generation row.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

__all__ = ["duplicate_sample_error", "is_valid_index", "resolve_sample_indices"]


def is_valid_index(value: Any) -> bool:
    """Return whether *value* is a usable sample index (non-negative int)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def duplicate_sample_error(problem_id: str, sample_index: int) -> ValueError:
    """Build the uniform conflict error for ``(problem_id, sample_index)``.

    A repeated index with identical content merges idempotently; a repeated
    index with different content is a data corruption and must surface.
    """
    return ValueError(
        f"Conflicting duplicate sample {problem_id!r}/{sample_index}: "
        "same sample_index carries different content"
    )


def resolve_sample_indices(
    item: dict[str, Any],
    sample_count: int,
    *,
    problem_id: str,
    used_indices: Collection[int] = (),
) -> list[int]:
    """Resolve the sample indices carried by one inference output row.

    Args:
        item: One raw inference output record.
        sample_count: Number of generations the row actually carries.
        problem_id: Stable problem identifier, used in error messages.
        used_indices: Indices already taken by earlier rows of the same
            problem; only consulted for legacy rows without index fields.

    Returns:
        Exactly ``sample_count`` sample indices, in generation order.

    Raises:
        ValueError: When an explicit ``sample_index``/``sample_indices``
            field is present but has the wrong type, length, or range.
    """
    raw_index = item.get("sample_index")
    raw_indices = item.get("sample_indices")

    # Single-generation rows prefer the scalar form.
    if sample_count == 1 and raw_index is not None:
        if not is_valid_index(raw_index):
            raise ValueError(
                f"Invalid sample_index {raw_index!r} for problem {problem_id!r}: "
                "expected a non-negative int"
            )
        return [int(raw_index)]

    # The list form is authoritative whenever present.
    if raw_indices is not None:
        if not isinstance(raw_indices, list) or len(raw_indices) != sample_count:
            raise ValueError(
                f"Invalid sample_indices {raw_indices!r} for problem "
                f"{problem_id!r}: expected a list of {sample_count} "
                "non-negative ints"
            )
        if any(not is_valid_index(index) for index in raw_indices):
            raise ValueError(
                f"Invalid sample_indices {raw_indices!r} for problem "
                f"{problem_id!r}: every index must be a non-negative int"
            )
        if len(set(raw_indices)) != len(raw_indices):
            raise ValueError(
                f"Invalid sample_indices {raw_indices!r} for problem "
                f"{problem_id!r}: indices must be unique within a row"
            )
        return list(raw_indices)

    if raw_index is not None:
        if not is_valid_index(raw_index):
            raise ValueError(
                f"Invalid sample_index {raw_index!r} for problem {problem_id!r}: "
                "expected a non-negative int"
            )
        raise ValueError(
            f"Invalid sample_index {raw_index!r} for problem {problem_id!r}: "
            f"a scalar index cannot describe {sample_count} generations; "
            "use sample_indices instead"
        )

    # Legacy rows without index fields: allocate the next unused indices.
    used = set(used_indices)
    indices: list[int] = []
    next_index = 0
    for _ in range(sample_count):
        while next_index in used:
            next_index += 1
        indices.append(next_index)
        used.add(next_index)
        next_index += 1
    return indices
