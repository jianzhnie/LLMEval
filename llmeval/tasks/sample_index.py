"""Shared sample-index protocol for scorer input rows.

Inference output rows normally carry sample identity in one of two fields:

- ``sample_index``   — non-negative int, for single-generation rows;
- ``sample_indices`` — list of non-negative ints, one per generation.

The math, code, and MC scorers all expand or merge such rows before scoring.
For compatibility with an older batched format, a multi-generation row may
also retain a stale scalar ``sample_index``; its explicit ``sample_indices``
list remains authoritative. A single-generation row carrying both fields is
always rejected because it is ambiguous.
This module resolves the two fields into a per-row list of sample indices so
every scorer interprets the protocol identically:

1. Single-generation rows use a valid scalar ``sample_index``; carrying both
   public index fields is rejected as ambiguous.
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
    has_index = "sample_index" in item
    has_indices = "sample_indices" in item
    raw_index = item.get("sample_index")
    raw_indices = item.get("sample_indices")

    if sample_count == 1 and has_index and has_indices:
        raise ValueError(
            f"Ambiguous sample indices for problem {problem_id!r}: "
            "provide exactly one of sample_index or sample_indices"
        )

    # Single-generation rows use the scalar form when it is explicitly present.
    if sample_count == 1 and has_index:
        if not is_valid_index(raw_index):
            raise ValueError(
                f"Invalid sample_index {raw_index!r} for problem {problem_id!r}: "
                "expected a non-negative int"
            )
        return [int(raw_index)]

    # The list form is authoritative whenever present.
    if has_indices:
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

    if has_index:
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
