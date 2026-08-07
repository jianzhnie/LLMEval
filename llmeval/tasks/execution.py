"""Execution-resource helpers shared by task scorers."""

from __future__ import annotations

import os


def resolve_max_workers(total: int, requested: int) -> int:
    """Clamp process workers to the workload, request, and available CPUs."""
    if total < 1:
        raise ValueError("total must be positive")
    if requested < 1:
        raise ValueError("requested workers must be positive")
    cpu_count = os.cpu_count() or 1
    return min(total, requested, max(1, cpu_count - 1))
