"""Atomic persistence primitives shared by task scorers and registry adapters."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from io import TextIOBase
from pathlib import Path
from typing import Any


@contextmanager
def _atomic_text_writer(path: str | Path) -> Iterator[TextIOBase]:
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
    except BaseException:
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
