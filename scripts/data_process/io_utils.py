"""Atomic output helpers shared by dataset preparation scripts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def has_valid_doc_ids(path: Path) -> bool:
    """Return whether a JSONL file has a unique non-empty doc_id on every row."""
    document_ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                document_id = item.get("doc_id") if isinstance(item, dict) else None
                if document_id is None or not str(document_id).strip():
                    return False
                document_key = str(document_id)
                if document_key in document_ids:
                    return False
                document_ids.add(document_key)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(document_ids)


@contextmanager
def atomic_output_path(path: str | Path) -> Generator[Path, None, None]:
    """Yield a temporary sibling path and publish it atomically on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=destination.suffix or ".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
