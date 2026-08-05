"""Content-addressed JSON caches used by inference and evaluation.

The cache deliberately uses files instead of a process-global in-memory store:
multiple workers can safely read the same result, and a corrupt entry can be
discarded without invalidating the rest of a run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

__all__ = ["CACHE_SCHEMA_VERSION", "ContentAddressedCache"]

CACHE_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats to JSON string tokens.

    Python's JSON encoder accepts ``Infinity`` and ``NaN`` by default, but
    those values are outside the JSON standard and are rejected by many other
    readers. String tokens remain lossless for current cache consumers, which
    already convert score values with ``float()`` when reading them.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


class ContentAddressedCache:
    """Store JSON values under SHA-256 keys in an isolated namespace."""

    def __init__(
        self,
        root: str | Path,
        namespace: str,
        *,
        read_only: bool = False,
        force_recompute: bool = False,
    ) -> None:
        if not namespace or "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must be a non-empty path component")
        self.root = Path(root) / namespace
        self.namespace = namespace
        self.read_only = read_only
        self.force_recompute = force_recompute
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, payload: Any) -> str:
        """Return the deterministic key for a request or evaluation payload."""
        envelope = {"schema_version": CACHE_SCHEMA_VERSION, "payload": payload}
        return hashlib.sha256(_canonical_json(envelope).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be a lowercase SHA-256 hex digest")
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Read a valid entry, returning ``None`` for misses or corruption."""
        if self.force_recompute:
            return None
        try:
            with self._path(key).open(encoding="utf-8") as handle:
                envelope = json.load(handle)
            if (
                not isinstance(envelope, dict)
                or envelope.get("schema_version") != CACHE_SCHEMA_VERSION
                or envelope.get("key") != key
                or not isinstance(envelope.get("value"), dict)
            ):
                return None
            return dict(envelope["value"])
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Atomically write one entry unless the cache is read-only."""
        if self.read_only:
            return
        if not isinstance(value, dict):
            raise TypeError("cache values must be JSON objects")
        path = self._path(key)
        self.root.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "value": _json_safe(value),
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    envelope,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
