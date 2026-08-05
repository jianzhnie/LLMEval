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
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheStats",
    "ContentAddressedCache",
    "build_cache",
    "log_cache_stats",
]

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheStats:
    """Runtime counters for one cache instance.

    Counters are intentionally process-local. The cache files remain the
    source of truth across processes, while these counters describe the
    requests handled by the current runner.
    """

    hits: int = 0
    misses: int = 0
    corrupt: int = 0
    writes: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return counters in the schema used by logs and summaries."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "corrupt": self.corrupt,
            "writes": self.writes,
        }


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
    if isinstance(value, list | tuple):
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
        rank: int | str | None = None,
    ) -> None:
        if not namespace or "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must be a non-empty path component")
        resolved_rank = rank
        if resolved_rank is None:
            # torchrun and common distributed launchers expose RANK. A
            # caller can also use LLMEVAL_CACHE_RANK for non-distributed
            # worker pools without changing the cache key payload.
            resolved_rank = os.environ.get("LLMEVAL_CACHE_RANK") or os.environ.get(
                "RANK"
            )
        if resolved_rank is not None and (
            not str(resolved_rank).strip()
            or "/" in str(resolved_rank)
            or "\\" in str(resolved_rank)
        ):
            raise ValueError("rank must be a non-empty path component when provided")

        self.root = Path(root) / namespace
        self.rank = str(resolved_rank) if resolved_rank is not None else None
        if self.rank is not None:
            self.root /= f"rank-{self.rank}"
        self.namespace = namespace
        self.read_only = read_only
        self.force_recompute = force_recompute
        self._stats = {"hits": 0, "misses": 0, "corrupt": 0, "writes": 0}
        self._stats_lock = Lock()
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
            self._increment("misses")
            return None
        path = self._path(key)
        if not path.exists():
            self._increment("misses")
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                envelope = json.load(handle)
            if (
                not isinstance(envelope, dict)
                or envelope.get("schema_version") != CACHE_SCHEMA_VERSION
                or envelope.get("key") != key
                or not isinstance(envelope.get("value"), dict)
            ):
                self._increment("corrupt", "misses")
                return None
            self._increment("hits")
            return dict(envelope["value"])
        except (ValueError, json.JSONDecodeError, TypeError):
            self._increment("corrupt", "misses")
            return None
        except OSError:
            # A concurrent cleanup or transient filesystem error is a miss,
            # but should not be reported as a corrupt cache entry.
            self._increment("misses")
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Atomically write one entry unless the cache is read-only."""
        if self.read_only:
            return
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
            self._increment("writes")
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def stats(self) -> CacheStats:
        """Return a consistent snapshot of this instance's runtime counters."""
        with self._stats_lock:
            return CacheStats(**self._stats)

    def delete(self, key: str) -> bool:
        """Delete one cache entry and report whether it existed."""
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def clear(self) -> int:
        """Delete all JSON entries in this namespace and return the count."""
        if self.read_only:
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed += 1
        return removed

    def prune(self, max_age_seconds: float) -> int:
        """Delete entries older than ``max_age_seconds`` and return the count."""
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        if self.read_only:
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
        return removed

    def _increment(self, *counter_names: str) -> None:
        """Increment one or more runtime counters atomically."""
        with self._stats_lock:
            for counter_name in counter_names:
                self._stats[counter_name] += 1


def build_cache(
    root: str | Path,
    namespace: str,
    *,
    read_only: bool = False,
    force_recompute: bool = False,
    rank: int | str | None = None,
) -> ContentAddressedCache | None:
    """Return a cache for ``root``/``namespace``, or ``None`` when caching is off.

    An empty ``root`` means the caller did not opt in, so all runners share one
    construction site instead of repeating the same conditional.
    """
    if not root:
        return None
    return ContentAddressedCache(
        root,
        namespace,
        read_only=read_only,
        force_recompute=force_recompute,
        rank=rank,
    )


def log_cache_stats(
    cache: ContentAddressedCache | None, logger: Any, label: str
) -> None:
    """Log one cache's runtime counters when caching is enabled."""
    if cache is not None:
        logger.info("%s cache statistics: %s", label, cache.stats().to_dict())


def _main() -> int:
    """Expose basic cache inspection and cleanup through ``python -m``."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or clean LLMEval caches")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("stats", "clear", "prune"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", required=True)
        command_parser.add_argument("--namespace", required=True)
        command_parser.add_argument("--rank")
        if command == "prune":
            command_parser.add_argument("--max-age-seconds", type=float, required=True)
    args = parser.parse_args()
    cache = ContentAddressedCache(args.root, args.namespace, rank=args.rank)
    if args.command == "stats":
        entries = list(cache.root.glob("*.json"))
        print(
            json.dumps(
                {
                    "root": str(cache.root),
                    "namespace": cache.namespace,
                    "rank": cache.rank,
                    "entries": len(entries),
                    "bytes": sum(path.stat().st_size for path in entries),
                    "runtime": cache.stats().to_dict(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    removed = (
        cache.prune(args.max_age_seconds) if args.command == "prune" else cache.clear()
    )
    print(json.dumps({"removed": removed, "root": str(cache.root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
