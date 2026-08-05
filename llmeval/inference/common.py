"""Shared data-loading and resume helpers for the inference runners.

The online, offline, verifier, and MC runners all implement the same pipeline
stages: load JSONL input, recover completed sample indices from the output file,
expand the dataset to ``n_samples`` copies per prompt, and persist
failed items for debugging.  Those helpers live here so each runner module
stays focused on its own backend (OpenAI API vs. vLLM engine).

Functions
---------
load_jsonl                 — parse a line-delimited JSON file
load_resume_state          — stable sample indices and legacy counts for resume
expand_data_with_resume    — expand raw items to remaining per-sample copies
prepare_data_with_resume   — attach remaining sample counts for batched online runs
sample_count_for_item      — read the runtime sample count from an item
expand_group_for_sampling   — expand grouped prompt records by sample count
save_failed_items          — persist failure records next to the output file
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llmeval.utils.log import init_logger

__all__ = [
    "ResumeState",
    "expand_data_with_resume",
    "expand_group_for_sampling",
    "is_explicit_tool_choice",
    "is_local_endpoint",
    "load_jsonl",
    "load_resume_state",
    "prepare_data_with_resume",
    "redact_config_for_logging",
    "require_document_id",
    "sample_count_for_item",
    "sample_seed_for_item",
    "save_failed_items",
    "validate_document_ids",
]

logger = init_logger("inference_common")


@dataclass
class ResumeState:
    """Completed stable sample indices and legacy prompt counts from one output."""

    completed_indices: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    legacy_counts: dict[str, int] = field(default_factory=dict)

    @property
    def completed_count(self) -> int:
        """Return the total number of completed samples represented by the state."""
        return sum(len(indices) for indices in self.completed_indices.values()) + sum(
            self.legacy_counts.values()
        )


def is_local_endpoint(base_url: str) -> bool:
    """Return whether an API URL targets the local machine."""
    hostname = urlparse(base_url).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"} or bool(
        hostname and hostname.endswith(".localhost")
    )


def require_document_id(item: dict[str, Any], index: int | None = None) -> str:
    """Return the dataset-provided ``doc_id`` or raise a preparation error.

    Document identity is assigned once by the benchmark preparation scripts
    and persisted in JSONL. Inference must never synthesize a replacement ID,
    because doing so makes resume state depend on input ordering or prompts.
    """
    document_id = item.get("doc_id")
    if document_id is None or not str(document_id).strip():
        location = f" at index {index}" if index is not None else ""
        raise ValueError(
            f"Input record{location} is missing required 'doc_id'. "
            "Regenerate the evaluation dataset with the data preparation script."
        )
    return str(document_id)


def validate_document_ids(items: list[dict[str, Any]]) -> None:
    """Validate that every prepared input record has a unique ``doc_id``."""
    first_indices: dict[str, int] = {}
    for index, item in enumerate(items):
        document_id = require_document_id(item, index)
        previous = first_indices.setdefault(document_id, index)
        if previous != index:
            raise ValueError(
                f"Duplicate doc_id {document_id!r} at indices {previous} and {index}. "
                "Each prepared question must have a unique ID."
            )


def sample_seed_for_item(base_seed: int, item: dict[str, Any]) -> int:
    """Derive a stable independent backend seed for one generated sample.

    The document ID and sample index are part of the seed so repeated requests
    for the same prompt do not accidentally reuse one deterministic sequence,
    while resume runs still reproduce the same sample.
    """
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    document_id = str(item.get("doc_id") or item.get("llmeval_verifier_id") or "")
    prompt = str(item.get("prompt") or item.get("question") or "")
    sample_index = item.get("sample_index", 0)
    try:
        sample_index = int(sample_index)
    except (TypeError, ValueError):
        sample_index = 0
    payload = f"{base_seed}\0{document_id}\0{prompt}\0{sample_index}".encode(
        "utf-8", errors="replace"
    )
    # Keep the value within the range accepted by common vLLM backends.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a line-delimited JSON file, skipping blank lines.

    Args:
        path: Input JSONL file path.

    Returns:
        Parsed items, one per non-blank line.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If an input line is not valid JSON.
        ValueError: If a line contains valid JSON but not an object.
    """
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL line {line_num} must contain an object, "
                        f"got {type(record).__name__}"
                    )
                records.append(record)
    except FileNotFoundError as e:
        logger.critical(f"Input file not found: {path}, {e}")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Invalid JSON in input file: {e}")
        raise
    return records


def load_resume_state(
    output_file: str | Path,
    input_key: str,
    response_key: str,
) -> ResumeState:
    """Load stable and legacy resume data in one pass over an output JSONL file."""
    state = ResumeState()
    output_path = Path(output_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return state

    try:
        with output_path.open(encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON on output line %d: %s", line_num, exc)
                    continue
                if not isinstance(item, dict):
                    logger.warning(
                        "Skipping non-object JSON on output line %d: %s",
                        line_num,
                        type(item).__name__,
                    )
                    continue

                prompt = item.get(input_key) or item.get("prompt")
                if prompt is None:
                    continue
                response = item.get(response_key)
                if response is None:
                    response = item.get("gen")
                if isinstance(response, list):
                    count = len(response)
                elif (
                    item.get("logprobs") is not None
                    or item.get("Verifier_response")
                    or item.get("Verifier_judgment")
                ):
                    count = 1
                else:
                    count = 0
                if count <= 0:
                    continue

                document_id = item.get("doc_id")
                if not document_id:
                    prompt_key = str(prompt)
                    state.legacy_counts[prompt_key] = (
                        state.legacy_counts.get(prompt_key, 0) + count
                    )
                    continue

                identity = (str(document_id), str(prompt))
                completed = state.completed_indices.setdefault(identity, set())
                raw_indices = item.get("_llmeval_sample_indices")
                if (
                    isinstance(raw_indices, list)
                    and len(raw_indices) == count
                    and all(
                        isinstance(index, int) and index >= 0 for index in raw_indices
                    )
                ):
                    indices = raw_indices
                elif isinstance(item.get("sample_index"), int):
                    indices = [int(item["sample_index"])]
                else:
                    indices = []
                    next_index = 0
                    for _ in range(count):
                        while next_index in completed:
                            next_index += 1
                        indices.append(next_index)
                        next_index += 1
                completed.update(indices)
    except OSError as exc:
        logger.error("Error reading resume state from %s: %s", output_file, exc)
    return state


def redact_config_for_logging(
    payload: dict[str, Any], *, replacement: str = "***"
) -> dict[str, Any]:
    """Return a copy of a config payload with credential-like values redacted.

    Configurations are commonly serialized with ``dataclasses.asdict`` before
    logging. Keep the redaction recursive so nested request/config dictionaries
    cannot accidentally reintroduce credentials into run logs.
    """
    sensitive_fragments = ("api_key", "authorization", "cookie")

    def redact(value: Any, key: str | None = None) -> Any:
        if key and any(fragment in key.lower() for fragment in sensitive_fragments):
            return replacement
        if isinstance(value, dict):
            return {str(name): redact(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    result = redact(payload)
    return result if isinstance(result, dict) else {}


def expand_data_with_resume(
    raw_data: list[dict[str, Any]],
    completed_indices: dict[tuple[str, str], set[int]],
    legacy_counts: dict[str, int],
    input_key: str,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Expand raw items into per-sample copies for every index still missing.

    Explicit index sets preserve holes from partially failed batched requests,
    so resume regenerates the missing sample instead of duplicating a later one.

    Args:
        raw_data: Items loaded from the input file; each must carry ``doc_id``.
        completed_indices: Completed sample indices per ``(doc_id, prompt)``.
        legacy_counts: Prompt-keyed completed counts for legacy output rows
            that predate stable IDs (treated as contiguous from index 0).
        input_key: Prompt field name (``"prompt"`` used as fallback).
        n_samples: Target number of samples per prompt.

    Returns:
        Expanded dataset holding only the samples still to process, each tagged
        with its ``sample_index``.
    """
    validate_document_ids(raw_data)

    expanded_data: list[dict[str, Any]] = []
    skipped_items = 0
    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict input item: %s", type(item).__name__)
            skipped_items += 1
            continue
        prompt_val: Any = item.get(input_key) or item.get("prompt")
        prompt = str(prompt_val) if prompt_val is not None else ""
        if not prompt.strip():
            logger.warning(
                f"No valid prompt found under keys [{input_key!r}, 'prompt'] "
                f"for item with keys: {list(item.keys())}"
            )
            skipped_items += 1
            continue

        document_id = require_document_id(item, index)
        used = set(completed_indices.get((document_id, prompt), set()))
        if not used:
            # Legacy prompt-keyed counts carry no per-index metadata; assume the
            # first ``legacy_done`` contiguous indices were written.
            legacy_done = legacy_counts.get(prompt, 0)
            used = set(range(max(legacy_done, 0)))
        for sample_index in range(n_samples):
            if sample_index in used:
                continue
            expanded_item = copy.deepcopy(item)
            expanded_item["sample_index"] = sample_index
            expanded_data.append(expanded_item)

    if skipped_items > 0:
        logger.warning(f"Skipped {skipped_items} items due to missing or empty prompt")
    return expanded_data


def prepare_data_with_resume(
    raw_data: list[dict[str, Any]],
    completed_indices: dict[tuple[str, str], set[int]],
    legacy_counts: dict[str, int],
    input_key: str,
    n_samples: int,
    sample_count_key: str = "n_samples",
) -> list[dict[str, Any]]:
    """Prepare grouped online requests using the exact missing sample indices.

    Stable-ID output can contain holes when one choice in a batched request was
    empty or failed. The returned metadata preserves those holes so a resumed
    request regenerates precisely ``target - completed`` rather than assuming
    completed samples form a prefix.
    """
    if not input_key:
        raise ValueError("input_key must be non-empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    validate_document_ids(raw_data)
    prepared_data: list[dict[str, Any]] = []
    skipped_items = 0
    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            skipped_items += 1
            continue
        prompt_val: Any = item.get(input_key) or item.get("prompt")
        prompt = str(prompt_val) if prompt_val is not None else ""
        if not prompt.strip():
            skipped_items += 1
            continue

        document_id = require_document_id(item, index)
        identity = (document_id, prompt)
        used = set(completed_indices.get(identity, set()))
        if not used:
            used = set(range(max(legacy_counts.get(prompt, 0), 0)))
        missing = [sample_index for sample_index in range(n_samples) if sample_index not in used]
        if not missing:
            continue

        prepared_item = copy.deepcopy(item)
        prepared_item[sample_count_key] = len(missing)
        prepared_item["_llmeval_requested_sample_indices"] = missing
        prepared_data.append(prepared_item)

    if skipped_items:
        logger.warning("Skipped %d items due to missing or empty prompt", skipped_items)
    return prepared_data


def sample_count_for_item(
    item: dict[str, Any], sample_count_key: str = "n_samples"
) -> int:
    """Return the sample count stored on a prepared item."""
    try:
        return max(1, int(item.get(sample_count_key, 1)))
    except (TypeError, ValueError):
        return 1


def expand_group_for_sampling(
    items: list[dict[str, Any]], sample_count_key: str = "n_samples"
) -> list[dict[str, Any]]:
    """Expand grouped prompt records according to their stored sample counts."""
    if not any(sample_count_key in item for item in items if isinstance(item, dict)):
        return items

    sample_items: list[dict[str, Any]] = []
    for item in items:
        sample_count = sample_count_for_item(item, sample_count_key)
        requested_indices = item.get("_llmeval_requested_sample_indices")
        if not (
            isinstance(requested_indices, list)
            and len(requested_indices) == sample_count
            and all(isinstance(index, int) and index >= 0 for index in requested_indices)
        ):
            sample_start = int(item.get("_llmeval_sample_start", 0))
            requested_indices = list(range(sample_start, sample_start + sample_count))
        for sample_index in requested_indices:
            sample_item = item.copy()
            if "doc_id" in item:
                sample_item["sample_index"] = sample_index
            sample_items.append(sample_item)
    return sample_items


def is_explicit_tool_choice(tool_choice: str | None) -> bool:
    """Return whether ``tool_choice`` should be sent to the API.

    The CLI default ``"none"`` means "do not enable tools" for OpenAI-compatible
    backends.  Omitting the field is more compatible than sending
    ``tool_choice="none"`` to servers that do not implement tool calling.
    """
    return bool(tool_choice and tool_choice.strip().lower() != "none")


def save_failed_items(
    output_file: str | Path, failed_items: list[dict[str, Any]]
) -> None:
    """Persist failed-item records next to the output file.

    Appends to ``<output_stem>_failed.jsonl`` so failures from earlier resume
    attempts remain available. ``splitext`` also ensures a non-JSONL output
    name cannot collapse onto the primary output file.

    Args:
        output_file: The run's output file (derives the failed-file path).
        failed_items: Records describing each failure (item + error).
    """
    failed_file = os.path.splitext(str(output_file))[0] + "_failed.jsonl"
    try:
        Path(failed_file).parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, "a", encoding="utf-8") as f:
            for entry in failed_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Failed items saved to: {failed_file}")
    except OSError as e:
        logger.error(f"Failed to save failed items to file: {e}")
