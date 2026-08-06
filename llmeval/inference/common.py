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
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llmeval.utils.log import init_logger

__all__ = [
    "ResumeState",
    "build_vllm_llm_kwargs",
    "expand_data_with_resume",
    "expand_group_for_sampling",
    "is_explicit_tool_choice",
    "is_local_endpoint",
    "iter_resume_records",
    "load_jsonl",
    "load_resume_state",
    "missing_indices_for_item",
    "missing_sample_indices",
    "prepare_data_with_resume",
    "process_batches_with_policy",
    "redact_config_for_logging",
    "require_document_id",
    "resolve_resume_identity",
    "sample_count_for_item",
    "sample_seed_for_item",
    "save_failed_items",
    "to_public_result_schema",
    "validate_document_ids",
]

logger = init_logger("inference_common")


def _batch_item_identity(item: dict[str, Any]) -> dict[str, Any]:
    """Return stable, compact identity fields for a failed batch item."""
    identity: dict[str, Any] = {}
    for key in (
        "doc_id",
        "llmeval_verifier_id",
        "sample_index",
        "_llmeval_requested_sample_indices",
    ):
        if key in item:
            public_key = (
                "requested_sample_indices"
                if key == "_llmeval_requested_sample_indices"
                else key
            )
            identity[public_key] = item[key]
    return identity


def process_batches_with_policy(
    items: Sequence[dict[str, Any]],
    batch_size: int,
    process_batch: Callable[[Sequence[dict[str, Any]]], None],
    *,
    fail_fast: bool = True,
    on_batch_complete: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Process batches with an explicit strict or fault-isolating policy.

    In strict mode the original exception is propagated immediately. In
    tolerant mode failed batches are skipped and returned as compact audit
    records so callers can persist and summarize them.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    failures: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(items), batch_size)):
        batch = items[start : start + batch_size]
        try:
            process_batch(batch)
        except Exception as exc:
            if fail_fast:
                raise
            failures.append(
                {
                    "error_category": "batch_processing",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "items": [_batch_item_identity(item) for item in batch],
                }
            )
        finally:
            if on_batch_complete is not None:
                on_batch_complete()
    return failures


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
        if not isinstance(item, dict):
            raise ValueError(
                f"Input record at index {index} must be an object, "
                f"got {type(item).__name__}"
            )
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
    except (FileNotFoundError, json.JSONDecodeError):
        # Add context at the CLI boundary; library callers receive the native
        # exception and can classify it without duplicate log records.
        raise
    return records


def load_resume_state(
    output_file: str | Path,
    input_key: str,
    response_key: str,
    *,
    repair_truncated_last_line: bool = False,
) -> ResumeState:
    """Load stable and legacy resume data in one pass over an output JSONL file."""
    state = ResumeState()
    output_path = Path(output_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return state

    try:
        for line_num, item in iter_resume_records(
            output_path,
            repair_truncated_last_line=repair_truncated_last_line,
        ):
            prompt = item.get(input_key) or item.get("prompt")
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
            if prompt is None or not str(prompt).strip():
                raise ValueError(
                    f"Resume file {output_path} line {line_num} has a completed "
                    "response but no non-empty prompt"
                )

            document_id = item.get("doc_id")
            if not document_id:
                prompt_key = str(prompt)
                state.legacy_counts[prompt_key] = (
                    state.legacy_counts.get(prompt_key, 0) + count
                )
                continue

            identity = (str(document_id), str(prompt))
            completed = state.completed_indices.setdefault(identity, set())
            index_fields = [
                key
                for key in (
                    "sample_indices",
                    "_llmeval_sample_indices",
                    "_llmeval_requested_sample_indices",
                )
                if key in item
            ]
            if len(index_fields) > 1:
                raise ValueError(
                    f"Resume file {output_path} line {line_num} has ambiguous "
                    f"sample index fields: {index_fields}"
                )
            index_field = index_fields[0] if index_fields else None
            if count == 1 and index_field is not None and "sample_index" in item:
                raise ValueError(
                    f"Resume file {output_path} line {line_num} has both "
                    "sample_index and sample_indices; provide exactly one"
                )
            if index_field is not None:
                raw_indices = item[index_field]
                valid_indices = (
                    isinstance(raw_indices, list)
                    and len(raw_indices) == count
                    and all(type(index) is int and index >= 0 for index in raw_indices)
                    and len(set(raw_indices)) == len(raw_indices)
                )
                if not valid_indices:
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has invalid "
                        f"{index_field}: expected {count} unique non-negative ints"
                    )
                indices = raw_indices
            elif count == 1 and "sample_index" in item:
                raw_sample_index = item["sample_index"]
                if type(raw_sample_index) is not int or raw_sample_index < 0:
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} sample_index "
                        f"must be a non-negative int, got {raw_sample_index!r}"
                    )
                indices = [raw_sample_index]
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
        # Continuing with a partially-read state would re-generate completed
        # samples and append duplicate rows — fail loudly instead.
        raise OSError(f"Failed to read resume state from {output_file}: {exc}") from exc
    return state


def iter_resume_records(
    path: str | Path,
    *,
    repair_truncated_last_line: bool = False,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield validated resume records with their source line numbers.

    Repair mode only ignores an invalid final non-empty line when the file ends
    without a newline, which is the shape produced by an interrupted append.
    """
    resume_path = Path(path)
    pending: tuple[int, str] | None = None
    with resume_path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if pending is not None:
                yield _parse_resume_record(resume_path, *pending)
            pending = (line_num, line)

    if pending is None:
        return

    line_num, line = pending
    try:
        yield _parse_resume_record(resume_path, line_num, line)
    except ValueError as exc:
        is_unterminated = not line.endswith(("\n", "\r"))
        if (
            not repair_truncated_last_line
            or not is_unterminated
            or not isinstance(exc.__cause__, json.JSONDecodeError)
        ):
            raise
        logger.warning(
            "Ignoring truncated final resume line in %s at line %d",
            resume_path,
            line_num,
        )


def _parse_resume_record(
    path: Path, line_num: int, line: str
) -> tuple[int, dict[str, Any]]:
    try:
        item = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in resume file {path} at line {line_num}: {exc.msg}"
        ) from exc
    if not isinstance(item, dict):
        raise ValueError(
            f"Resume file {path} line {line_num} must contain an object, "
            f"got {type(item).__name__}"
        )
    return line_num, item


def missing_sample_indices(target_samples: int, completed: set[int]) -> list[int]:
    """Return the exact non-contiguous sample indices still required."""
    if target_samples <= 0:
        raise ValueError(f"target_samples must be positive, got {target_samples}")
    return [index for index in range(target_samples) if index not in completed]


def resolve_resume_identity(
    item: dict[str, Any], input_key: str, index: int | None = None
) -> tuple[str, str]:
    """Return the validated stable ``(doc_id, prompt)`` resume identity."""
    if not input_key:
        raise ValueError("input_key must be non-empty")
    prompt_value = item.get(input_key) or item.get("prompt")
    prompt = str(prompt_value) if prompt_value is not None else ""
    if not prompt.strip():
        location = f" at index {index}" if index is not None else ""
        raise ValueError(
            f"Input record{location} has no non-empty prompt under "
            f"{input_key!r} or 'prompt'"
        )
    return require_document_id(item, index), prompt


def missing_indices_for_item(
    item: dict[str, Any],
    *,
    input_key: str,
    target_samples: int,
    completed_indices: dict[tuple[str, str], set[int]],
    legacy_counts: dict[str, int],
    index: int | None = None,
) -> tuple[tuple[str, str], list[int]]:
    """Resolve one item's stable identity and exact missing sample indices."""
    identity = resolve_resume_identity(item, input_key, index)
    completed = set(completed_indices.get(identity, set()))
    if not completed:
        completed.update(range(max(legacy_counts.get(identity[1], 0), 0)))
    return identity, missing_sample_indices(target_samples, completed)


def to_public_result_schema(item: dict[str, Any]) -> dict[str, Any]:
    """Copy a result and remove inference-only request metadata."""
    result = dict(item)
    for key in tuple(result):
        if key.startswith("_llmeval_"):
            result.pop(key)
    return result


def build_vllm_llm_kwargs(args: Any) -> dict[str, Any]:
    """Assemble the vLLM ``LLM(**kwargs)`` constructor arguments.

    Shared by the offline and verifier runners, whose argument classes both
    inherit the same ``VLLMEngineArguments`` fields.  Optional fields
    (``max_num_batched_tokens``, ``quantization``, ``revision``) are only
    included when explicitly set.
    """
    llm_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "device": args.device,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.quantization is not None:
        llm_kwargs["quantization"] = args.quantization
    model_revision = getattr(args, "model_revision", None)
    if model_revision is not None:
        llm_kwargs["revision"] = model_revision
    return llm_kwargs


def redact_config_for_logging(
    payload: dict[str, Any], *, replacement: str = "***"
) -> dict[str, Any]:
    """Return a copy of a config payload with credential-like values redacted.

    Configurations are commonly serialized with ``dataclasses.asdict`` before
    logging. Keep the redaction recursive so nested request/config dictionaries
    cannot accidentally reintroduce credentials into run logs.
    """
    sensitive_fragments = (
        "api_key",
        "api-token",
        "api_token",
        "authorization",
        "cookie",
        "password",
        "secret",
        "access_token",
        "refresh_token",
    )

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
    for index, item in enumerate(raw_data):
        _, missing = missing_indices_for_item(
            item,
            input_key=input_key,
            target_samples=n_samples,
            completed_indices=completed_indices,
            legacy_counts=legacy_counts,
            index=index,
        )
        for sample_index in missing:
            expanded_item = copy.deepcopy(item)
            expanded_item["sample_index"] = sample_index
            expanded_item["expected_samples"] = n_samples
            expanded_data.append(expanded_item)
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
    for index, item in enumerate(raw_data):
        _, missing = missing_indices_for_item(
            item,
            input_key=input_key,
            target_samples=n_samples,
            completed_indices=completed_indices,
            legacy_counts=legacy_counts,
            index=index,
        )
        if not missing:
            continue

        prepared_item = copy.deepcopy(item)
        prepared_item[sample_count_key] = len(missing)
        prepared_item["_llmeval_target_samples"] = n_samples
        prepared_item["_llmeval_requested_sample_indices"] = missing
        prepared_data.append(prepared_item)

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
        if requested_indices is None:
            sample_start = int(item.get("_llmeval_sample_start", 0))
            requested_indices = list(range(sample_start, sample_start + sample_count))
        elif not (
            isinstance(requested_indices, list)
            and len(requested_indices) == sample_count
            and all(type(index) is int and index >= 0 for index in requested_indices)
            and len(set(requested_indices)) == len(requested_indices)
        ):
            raise ValueError(
                "_llmeval_requested_sample_indices must contain exactly "
                f"{sample_count} unique non-negative ints"
            )
        for sample_index in requested_indices:
            sample_item = to_public_result_schema(item)
            if "doc_id" in item:
                sample_item["sample_index"] = sample_index
            # The target sample count is used by the caller's result builder
            # for ``expected_samples``; strip the request-scoped indices but
            # keep the target so resumed rows report the full depth.
            sample_item["_llmeval_target_samples"] = int(
                item.get("_llmeval_target_samples", sample_count)
            )
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
    output_file: str | Path,
    failed_items: list[dict[str, Any]],
    *,
    run_id: str | None = None,
) -> None:
    """Append failed-item audit records next to the output file.

    Each invocation gets a run ID. A stable failure ID is derived from sample
    identity and error category, while variable text remains in the audit
    payload. Records are intentionally not deduplicated: repeated failures
    across resume runs are meaningful history and appending stays O(new rows).

    Args:
        output_file: The run's output file (derives the failed-file path).
        failed_items: Records describing each failure (item + error).
    """
    if not failed_items:
        return
    failed_file = os.path.splitext(str(output_file))[0] + "_failed.jsonl"
    current_run_id = run_id or uuid.uuid4().hex
    failed_path = Path(failed_file)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("a", encoding="utf-8") as handle:
        for entry in failed_items:
            record = dict(entry)
            record.setdefault("run_id", current_run_id)
            record.setdefault("failure_id", _failure_id(entry))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Failed items saved to: %s", failed_file)


def _failure_id(entry: dict[str, Any]) -> str:
    """Build a stable failure identity without volatile exception text."""
    source = entry.get("item") if isinstance(entry.get("item"), dict) else entry
    identity = {
        "doc_id": source.get("doc_id"),
        "sample_index": source.get("sample_index"),
        "sample_indices": source.get("sample_indices")
        or source.get("requested_sample_indices")
        or source.get("_llmeval_requested_sample_indices"),
        "error_category": entry.get("error_category")
        or entry.get("error_type")
        or "unknown",
        "batch_index": entry.get("batch_index"),
        "items": entry.get("items"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]
