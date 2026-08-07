"""Shared infrastructure for inference runners.

This module owns backend-independent concerns: JSONL parsing, resume-state
validation, sampling-plan construction, configuration redaction, and failure
auditing.  Backend-specific request shapes stay in the online/offline runners.
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
    "is_explicit_tool_choice",
    "is_local_endpoint",
    "iter_resume_records",
    "load_jsonl",
    "load_resume_state",
    "get_request_seed",
    "process_batches_with_policy",
    "redact_config_for_logging",
    "require_document_id",
    "save_failed_items",
    "validate_document_ids",
]

logger = init_logger("inference_common")


def _batch_item_identity(item: dict[str, Any]) -> dict[str, Any]:
    """Return stable, compact identity fields for a failed batch item."""
    identity: dict[str, Any] = {}
    for key in ("doc_id",):
        if key in item:
            identity[key] = item[key]
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
    """Completed row counts and recorded prompts keyed by document ID."""

    completed_counts: dict[str, int] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)

    @property
    def completed_count(self) -> int:
        """Return the total number of completed samples represented by the state."""
        return sum(self.completed_counts.values())


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


def _derive_request_seed(
    base_seed: int, item: dict[str, Any], generation_ordinal: int
) -> int:
    """Derive a stable seed for one repeated generation request."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if generation_ordinal < 0:
        raise ValueError("generation_ordinal must be non-negative")
    document_id = str(item.get("doc_id") or "")
    prompt = str(item.get("prompt") or item.get("question") or "")
    payload = f"{base_seed}\0{document_id}\0{prompt}\0{generation_ordinal}".encode(
        "utf-8", errors="replace"
    )
    # Keep the value within the range accepted by common vLLM backends.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def get_request_seed(item: dict[str, Any]) -> int:
    """Consume the transient seed attached during request expansion."""
    value = item.get("_request_seed")
    if type(value) is not int or value < 0:
        raise ValueError("Expanded inference item is missing a valid request seed")
    return value


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
    """Load completed-row counts keyed by stable document ID."""
    state = ResumeState()
    output_path = Path(output_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return state

    try:
        for line_num, item in iter_resume_records(
            output_path,
            repair_truncated_last_line=repair_truncated_last_line,
        ):
            internal_fields = [key for key in item if key.startswith("_llmeval_")]
            if internal_fields:
                raise ValueError(
                    f"Resume file {output_path} line {line_num} uses unsupported "
                    f"internal fields: {internal_fields}"
                )

            if not _is_completed_resume_record(
                item, response_key, output_path, line_num
            ):
                continue

            document_id = item.get("doc_id")
            if document_id is None or not str(document_id).strip():
                raise ValueError(
                    f"Resume file {output_path} line {line_num} is missing required "
                    "'doc_id'; migrate legacy resume output before continuing"
                )
            document_key = str(document_id)
            state.completed_counts[document_key] = (
                state.completed_counts.get(document_key, 0) + 1
            )

            prompt = item.get(input_key) or item.get("prompt")
            if prompt is not None and str(prompt).strip():
                prompt_text = str(prompt)
                previous_prompt = state.prompts.setdefault(document_key, prompt_text)
                if previous_prompt != prompt_text:
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has conflicting "
                        f"prompts for doc_id={document_key!r}"
                    )
    except OSError as exc:
        # Continuing with a partially-read state would re-generate completed
        # samples and append duplicate rows — fail loudly instead.
        raise OSError(f"Failed to read resume state from {output_file}: {exc}") from exc
    return state


def _is_completed_resume_record(
    item: dict[str, Any], response_key: str, output_path: Path, line_num: int
) -> bool:
    """Validate the one-result-per-row protocol and report completion."""
    if item.get("logprobs") is not None:
        return True

    response = item.get(response_key)
    if response is None and response_key != "gen":
        response = item.get("gen")
    if response is None:
        return False
    if isinstance(response, list):
        if len(response) != 1:
            raise ValueError(
                f"Resume file {output_path} line {line_num} must contain exactly "
                "one generation; migrate grouped output to one row per sample"
            )
        response = response[0]
    if not isinstance(response, str) or not response.strip():
        raise ValueError(
            f"Resume file {output_path} line {line_num} has an empty or invalid "
            f"{response_key!r} result"
        )
    return True


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


def expand_data_with_resume(
    raw_data: list[dict[str, Any]],
    resume_state: ResumeState,
    input_key: str,
    n_samples: int,
    *,
    base_seed: int,
    prompt_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Copy each input once per remaining generation request."""
    if not input_key:
        raise ValueError("input_key must be non-empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    validate_document_ids(raw_data)
    expanded: list[dict[str, Any]] = []
    for index, source in enumerate(raw_data):
        if prompt_resolver is None:
            prompt_value = source.get(input_key) or source.get("prompt")
            prompt = str(prompt_value) if prompt_value is not None else ""
        else:
            prompt = str(prompt_resolver(source))
        if not prompt.strip():
            raise ValueError(
                f"Input record at index {index} has no non-empty prompt under "
                f"{input_key!r} or 'prompt'"
            )

        document_id = str(source["doc_id"])
        recorded_prompt = resume_state.prompts.get(document_id)
        if recorded_prompt is not None and recorded_prompt != prompt:
            raise ValueError(
                f"Input record at index {index} changed prompt for doc_id="
                f"{document_id!r}; use a new output file"
            )
        completed = resume_state.completed_counts.get(document_id, 0)
        if completed > n_samples:
            raise ValueError(
                f"Resume output contains {completed} rows for doc_id={document_id!r}, "
                f"exceeding requested n_samples={n_samples}"
            )
        for generation_ordinal in range(completed, n_samples):
            item = copy.deepcopy(source)
            item["_request_seed"] = _derive_request_seed(
                base_seed, item, generation_ordinal
            )
            item["expected_samples"] = n_samples
            expanded.append(item)
    return expanded


def build_vllm_llm_kwargs(args: Any) -> dict[str, Any]:
    """Assemble the vLLM ``LLM(**kwargs)`` constructor arguments.

    Used by the offline runner. Optional fields
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
