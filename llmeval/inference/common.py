"""Backend-independent persistence, resume, and request helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llmeval.utils.log import init_logger
from llmeval.utils.prompts import is_chat_template_applied

__all__ = [
    "ResumeState",
    "append_jsonl",
    "build_chat_messages",
    "build_vllm_llm_kwargs",
    "ensure_raw_prompt",
    "get_request_seed",
    "is_explicit_tool_choice",
    "is_local_endpoint",
    "load_jsonl",
    "load_resume_state",
    "process_batches_with_policy",
    "prepare_sample_requests",
    "redact_config_for_logging",
    "save_failed_items",
]

logger = init_logger("inference_common")


# ---------------------------------------------------------------------------
# JSONL persistence and failure auditing
# ---------------------------------------------------------------------------


def append_jsonl(
    path: str | Path, records: Sequence[dict[str, Any]], lock: Any
) -> None:
    """Append JSON objects under a caller-provided synchronization lock."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        try:
            with output_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        except Exception as exc:
            raise OSError(
                f"Failed to append JSONL results to {output_path}: {exc}"
            ) from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON objects from non-blank lines in ``path``."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL line {line_num} must contain an object, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def _iter_resume_records(
    path: str | Path,
    *,
    repair_truncated_last_line: bool = False,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield strict resume rows, optionally ignoring one truncated final row."""
    resume_path = Path(path)
    with resume_path.open(encoding="utf-8") as handle:
        line_num = 0
        line = handle.readline()
        while line:
            line_num += 1
            next_line = handle.readline()
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    is_truncated_final_line = (
                        repair_truncated_last_line
                        and not next_line
                        and not line.endswith(("\n", "\r"))
                    )
                    if is_truncated_final_line:
                        logger.warning(
                            "Ignoring truncated final resume line in %s at line %d",
                            resume_path,
                            line_num,
                        )
                        return
                    raise ValueError(
                        f"Invalid JSON in resume file {resume_path} at line "
                        f"{line_num}: {exc.msg}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Resume file {resume_path} line {line_num} must contain an "
                        f"object, got {type(item).__name__}"
                    )
                yield line_num, item
            line = next_line


def save_failed_items(
    output_file: str | Path,
    failed_items: list[dict[str, Any]],
    *,
    run_id: str | None = None,
) -> None:
    """Append stable, run-scoped failure audit records next to an output file."""
    if not failed_items:
        return

    output_path = Path(output_file)
    failed_path = output_path.with_name(f"{output_path.stem}_failed.jsonl")
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    current_run_id = run_id or uuid.uuid4().hex

    with failed_path.open("a", encoding="utf-8") as handle:
        for entry in failed_items:
            nested_item = entry.get("item")
            source: dict[str, Any] = nested_item if isinstance(nested_item, dict) else entry
            identity = {
                "doc_id": source.get("doc_id"),
                "error_category": entry.get("error_category")
                or entry.get("error_type")
                or "unknown",
                "batch_index": entry.get("batch_index"),
                "items": entry.get("items"),
            }
            failure_id = hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24]

            record = dict(entry)
            record.setdefault("run_id", current_run_id)
            record.setdefault("failure_id", failure_id)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Failed items saved to: %s", failed_path)


# ---------------------------------------------------------------------------
# Resume state and request expansion
# ---------------------------------------------------------------------------


@dataclass
class ResumeState:
    """Completed row counts and recorded prompts keyed by document ID."""

    completed_counts: dict[str, int] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)

    @property
    def completed_count(self) -> int:
        """Return the total number of completed samples represented by the state."""
        return sum(self.completed_counts.values())


def get_request_seed(item: dict[str, Any]) -> int:
    """Return the validated transient seed attached during request expansion."""
    value = item.get("_request_seed")
    if type(value) is not int or value < 0:
        raise ValueError("Expanded inference item is missing a valid request seed")
    return value


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
        for line_num, item in _iter_resume_records(
            output_path,
            repair_truncated_last_line=repair_truncated_last_line,
        ):
            internal_fields = [
                key
                for key in item
                if key.startswith("_llmeval_") or key == "_request_seed"
            ]
            if internal_fields:
                raise ValueError(
                    f"Resume file {output_path} line {line_num} uses unsupported "
                    f"internal fields: {internal_fields}"
                )
            if not _is_completed_record(item, response_key, output_path, line_num):
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
            if prompt is not None:
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has an invalid "
                        "prompt; text prompts must be non-empty strings"
                    )
                prompt_text = prompt
                previous_prompt = state.prompts.setdefault(document_key, prompt_text)
                if previous_prompt != prompt_text:
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has conflicting "
                        f"prompts for doc_id={document_key!r}"
                    )
    except OSError as exc:
        raise OSError(f"Failed to read resume state from {output_file}: {exc}") from exc
    return state


def _is_completed_record(
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


def prepare_sample_requests(
    raw_data: list[dict[str, Any]],
    resume_state: ResumeState,
    input_key: str,
    n_samples: int,
    *,
    base_seed: int,
    prompt_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Return one independent request row per unfinished sample."""
    if not input_key:
        raise ValueError("input_key must be non-empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")

    expanded: list[dict[str, Any]] = []
    first_indices: dict[str, int] = {}
    for index, source in enumerate(raw_data):
        if not isinstance(source, dict):
            raise ValueError(
                f"Input record at index {index} must be an object, "
                f"got {type(source).__name__}"
            )
        document_id = source.get("doc_id")
        if document_id is None or not str(document_id).strip():
            raise ValueError(
                f"Input record at index {index} is missing required 'doc_id'. "
                "Regenerate the evaluation dataset with the data preparation script."
            )
        document_key = str(document_id)
        previous_index = first_indices.setdefault(document_key, index)
        if previous_index != index:
            raise ValueError(
                f"Duplicate doc_id {document_key!r} at indices {previous_index} and "
                f"{index}. Each prepared question must have a unique ID."
            )
        if prompt_resolver is None:
            prompt_value = source.get(input_key) or source.get("prompt")
        else:
            prompt_value = prompt_resolver(source)
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            detail = (
                f"got {type(prompt_value).__name__}"
                if prompt_value is not None
                else "field is missing"
            )
            raise ValueError(
                f"Input record at index {index} must contain a non-empty string prompt "
                f"under {input_key!r} or 'prompt'; {detail}"
            )
        prompt = prompt_value

        recorded_prompt = resume_state.prompts.get(document_key)
        if recorded_prompt is not None and recorded_prompt != prompt:
            raise ValueError(
                f"Input record at index {index} changed prompt for doc_id="
                f"{document_key!r}; use a new output file"
            )
        completed = resume_state.completed_counts.get(document_key, 0)
        if completed > n_samples:
            raise ValueError(
                f"Resume output contains {completed} rows for doc_id={document_key!r}, "
                f"exceeding requested n_samples={n_samples}"
            )

        for generation_ordinal in range(completed, n_samples):
            seed_payload = (
                f"{base_seed}\0{document_key}\0{prompt}\0{generation_ordinal}"
            ).encode("utf-8", errors="replace")
            item = copy.deepcopy(source)
            item["_request_seed"] = (
                int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "big")
                & 0x7FFFFFFF
            )
            item["expected_samples"] = n_samples
            expanded.append(item)
    return expanded


# ---------------------------------------------------------------------------
# Backend request and configuration helpers
# ---------------------------------------------------------------------------


def ensure_raw_prompt(prompt: str) -> None:
    """Reject prompts that already contain a serialized chat template."""
    if is_chat_template_applied(prompt):
        raise ValueError(
            "Query already contains a chat_template; provide the raw prompt because "
            "the inference backend applies its template automatically"
        )


def build_chat_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    """Build chat messages with an optional leading system message."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def process_batches_with_policy(
    items: Sequence[dict[str, Any]],
    batch_size: int,
    process_batch: Callable[[Sequence[dict[str, Any]]], None],
    *,
    fail_fast: bool = True,
    on_batch_complete: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Process batches strictly or return compact audit records for failures."""
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
                    "items": [
                        {"doc_id": item["doc_id"]} if "doc_id" in item else {}
                        for item in batch
                    ],
                }
            )
        finally:
            if on_batch_complete is not None:
                on_batch_complete()
    return failures


def is_local_endpoint(base_url: str) -> bool:
    """Return whether an API URL targets the local machine."""
    hostname = urlparse(base_url).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"} or bool(
        hostname and hostname.endswith(".localhost")
    )


def build_vllm_llm_kwargs(args: Any) -> dict[str, Any]:
    """Build vLLM constructor arguments, omitting unset optional fields."""
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
    """Recursively redact credential-like values from a configuration copy."""
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

    return redact(payload)


def is_explicit_tool_choice(tool_choice: str | None) -> bool:
    """Return whether ``tool_choice`` should be sent to an API backend."""
    return bool(tool_choice and tool_choice.strip().lower() != "none")
