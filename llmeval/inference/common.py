"""Backend-independent persistence, resume, and request helpers."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from tqdm import tqdm

from llmeval.tasks.postprocess import atomic_write_json
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import is_chat_template_applied

__all__ = [
    "ResumeState",
    "append_jsonl",
    "build_chat_messages",
    "build_vllm_llm_kwargs",
    "derive_request_seed",
    "ensure_raw_prompt",
    "is_local_endpoint",
    "load_jsonl",
    "load_resume_state",
    "prepare_sample_requests",
    "redact_config_for_logging",
    "run_concurrent_requests",
    "warn_result_manifest",
    "write_run_manifest",
]

logger = init_logger("inference_common")
_T = TypeVar("_T")
_R = TypeVar("_R")


def reject_nonfinite_json(value: str) -> None:
    """Reject JavaScript-style numeric constants outside the JSON standard."""
    raise ValueError(f"non-standard JSON numeric constant: {value}")


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def append_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    """Append JSON objects to a newline-terminated result file.

    An existing file whose final record lacks a trailing newline (e.g. a
    crash before flush, which ``load_resume_state`` still parses) is repaired
    by adding the newline before appending, matching resume semantics.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    try:
        payload = "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        )
        if output_path.exists() and output_path.stat().st_size:
            with output_path.open("r+b") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.seek(0, 2)
                    handle.write(b"\n")
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
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
            record = json.loads(line, parse_constant=reject_nonfinite_json)
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL line {line_num} must contain an object, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def _manifest_path(result_path: str | Path) -> Path:
    return Path(f"{result_path}.manifest.json")


def _load_run_manifest(result_path: str | Path) -> tuple[list[str], int] | None:
    """Load and validate a result manifest."""
    path = _manifest_path(result_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite_json
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid run manifest {path}: {exc}") from exc
    doc_ids = payload.get("doc_ids") if isinstance(payload, dict) else None
    n_samples = payload.get("n_samples") if isinstance(payload, dict) else None
    if (
        not isinstance(doc_ids, list)
        or any(not isinstance(value, str) or not value.strip() for value in doc_ids)
        or len(doc_ids) != len(set(doc_ids))
        or type(n_samples) is not int
        or n_samples <= 0
    ):
        raise ValueError(f"Run manifest {path} has invalid doc_ids or n_samples")
    return doc_ids, n_samples


def write_run_manifest(
    result_path: str | Path,
    raw_data: Sequence[dict[str, Any]],
    n_samples: int,
) -> None:
    """Create or validate the expected-result sidecar for resumable inference."""
    if type(n_samples) is not int or n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    doc_ids = [str(item.get("doc_id", "")) for item in raw_data]
    if any(not document_id.strip() for document_id in doc_ids):
        raise ValueError("Every input record must have a non-empty doc_id")
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Input data contains duplicate doc_id values")

    expected = (doc_ids, n_samples)
    existing = _load_run_manifest(result_path)
    if existing is not None:
        if existing != expected:
            raise ValueError(
                f"Run manifest {_manifest_path(result_path)} does not match the "
                "current input data or n_samples; use a new output file"
            )
        return

    output_path = Path(result_path)
    if output_path.exists() and output_path.stat().st_size:
        logger.warning(
            "Existing result %s has no run manifest; preserving legacy resume behavior",
            output_path,
        )
        return

    atomic_write_json(
        _manifest_path(result_path),
        {"doc_ids": doc_ids, "n_samples": n_samples},
        indent=2,
    )


def warn_result_manifest(
    records: Sequence[dict[str, Any]], result_path: str | Path
) -> None:
    """Warn about incomplete inference output without blocking evaluation."""
    try:
        manifest = _load_run_manifest(result_path)
    except ValueError as exc:
        logger.warning("Result completeness check: %s", exc)
        return
    if manifest is None:
        return

    doc_ids, n_samples = manifest
    expected = {
        (document_id, str(sample_index), str(n_samples))
        for document_id in doc_ids
        for sample_index in range(n_samples)
    }
    actual_rows = [
        (
            str(item.get("doc_id", "")),
            str(item.get("sample_index")),
            str(item.get("n_samples")),
        )
        for item in records
        if not item.get("error")
    ]
    actual = set(actual_rows)
    missing = expected - actual
    unexpected = actual - expected
    duplicates = len(actual_rows) - len(actual)
    if missing or unexpected or duplicates:
        logger.warning(
            "Result completeness check: missing=%d, unexpected=%d, duplicates=%d; "
            "evaluation will continue",
            len(missing),
            len(unexpected),
            duplicates,
        )


def _iter_resume_records(
    path: str | Path,
    *,
    repair_truncated_last_line: bool = False,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield strict resume rows, optionally removing one truncated final row.

    When ``repair_truncated_last_line`` is requested, the file is opened
    read-write so truncation and newline repair can write back; if the file
    is not writable (read-only mount, permissions), the iterator degrades to
    read-only mode and the repair is skipped — a readable intact file must
    still resume normally.
    """
    resume_path = Path(path)
    mode = "r+b" if repair_truncated_last_line else "rb"
    can_repair = repair_truncated_last_line
    try:
        handle = resume_path.open(mode)
    except (OSError, PermissionError) as exc:
        if not repair_truncated_last_line:
            raise
        logger.warning(
            "Cannot open resume file %s read-write for repair (%s); "
            "continuing read-only — truncated-final-line repair will be skipped",
            resume_path,
            exc,
        )
        handle = resume_path.open("rb")
        can_repair = False
    with handle:
        line_num = 0
        line_start = handle.tell()
        raw_line = handle.readline()
        while raw_line:
            line_num += 1
            next_line = handle.readline()
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                is_truncated_final_line = (
                    can_repair and not next_line and not raw_line.endswith(b"\n")
                )
                if is_truncated_final_line:
                    handle.truncate(line_start)
                    logger.warning(
                        "Removed truncated final resume line in %s at line %d",
                        resume_path,
                        line_num,
                    )
                    return
                raise ValueError(
                    f"Invalid UTF-8 in resume file {resume_path} at line {line_num}"
                ) from exc
            if line.strip():
                try:
                    item = json.loads(line, parse_constant=reject_nonfinite_json)
                except json.JSONDecodeError as exc:
                    is_truncated_final_line = (
                        can_repair and not next_line and not raw_line.endswith(b"\n")
                    )
                    if is_truncated_final_line:
                        handle.truncate(line_start)
                        logger.warning(
                            "Removed truncated final resume line in %s at line %d",
                            resume_path,
                            line_num,
                        )
                        return
                    raise ValueError(
                        f"Invalid JSON in resume file {resume_path} at line "
                        f"{line_num}: {exc.msg}"
                    ) from exc
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid JSON in resume file {resume_path} at line "
                        f"{line_num}: {exc}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Resume file {resume_path} line {line_num} must contain an "
                        f"object, got {type(item).__name__}"
                    )
                if can_repair and not next_line and not raw_line.endswith(b"\n"):
                    handle.write(b"\n")
                    handle.flush()
                    logger.warning(
                        "Added missing final newline to resume file %s at line %d",
                        resume_path,
                        line_num,
                    )
                yield line_num, item
            elif can_repair and not next_line and not raw_line.endswith(b"\n"):
                handle.write(b"\n")
                handle.flush()
            line_start = handle.tell() - len(next_line)
            raw_line = next_line


# ---------------------------------------------------------------------------
# Resume state and request expansion
# ---------------------------------------------------------------------------


@dataclass
class ResumeState:
    """Completed sample indices and recorded prompts keyed by document ID."""

    completed_indices: dict[str, set[int]] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    n_samples_by_document: dict[str, int] = field(default_factory=dict)

    @property
    def completed_count(self) -> int:
        """Return the total number of completed samples represented by the state."""
        return sum(len(indices) for indices in self.completed_indices.values())


def run_concurrent_requests(
    items: Sequence[_T],
    worker: Callable[[_T], _R],
    on_success: Callable[[_R], None],
    *,
    max_workers: int,
    thread_name_prefix: str,
    description: str = "Processing samples",
) -> tuple[int, int]:
    """Run request workers and persist successes from the coordinator thread.

    Worker exceptions are counted as inference failures. Exceptions raised by
    ``on_success`` propagate because persistence failures invalidate the run.
    """
    processed = 0
    failed = 0
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=thread_name_prefix
    )
    futures = [executor.submit(worker, item) for item in items]
    try:
        with tqdm(total=len(items), desc=description, unit="sample") as progress:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    logger.warning("Inference sample failed: %s", exc)
                    failed += 1
                else:
                    on_success(result)
                    processed += 1
                finally:
                    progress.update(1)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return processed, failed


def derive_request_seed(
    base_seed: int,
    document_id: str,
    prompt: str,
    sample_index: int,
) -> int:
    """Derive a stable per-sample seed without storing private request fields."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")
    payload = f"{base_seed}\0{document_id}\0{prompt}\0{sample_index}".encode(
        "utf-8", errors="replace"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_resume_state(
    output_file: str | Path,
    input_key: str,
    response_key: str,
    *,
    repair_truncated_last_line: bool = False,
    expected_scoring_mode: str | None = None,
) -> ResumeState:
    """Load completed-row counts keyed by stable document ID."""
    if expected_scoring_mode is not None and (
        not isinstance(expected_scoring_mode, str) or not expected_scoring_mode.strip()
    ):
        raise ValueError("expected_scoring_mode must be a non-empty string")
    state = ResumeState()
    output_path = Path(output_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return state

    try:
        for line_num, item in _iter_resume_records(
            output_path,
            repair_truncated_last_line=repair_truncated_last_line,
        ):
            # Legacy failure rows remain retryable. Current inference paths do
            # not persist them.
            if item.get("error"):
                continue

            if (
                expected_scoring_mode is not None
                and item.get("scoring_mode") != expected_scoring_mode
            ):
                raise ValueError(
                    f"Resume file {output_path} line {line_num} has "
                    f"scoring_mode={item.get('scoring_mode')!r}, expected "
                    f"{expected_scoring_mode!r}; this row lacks a matching "
                    "scoring_mode (e.g. output written by an older version). "
                    "Use a new output file or migrate the row by adding "
                    f"scoring_mode={expected_scoring_mode!r}."
                )

            logprobs = item.get("logprobs")
            has_valid_logprobs = (
                isinstance(logprobs, list)
                and bool(logprobs)
                and any(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in logprobs
                )
                and all(
                    value is None
                    or (
                        isinstance(value, int | float)
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    )
                    for value in logprobs
                )
            )
            if not has_valid_logprobs:
                response = item.get(response_key)
                if response is None and response_key != "gen":
                    response = item.get("gen")
                if response is None:
                    continue
                if isinstance(response, list):
                    # An explicit empty list represents one empty model answer
                    # (same convention as resolve_single_generation) and is a
                    # completed sample; a multi-generation list is a legacy
                    # grouped row that must be migrated.
                    if len(response) == 0:
                        response = ""
                    elif len(response) != 1:
                        raise ValueError(
                            f"Resume file {output_path} line {line_num} must contain "
                            "exactly one generation; migrate grouped output to one "
                            "row per sample"
                        )
                    else:
                        response = response[0]
                if not isinstance(response, str):
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has an invalid "
                        f"{response_key!r} result"
                    )

            document_id = item.get("doc_id")
            if document_id is None or not str(document_id).strip():
                raise ValueError(
                    f"Resume file {output_path} line {line_num} is missing required "
                    "'doc_id'; migrate legacy resume output before continuing"
                )
            document_key = str(document_id)
            completed = state.completed_indices.setdefault(document_key, set())
            row_n_samples = item.get("n_samples")
            if row_n_samples is not None:
                if type(row_n_samples) is not int or row_n_samples <= 0:
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has invalid "
                        f"n_samples={row_n_samples!r}"
                    )
                previous_n_samples = state.n_samples_by_document.setdefault(
                    document_key, row_n_samples
                )
                if previous_n_samples != row_n_samples:
                    raise ValueError(
                        f"Resume file {output_path} line {line_num} has conflicting "
                        f"n_samples for doc_id={document_key!r}"
                    )
            sample_index = item.get("sample_index")
            if sample_index is None:
                # Legacy rows did not persist sample identity. Assign the next
                # free index in file order so existing result files remain usable.
                sample_index = 0
                while sample_index in completed:
                    sample_index += 1
            if type(sample_index) is not int or sample_index < 0:
                raise ValueError(
                    f"Resume file {output_path} line {line_num} has invalid "
                    f"sample_index={sample_index!r}"
                )
            if sample_index in completed:
                raise ValueError(
                    f"Resume file {output_path} line {line_num} duplicates "
                    f"sample_index={sample_index} for doc_id={document_key!r}"
                )
            completed.add(sample_index)

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


def prepare_sample_requests(
    raw_data: list[dict[str, Any]],
    resume_state: ResumeState,
    input_key: str,
    n_samples: int,
    *,
    prompt_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Return one independent request row per unfinished sample."""
    if not input_key:
        raise ValueError("input_key must be non-empty")
    if type(n_samples) is not int or n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
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
        completed = resume_state.completed_indices.get(document_key, set())
        recorded_n_samples = resume_state.n_samples_by_document.get(document_key)
        if recorded_n_samples is not None and recorded_n_samples != n_samples:
            raise ValueError(
                f"Resume output records n_samples={recorded_n_samples} for "
                f"doc_id={document_key!r}, but n_samples={n_samples} was requested"
            )
        invalid_indices = sorted(index for index in completed if index >= n_samples)
        if invalid_indices:
            raise ValueError(
                f"Resume output contains sample indices {invalid_indices} for "
                f"doc_id={document_key!r}, outside requested n_samples={n_samples}"
            )

        for generation_ordinal in range(n_samples):
            if generation_ordinal in completed:
                continue
            item = copy.deepcopy(source)
            item["sample_index"] = generation_ordinal
            item["n_samples"] = n_samples
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
    if args.model_revision is not None:
        llm_kwargs["revision"] = args.model_revision
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
        if key == "extra_body" and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return replacement
        if isinstance(value, dict):
            return {str(name): redact(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(payload)
