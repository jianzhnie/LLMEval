"""Shared data-loading and resume helpers for the inference runners.

The online, offline, verifier, and MC runners all implement the same pipeline
stages: load JSONL input, count completed samples in the output file for
resume, expand the dataset to ``n_samples`` copies per prompt, and persist
failed items for debugging.  Those helpers live here so each runner module
stays focused on its own backend (OpenAI API vs. vLLM engine).

Functions
---------
load_jsonl                 — parse a line-delimited JSON file
count_completed_samples    — per-prompt completed-sample counts for resume
expand_data_with_resume    — expand raw items to remaining per-sample copies
prepare_data_with_resume   — attach remaining sample counts for batched online runs
sample_count_for_item      — read the runtime sample count from an item
expand_group_for_sampling   — expand grouped prompt records by sample count
save_failed_items          — persist failure records next to the output file
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from llmeval.utils.log import init_logger

__all__ = [
    "completed_sample_indices_by_identity",
    "count_completed_samples",
    "count_completed_samples_by_id",
    "count_completed_samples_by_identity",
    "expand_data_with_resume",
    "expand_group_for_sampling",
    "is_explicit_tool_choice",
    "load_jsonl",
    "prepare_data_with_resume",
    "require_document_id",
    "sample_count_for_item",
    "sample_seed_for_item",
    "save_failed_items",
    "validate_document_ids",
]

logger = init_logger("inference_common")


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
    sample_index = item.get("_llmeval_sample_index", 0)
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


def count_completed_samples(
    output_file: str | Path,
    input_key: str,
    response_key: str,
    *,
    legacy_only: bool = False,
) -> dict[str, int]:
    """Count completed samples per prompt for resume.

    Scans the output JSONL and sums how many generations each prompt already
    has, so an interrupted run can regenerate only the missing samples.
    Malformed lines are skipped with a warning; a missing or empty output
    file yields empty counts.

    Args:
        output_file: Output JSONL from a previous (interrupted) run.
        input_key: Prompt field name (``"prompt"`` used as fallback).
        response_key: Generation-list field name (``"gen"`` used as fallback).
        legacy_only: Ignore records carrying ``doc_id``. This is used
            when stable-ID and legacy records coexist in one output file.

    Returns:
        Mapping of prompt text to its completed-sample count.
    """
    completed_counts: dict[str, int] = collections.defaultdict(int)

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return completed_counts

    try:
        with open(output_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if not isinstance(item, dict):
                        logger.warning(
                            "Skipping non-object JSON on output line %d: %s",
                            line_num,
                            type(item).__name__,
                        )
                        continue
                    if legacy_only and item.get("doc_id"):
                        continue
                    prompt: Any = item.get(input_key) or item.get("prompt")
                    gen_response = item.get(response_key) or item.get("gen")
                    # Guard against a null / non-list gen field (e.g. a
                    # partially-written line): len(None) would raise here.
                    gen_count = (
                        len(gen_response) if isinstance(gen_response, list) else 0
                    )
                    if prompt is not None:
                        completed_counts[str(prompt)] += gen_count
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
    except Exception as e:
        logger.error(f"Error reading output file for resume check: {e}")

    return completed_counts


def count_completed_samples_by_id(
    output_file: str | Path,
    response_key: str,
    id_key: str = "doc_id",
) -> dict[str, int]:
    """Count completed generations by stable document ID.

    New output records carry ``doc_id``. A record with a generation
    list contributes its list length; a verifier-style record with a scalar
    response contributes one.  Records without the ID are intentionally
    ignored so callers can explicitly choose a legacy prompt-based fallback.
    """
    completed_counts: dict[str, int] = collections.defaultdict(int)
    output_path = Path(output_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return completed_counts

    try:
        with open(output_path, encoding="utf-8") as handle:
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

                document_id = item.get(id_key)
                if not document_id:
                    continue
                response = item.get(response_key) or item.get("gen")
                if isinstance(response, list):
                    count = len(response)
                elif response is not None:
                    count = 1
                elif item.get("logprobs") is not None or item.get("Verifier_judgment"):
                    # One loglikelihood/verifier record represents one sample;
                    # the logprobs list contains choices, not generations.
                    count = 1
                else:
                    count = 0
                if count:
                    completed_counts[str(document_id)] += count
    except OSError as exc:
        logger.error("Error reading stable resume state from %s: %s", output_file, exc)

    return completed_counts


def count_completed_samples_by_identity(
    output_file: str | Path,
    input_key: str,
    response_key: str,
) -> dict[tuple[str, str], int]:
    """Count completions by prepared ``doc_id`` and the rendered prompt.

    ``doc_id`` remains the persistent question identifier. Including the
    prompt in resume state prevents stale generations from being reused after
    a prompt template or few-shot prefix changes.
    """
    return {
        identity: len(indices)
        for identity, indices in completed_sample_indices_by_identity(
            output_file, input_key, response_key
        ).items()
    }


def completed_sample_indices_by_identity(
    output_file: str | Path,
    input_key: str,
    response_key: str,
) -> dict[tuple[str, str], set[int]]:
    """Return completed sample indices for each stable document and prompt.

    Explicit sample indices are deduplicated across resumed output rows. Older
    rows without indices are assigned the next unused positions in file order.
    """
    completed: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    output_path = Path(output_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return completed

    try:
        with open(output_path, encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON on output line %d: %s", line_num, exc)
                    continue
                if not isinstance(item, dict):
                    continue
                document_id = item.get("doc_id")
                prompt = item.get(input_key) or item.get("prompt")
                if not document_id or prompt is None:
                    continue

                response = item.get(response_key)
                if response is None:
                    response = item.get("gen")
                if isinstance(response, list):
                    count = len(response)
                elif (
                    response is not None
                    or item.get("logprobs") is not None
                    or item.get("Verifier_response")
                ):
                    count = 1
                else:
                    count = 0
                if not count:
                    continue

                identity = (str(document_id), str(prompt))
                raw_indices = item.get("_llmeval_sample_indices")
                if (
                    isinstance(raw_indices, list)
                    and len(raw_indices) == count
                    and all(
                        isinstance(index, int) and index >= 0 for index in raw_indices
                    )
                ):
                    indices = raw_indices
                elif isinstance(item.get("_llmeval_sample_index"), int):
                    indices = [int(item["_llmeval_sample_index"])]
                else:
                    indices = []
                    next_index = 0
                    for _ in range(count):
                        while next_index in completed[identity]:
                            next_index += 1
                        indices.append(next_index)
                        next_index += 1
                completed[identity].update(indices)
    except OSError as exc:
        logger.error(
            "Error reading resume identity state from %s: %s", output_file, exc
        )

    return completed


def expand_data_with_resume(
    raw_data: list[dict[str, Any]],
    completed_counts: dict[object, int],
    input_key: str,
    n_samples: int,
    stable_ids: bool = False,
) -> list[dict[str, Any]]:
    """Expand raw items into per-sample copies, minus already-completed ones.

    Each remaining copy is a deep copy so that per-sample mutation (appending
    to the gen list) never leaks across copies of the same raw item.

    Args:
        raw_data: Items loaded from the input file.
        completed_counts: Completed-sample count per prompt (resume state).
        input_key: Prompt field name (``"prompt"`` used as fallback).
        n_samples: Target number of samples per prompt.
        stable_ids: Require dataset-provided ``doc_id`` and use it as the resume key.

    Returns:
        Expanded dataset holding only the samples still to process.
    """
    expanded_data: list[dict[str, Any]] = []
    skipped_items = 0

    if stable_ids:
        validate_document_ids(raw_data)

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

        document_id = require_document_id(item, index) if stable_ids else prompt
        resume_key: object = (document_id, prompt) if stable_ids else document_id
        completed = completed_counts.get(resume_key, 0)
        if stable_ids and completed == 0:
            # A prompt-keyed entry is only present when the caller detected
            # legacy output.  Stable IDs remain the primary identity.
            completed = completed_counts.get(prompt, 0)
        for sample_index in range(completed, n_samples):
            expanded_item = copy.deepcopy(item)
            if stable_ids:
                expanded_item["_llmeval_sample_index"] = sample_index
            expanded_data.append(expanded_item)

    if skipped_items > 0:
        logger.warning(f"Skipped {skipped_items} items due to missing or empty prompt")

    return expanded_data


def prepare_data_with_resume(
    raw_data: list[dict[str, Any]],
    completed_counts: dict[object, int],
    input_key: str,
    n_samples: int,
    sample_count_key: str = "n_samples",
    stable_ids: bool = False,
) -> list[dict[str, Any]]:
    """Prepare one prompt record per item with a remaining sample count.

    This variant is used by online inference, where the remaining sample count
    is passed directly to the API as ``n`` instead of expanding to repeated
    copies of the same prompt.

    Args:
        raw_data: Items loaded from the input file.
        completed_counts: Completed-sample count per prompt (resume state).
        input_key: Prompt field name (``"prompt"`` used as fallback).
        n_samples: Target number of samples per prompt.
        sample_count_key: Output field name used to store the remaining sample
            count for the prompt.
        stable_ids: Require dataset-provided ``doc_id`` and use it as the resume key.

    Returns:
        Prepared dataset holding only the prompts still to process.

    Raises:
        ValueError: If ``input_key`` is empty or ``n_samples`` is not positive.
    """
    if not input_key:
        raise ValueError("input_key must be non-empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    prepared_data: list[dict[str, Any]] = []
    skipped_items = 0

    if stable_ids:
        validate_document_ids(raw_data)

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict input item: {type(item)}")
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

        document_id = require_document_id(item, index) if stable_ids else prompt
        resume_key: object = (document_id, prompt) if stable_ids else document_id
        completed = completed_counts.get(resume_key, 0)
        if stable_ids and completed == 0:
            completed = completed_counts.get(prompt, 0)
        remaining = max(0, n_samples - completed)
        if remaining <= 0:
            continue

        prepared_item = copy.deepcopy(item)
        prepared_item[sample_count_key] = remaining
        if stable_ids:
            prepared_item["_llmeval_sample_start"] = completed
        prepared_data.append(prepared_item)

    if skipped_items > 0:
        logger.warning(f"Skipped {skipped_items} items due to missing or empty prompt")

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
        sample_start = int(item.get("_llmeval_sample_start", 0))
        for offset in range(sample_count):
            sample_item = item.copy()
            if "doc_id" in item:
                sample_item["_llmeval_sample_index"] = sample_start + offset
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

    Writes to ``<output_stem>_failed.jsonl``.  NOTE: splitext, not
    str.replace — a non-.jsonl output name must not collapse onto the output
    file itself ("w" mode would truncate it).

    Args:
        output_file: The run's output file (derives the failed-file path).
        failed_items: Records describing each failure (item + error).
    """
    failed_file = os.path.splitext(str(output_file))[0] + "_failed.jsonl"
    try:
        Path(failed_file).parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, "w", encoding="utf-8") as f:
            for entry in failed_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Failed items saved to: {failed_file}")
    except OSError as e:
        logger.error(f"Failed to save failed items to file: {e}")
