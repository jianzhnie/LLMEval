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
import json
import os
from pathlib import Path
from typing import Any

from llmeval.utils.log import init_logger

__all__ = [
    "count_completed_samples",
    "expand_data_with_resume",
    "expand_group_for_sampling",
    "is_explicit_tool_choice",
    "load_jsonl",
    "prepare_data_with_resume",
    "sample_count_for_item",
    "save_failed_items",
]

logger = init_logger("inference_common")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a line-delimited JSON file, skipping blank lines.

    Args:
        path: Input JSONL file path.

    Returns:
        Parsed items, one per non-blank line.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If an input line is not valid JSON.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError as e:
        logger.critical(f"Input file not found: {path}, {e}")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Invalid JSON in input file: {e}")
        raise


def count_completed_samples(
    output_file: str | Path,
    input_key: str,
    response_key: str,
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
                    item: dict[str, Any] = json.loads(line.strip())
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


def expand_data_with_resume(
    raw_data: list[dict[str, Any]],
    completed_counts: dict[str, int],
    input_key: str,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Expand raw items into per-sample copies, minus already-completed ones.

    Each remaining copy is a deep copy so that per-sample mutation (appending
    to the gen list) never leaks across copies of the same raw item.

    Args:
        raw_data: Items loaded from the input file.
        completed_counts: Completed-sample count per prompt (resume state).
        input_key: Prompt field name (``"prompt"`` used as fallback).
        n_samples: Target number of samples per prompt.

    Returns:
        Expanded dataset holding only the samples still to process.
    """
    expanded_data: list[dict[str, Any]] = []
    skipped_items = 0

    for item in raw_data:
        prompt_val: Any = item.get(input_key) or item.get("prompt")
        prompt = str(prompt_val) if prompt_val is not None else ""

        if not prompt.strip():
            logger.warning(
                f"No valid prompt found under keys [{input_key!r}, 'prompt'] "
                f"for item with keys: {list(item.keys())}"
            )
            skipped_items += 1
            continue

        completed = completed_counts.get(prompt, 0)
        remaining = max(0, n_samples - completed)

        for _ in range(remaining):
            expanded_data.append(copy.deepcopy(item))

    if skipped_items > 0:
        logger.warning(f"Skipped {skipped_items} items due to missing or empty prompt")

    return expanded_data


def prepare_data_with_resume(
    raw_data: list[dict[str, Any]],
    completed_counts: dict[str, int],
    input_key: str,
    n_samples: int,
    sample_count_key: str = "n_samples",
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

    Returns:
        Prepared dataset holding only the prompts still to process.
    """
    prepared_data: list[dict[str, Any]] = []
    skipped_items = 0

    for item in raw_data:
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

        completed = completed_counts.get(prompt, 0)
        remaining = max(0, n_samples - completed)
        if remaining <= 0:
            continue

        prepared_item = copy.deepcopy(item)
        prepared_item[sample_count_key] = remaining
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
        sample_items.extend([item] * sample_count_for_item(item, sample_count_key))
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
        with open(failed_file, "w", encoding="utf-8") as f:
            for entry in failed_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Failed items saved to: {failed_file}")
    except OSError as e:
        logger.error(f"Failed to save failed items to file: {e}")
