"""Shared provenance and contamination helpers for task evaluation.

The evaluator consumes model-output JSONL files, so it does not have the same
request objects that lm-evaluation-harness logs.  These helpers add the same
kind of reproducibility anchors in a lightweight form:

* stable document / prompt / target hashes
* task version and git commit metadata in summary files
* optional exact prompt-contamination checks against a local reference file
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

__all__ = [
    "annotate_dataset_contamination",
    "build_run_provenance",
    "build_sample_provenance",
    "get_git_hash",
    "hash_evaluation_inputs",
    "hash_json",
    "hash_string",
    "load_contamination_sources",
]

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "input", "question", "problem", "text")
_TARGET_KEYS: tuple[str, ...] = ("answer", "label", "target", "gold")
_RUNTIME_FIELDS: set[str] = {
    "gen",
    "logprobs",
    "accuracy",
    "extracted_answer",
    "extracted_gold",
    "doc_hash",
    "prompt_hash",
    "target_hash",
    "contamination",
    "_llmeval_contamination",
    "_llmeval_group_id",
    "_llmeval_sample_index",
    "raw_gen",
    "filtered_gen",
    "filter_trace",
    "correct",
    "correct_norm",
    "correct_bytes",
    "pred",
    "passed",
    "result",
    "stderr",
}
_CONTAMINATION_FIELD = "_llmeval_contamination"


def hash_string(text: Any) -> str:
    """Return a stable SHA-256 hash for a text value."""
    if text is None:
        text = ""
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def hash_json(value: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable value."""
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return hash_string(payload)


def hash_evaluation_inputs(
    eval_dataset: list[dict[str, Any]], response_key: str = "gen"
) -> str:
    """Hash scorer inputs while excluding fields produced during evaluation.

    The model response and MC logprobs remain part of the key. Runtime scoring
    artifacts are removed so scorers may annotate input records in place
    without changing the key for an otherwise identical repeated evaluation.
    """
    runtime_fields = set(_RUNTIME_FIELDS)
    runtime_fields.discard(response_key)
    runtime_fields.discard("logprobs")
    normalized = [
        {key: value for key, value in item.items() if key not in runtime_fields}
        for item in eval_dataset
    ]
    return hash_json(normalized)


def _first_present(
    item: dict[str, Any], preferred_key: str, fallbacks: tuple[str, ...]
) -> Any:
    keys = (preferred_key, *fallbacks) if preferred_key else fallbacks
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _doc_for_hash(item: dict[str, Any], response_key: str) -> dict[str, Any]:
    runtime_fields = set(_RUNTIME_FIELDS)
    if response_key:
        runtime_fields.add(response_key)
    return {key: value for key, value in item.items() if key not in runtime_fields}


def build_sample_provenance(
    item: dict[str, Any],
    input_key: str = "prompt",
    label_key: str = "answer",
    response_key: str = "gen",
) -> dict[str, Any]:
    """Build per-sample hashes and contamination status for a cache record."""
    prompt = _first_present(item, input_key, _PROMPT_KEYS)
    target = _first_present(item, label_key, _TARGET_KEYS)
    provenance: dict[str, Any] = {
        "doc_hash": hash_json(_doc_for_hash(item, response_key)),
        "prompt_hash": hash_string(prompt),
        "target_hash": hash_string(target),
    }

    contamination = item.get(_CONTAMINATION_FIELD)
    if isinstance(contamination, dict):
        provenance["contamination"] = contamination
    return provenance


def _get_git_hash(repo_path: str | Path | None = None) -> str | None:
    cwd = Path(repo_path or os.getcwd())
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = commit_result.stdout.strip()
    if not commit:
        return None

    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return commit

    status = status_result.stdout
    if not status:
        return commit

    # ``git describe --dirty`` only yields the same ``-dirty`` suffix for all
    # worktree edits. Hash the actual tracked diff and untracked file contents
    # so evaluation caches cannot survive a scorer change in a dirty checkout.
    try:
        diff_result = subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            cwd=cwd,
            check=True,
            capture_output=True,
        )
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = Path(root_result.stdout.strip())
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return f"{commit}-dirty-{hash_string(status)[:16]}"

    digest = hashlib.sha256()
    digest.update(status.encode("utf-8", errors="replace"))
    digest.update(diff_result.stdout)
    for raw_path in untracked_result.stdout.split(b"\0"):
        if not raw_path:
            continue
        digest.update(raw_path)
        path = repo_root / os.fsdecode(raw_path)
        try:
            if path.is_file():
                digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return f"{commit}-dirty-{digest.hexdigest()[:16]}"


def get_git_hash(repo_path: str | Path | None = None) -> str | None:
    """Return the current repository revision used for cache provenance."""
    return _get_git_hash(repo_path)


def _resolve_task_name(
    eval_dataset: list[dict[str, Any]], task_name: str | None
) -> str | None:
    if task_name:
        return task_name
    task_names = {
        str(item.get("task"))
        for item in eval_dataset
        if isinstance(item, dict) and item.get("task") is not None
    }
    if len(task_names) == 1:
        return next(iter(task_names))
    return None


def _resolve_task_version(eval_dataset: list[dict[str, Any]]) -> str:
    versions: set[str] = set()
    for item in eval_dataset:
        if not isinstance(item, dict):
            continue
        value = item.get("task_version", item.get("version"))
        metadata = item.get("metadata")
        if value is None and isinstance(metadata, dict):
            value = metadata.get("version")
        if value is not None:
            versions.add(str(value))

    if not versions:
        return "N/A"
    if len(versions) == 1:
        return next(iter(versions))
    return "mixed:" + ",".join(sorted(versions))


def _contamination_summary(eval_dataset: list[dict[str, Any]]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = [
        item[_CONTAMINATION_FIELD]
        for item in eval_dataset
        if isinstance(item, dict) and isinstance(item.get(_CONTAMINATION_FIELD), dict)
    ]
    if not checked:
        return {
            "checked": False,
            "total": len(eval_dataset),
            "contaminated": 0,
        }

    contaminated = sum(1 for record in checked if record.get("contaminated"))
    return {
        "checked": True,
        "total": len(checked),
        "contaminated": contaminated,
        "clean": len(checked) - contaminated,
    }


def build_run_provenance(
    eval_dataset: list[dict[str, Any]],
    task_name: str | None = None,
    input_key: str = "prompt",
    label_key: str = "answer",
    response_key: str = "gen",
    seed: int | None = None,
) -> dict[str, Any]:
    """Build run-level provenance metadata for a scorer summary file."""
    samples = [
        build_sample_provenance(item, input_key, label_key, response_key)
        for item in eval_dataset
        if isinstance(item, dict)
    ]
    prompt_hashes = [sample["prompt_hash"] for sample in samples]
    target_hashes = [sample["target_hash"] for sample in samples]
    return {
        "schema_version": 1,
        "task_name": _resolve_task_name(eval_dataset, task_name),
        "task_version": _resolve_task_version(eval_dataset),
        "git_hash": _get_git_hash(),
        "date": time.time(),
        "seed": seed,
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "fewshot_seed": seed,
        "generation_seed": seed,
        "num_records": len(eval_dataset),
        "prompt_hash": hash_json(prompt_hashes),
        "target_hash": hash_json(target_hashes),
        "dataset_hash": hash_json(samples),
        "contamination": _contamination_summary(eval_dataset),
    }


def _normalize_contamination_text(value: Any) -> str:
    return " ".join(str(value).lower().split())


def load_contamination_sources(path: str | Path) -> list[str]:
    """Load exact contamination reference strings from JSONL or plain text."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"contamination_path does not exist: {source_path}")

    sources: list[str] = []
    with open(source_path, encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                sources.append(text)
                continue

            if isinstance(parsed, dict):
                value = _first_present(parsed, "prompt", _PROMPT_KEYS)
                if value is not None:
                    sources.append(str(value))
            else:
                sources.append(str(parsed))
    return sources


def annotate_dataset_contamination(
    eval_dataset: list[dict[str, Any]],
    contamination_sources: list[str],
    input_key: str = "prompt",
    min_length: int = 32,
) -> None:
    """Annotate items with exact prompt-contamination matches.

    The check is intentionally conservative: a prompt is contaminated only when
    the normalized prompt and a normalized reference string contain each other,
    and the query length passes ``min_length``.
    """
    normalized_sources = [
        source
        for source in (
            _normalize_contamination_text(text) for text in contamination_sources
        )
        if len(source) >= min_length
    ]

    for item in eval_dataset:
        if not isinstance(item, dict):
            continue
        prompt = _first_present(item, input_key, _PROMPT_KEYS)
        query = _normalize_contamination_text(prompt)
        match_hash = None
        contaminated = False
        if len(query) >= min_length:
            for source in normalized_sources:
                if query in source or source in query:
                    contaminated = True
                    match_hash = hash_string(source)
                    break

        item[_CONTAMINATION_FIELD] = {
            "checked": True,
            "contaminated": contaminated,
            "query_hash": hash_string(query),
            "match_hash": match_hash,
        }
