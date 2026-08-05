"""Tests for task provenance and contamination helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from llmeval.tasks.provenance import (
    annotate_dataset_contamination,
    build_run_provenance,
    build_sample_provenance,
    get_git_hash,
    hash_evaluation_inputs,
)


def test_sample_provenance_hashes_prompt_target_and_excludes_response() -> None:
    item = {"prompt": "What is 2+2?", "answer": "4", "gen": ["4"]}
    changed_response = {"prompt": "What is 2+2?", "answer": "4", "gen": ["5"]}

    provenance = build_sample_provenance(item)
    changed_provenance = build_sample_provenance(changed_response)

    assert provenance["prompt_hash"]
    assert provenance["target_hash"]
    assert provenance["doc_hash"] == changed_provenance["doc_hash"]


def test_evaluation_input_hash_ignores_scorer_annotations() -> None:
    item = {"doc_id": "math:1", "prompt": "2+2", "answer": "4", "gen": ["4"]}
    original_hash = hash_evaluation_inputs([item])

    item.update(
        {
            "raw_gen": "4",
            "filtered_gen": "4",
            "filter_trace": [{"filter": "strip", "output": "4"}],
            "evaluation_status": "completed",
            "sample_correct": [True],
            "sample_count": 1,
            "effective_sample_count": 1,
            "inference_provenance": {"seed": 7},
        }
    )

    assert hash_evaluation_inputs([item]) == original_hash


def test_run_provenance_records_task_version_and_seed() -> None:
    data = [
        {
            "task": "math_opensource/aime24",
            "task_version": "2026.08",
            "prompt": "Compute 1+1",
            "answer": "2",
            "gen": ["2"],
        }
    ]

    provenance = build_run_provenance(
        data,
        task_name="math_opensource/aime24",
        seed=123,
    )

    assert provenance["schema_version"] == 1
    assert provenance["task_name"] == "math_opensource/aime24"
    assert provenance["task_version"] == "2026.08"
    assert provenance["seed"] == 123
    assert provenance["contamination"]["checked"] is False


def test_contamination_annotation_flags_exact_prompt_overlap() -> None:
    prompt = "This is a deliberately long benchmark prompt for overlap checking."
    data = [{"prompt": prompt, "answer": "A"}]

    annotate_dataset_contamination(data, [f"prefix {prompt} suffix"])
    provenance = build_sample_provenance(data[0])

    assert provenance["contamination"]["checked"] is True
    assert provenance["contamination"]["contaminated"] is True
    assert provenance["contamination"]["match_hash"]


def test_git_hash_changes_when_dirty_worktree_content_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "LLMEval Test"], cwd=tmp_path, check=True
    )
    source = tmp_path / "scorer.py"
    source.write_text("return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "scorer.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    clean = get_git_hash(tmp_path)
    source.write_text("return 2\n", encoding="utf-8")
    first_dirty = get_git_hash(tmp_path)
    source.write_text("return 3\n", encoding="utf-8")
    second_dirty = get_git_hash(tmp_path)

    assert clean is not None
    assert first_dirty is not None and first_dirty.startswith(f"{clean}-dirty-")
    assert second_dirty is not None and second_dirty.startswith(f"{clean}-dirty-")
    assert first_dirty != second_dirty
