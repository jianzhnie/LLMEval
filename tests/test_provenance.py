"""Tests for task provenance and contamination helpers."""

from __future__ import annotations

from llmeval.tasks.provenance import (
    annotate_dataset_contamination,
    build_run_provenance,
    build_sample_provenance,
)


def test_sample_provenance_hashes_prompt_target_and_excludes_response() -> None:
    item = {"prompt": "What is 2+2?", "answer": "4", "gen": ["4"]}
    changed_response = {"prompt": "What is 2+2?", "answer": "4", "gen": ["5"]}

    provenance = build_sample_provenance(item)
    changed_provenance = build_sample_provenance(changed_response)

    assert provenance["prompt_hash"]
    assert provenance["target_hash"]
    assert provenance["doc_hash"] == changed_provenance["doc_hash"]


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
