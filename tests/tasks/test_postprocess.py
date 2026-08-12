"""Tests for llmeval.tasks.postprocess."""

from __future__ import annotations

import pytest

from llmeval.tasks.postprocess import (
    TextFilterPipeline,
    normalize_single_generation_samples,
    resolve_max_workers,
    strip_reasoning_wrappers,
)


def test_strip_reasoning_wrappers_prefers_answer_tag() -> None:
    text = "<think>reasoning</think><answer> 42 </answer> tail"
    assert strip_reasoning_wrappers(text) == "42"


def test_strip_reasoning_wrappers_uses_final_answer_tag() -> None:
    text = "<answer>A</answer> correction <answer>B</answer>"
    assert strip_reasoning_wrappers(text) == "B"


def test_strip_reasoning_wrappers_uses_final_think_block() -> None:
    text = "<think>x</think> middle <think>y</think> FINAL"
    assert strip_reasoning_wrappers(text) == "FINAL"


def test_normalize_repeated_samples_preserves_identical_responses() -> None:
    rows = [
        {"doc_id": "d0", "answer": "1", "gen": ["a"]},
        {"doc_id": "d0", "answer": "1", "gen": ["b"]},
        {"doc_id": "d0", "answer": "1", "gen": ["a"]},
    ]
    normalized = normalize_single_generation_samples(
        rows,
        "gen",
        problem_identity=lambda item, _index: str(item["doc_id"]),
        conflict_keys=("answer",),
        record_kind="document",
    )
    assert [row["gen"] for row in normalized] == [["a"], ["b"], ["a"]]


def test_normalize_empty_generation_as_one_empty_sample() -> None:
    normalized = normalize_single_generation_samples(
        [{"doc_id": "d0", "gen": []}],
        "gen",
        problem_identity=lambda item, _index: str(item["doc_id"]),
    )

    assert normalized[0]["gen"] == [""]


def test_normalize_repeated_samples_rejects_conflicts() -> None:
    rows = [
        {"doc_id": "d0", "answer": "1", "gen": ["a"]},
        {"doc_id": "d0", "answer": "2", "gen": ["b"]},
    ]
    with pytest.raises(ValueError, match="Conflicting 'answer'"):
        normalize_single_generation_samples(
            rows,
            "gen",
            problem_identity=lambda item, _index: str(item["doc_id"]),
            conflict_keys=("answer",),
            record_kind="document",
        )


def test_normalize_repeated_samples_orders_and_validates_sample_indices() -> None:
    rows = [
        {"doc_id": "d0", "answer": "1", "gen": ["second"], "sample_index": 1},
        {"doc_id": "d1", "answer": "2", "gen": ["only"], "sample_index": 0},
        {"doc_id": "d0", "answer": "1", "gen": ["first"], "sample_index": 0},
    ]
    normalized = normalize_single_generation_samples(
        rows,
        "gen",
        problem_identity=lambda item, _index: str(item["doc_id"]),
        conflict_keys=("answer",),
    )

    assert [row["gen"] for row in normalized] == [["first"], ["only"], ["second"]]

    with pytest.raises(ValueError, match="Duplicate sample_index 0"):
        normalize_single_generation_samples(
            [
                {"doc_id": "d0", "gen": ["a"], "sample_index": 0},
                {"doc_id": "d0", "gen": ["b"], "sample_index": 0},
            ],
            "gen",
            problem_identity=lambda item, _index: str(item["doc_id"]),
        )


def test_registered_pipeline_records_each_filter_step() -> None:
    pipeline = TextFilterPipeline(
        "test_pipeline", (("strip", str.strip), ("upper", str.upper))
    )

    output, trace = pipeline.apply_with_trace(" a ")

    assert output == "A"
    assert trace["pipeline"] == "test_pipeline"
    assert "pipeline_version" not in trace
    assert [step["name"] for step in trace["filters"]] == ["strip", "upper"]
    assert trace["filters"][0] == {
        "name": "strip",
        "changed": True,
        "input_length": 3,
        "output_length": 1,
    }
    assert trace["filters"][1]["output_length"] == 1
    assert "raw" not in trace
    assert "output" not in trace


def test_resolve_max_workers_respects_cpu_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llmeval.tasks.postprocess.os.cpu_count", lambda: 8)

    assert resolve_max_workers(total=20, requested=12) == 7
    assert resolve_max_workers(total=3, requested=12) == 3


def test_normalize_rejects_missing_generation() -> None:
    normalized = normalize_single_generation_samples(
        [{"doc_id": "d0"}],
        "gen",
        problem_identity=lambda item, _index: str(item["doc_id"]),
    )

    assert "gen" not in normalized[0]


def test_normalize_rejects_non_contiguous_sample_indices() -> None:
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        normalize_single_generation_samples(
            [
                {"doc_id": "d0", "gen": ["a"], "sample_index": 0},
                {"doc_id": "d0", "gen": ["b"], "sample_index": 2},
            ],
            "gen",
            problem_identity=lambda item, _index: str(item["doc_id"]),
        )


def test_normalize_uses_sample_count_metadata() -> None:
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        normalize_single_generation_samples(
            [
                {
                    "doc_id": "d0",
                    "gen": ["a"],
                    "sample_index": 0,
                    "n_samples": 2,
                }
            ],
            "gen",
            problem_identity=lambda item, _index: str(item["doc_id"]),
        )
