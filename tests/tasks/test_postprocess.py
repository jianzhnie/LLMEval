"""Tests for llmeval.tasks.postprocess."""

from __future__ import annotations

import pytest

from llmeval.tasks.postprocess import (
    FilterRegistry,
    normalize_single_generation_samples,
    resolve_max_workers,
    strip_reasoning_wrappers,
)


def test_strip_reasoning_wrappers_prefers_answer_tag() -> None:
    text = "<think>reasoning</think><answer> 42 </answer> tail"
    assert strip_reasoning_wrappers(text) == "42"


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
    registry = FilterRegistry()
    registry.register("strip", str.strip, version="2")
    registry.register("upper", str.upper, version="1")
    pipeline = registry.build_pipeline("test_pipeline", "3", "strip", "upper")

    output, trace = pipeline.apply_with_trace(" a ")

    assert output == "A"
    assert trace["pipeline"] == "test_pipeline"
    assert trace["pipeline_version"] == "3"
    assert [step["name"] for step in trace["filters"]] == ["strip", "upper"]
    assert trace["filters"][0]["input"] == " a "
    assert trace["filters"][1]["output"] == "A"


def test_resolve_max_workers_respects_cpu_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llmeval.tasks.postprocess.os.cpu_count", lambda: 8)

    assert resolve_max_workers(total=20, requested=12) == 7
    assert resolve_max_workers(total=3, requested=12) == 3


def test_pipeline_rejects_unknown_filters() -> None:
    with pytest.raises(ValueError, match="Unknown text filter"):
        FilterRegistry().build_pipeline("test", "1", "missing")
