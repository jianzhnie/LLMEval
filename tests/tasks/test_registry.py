"""Regression tests for P1 registry and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmeval.tasks.registry import (
    CodeTask,
    EvaluationContext,
    EvaluationResult,
    MathTask,
    MCTask,
    MetricValue,
    ScorerResult,
    TaskRegistry,
    metric_from_samples,
    persist_evaluation_result,
)


def test_evaluation_result_default_serialization_is_compact(tmp_path: Path) -> None:
    result = EvaluationResult(
        task_name="mc_opensource/mmlu",
        task_version="mc_v1",
        metrics={"acc": MetricValue(1.0, 1)},
        sample_count=1,
        effective_sample_count=1,
        per_item=[{"prompt": "large generated content"}],
    )

    assert "per_item" not in result.to_dict()
    assert result.to_dict(include_per_item=True)["per_item"] == result.per_item

    persist_evaluation_result(result, tmp_path / "score.jsonl")
    summary = json.loads((tmp_path / "score.summary.json").read_text())
    assert "per_item" not in summary
    assert summary["acc"] == 1.0


def test_per_item_output_supports_compact_and_debug_schemas(tmp_path: Path) -> None:
    record = {
        "doc_id": "mmlu:0",
        "prompt": "large prompt",
        "gold": 1,
        "pred": 1,
        "correct": True,
        "score": 1.0,
        "scoring_mode": "first_token",
        "raw_gen": "large response",
        "logprobs": [-2.0, -0.1],
        "filter_trace": {"pipeline": "mc"},
        "unused": None,
    }
    result = EvaluationResult(
        task_name="mc_opensource/mmlu",
        task_version="mc_v1",
        metrics={"acc": MetricValue(1.0, 1)},
        sample_count=1,
        effective_sample_count=1,
        per_item=[record],
    )

    compact_path = tmp_path / "compact.jsonl"
    persist_evaluation_result(result, compact_path)
    compact = json.loads(compact_path.read_text())
    assert compact == {
        "doc_id": "mmlu:0",
        "gold": 1,
        "pred": 1,
        "correct": True,
        "score": 1.0,
        "scoring_mode": "first_token",
    }

    debug_path = tmp_path / "debug.jsonl"
    persist_evaluation_result(result, debug_path, output_schema="debug")
    assert json.loads(debug_path.read_text()) == record


def test_bootstrap_is_deterministic() -> None:
    samples = [0.0, 1.0, 1.0, 0.0]
    first = metric_from_samples(samples, 7, n_resamples=100)
    second = metric_from_samples(samples, 7, n_resamples=100)
    different = metric_from_samples(samples, 8, n_resamples=100)
    assert first == second
    assert first.value == pytest.approx(0.5)
    assert first.count == 4
    assert (first.stderr, first.ci_low, first.ci_high) != (
        different.stderr,
        different.ci_low,
        different.ci_high,
    )


def test_registry_reports_registered_families() -> None:
    task = MathTask(
        lambda **_: ScorerResult(
            metrics={"accuracy": 1.0},
            observations={"accuracy": [1.0]},
            sample_count=1,
            effective_sample_count=1,
        )
    )
    registry = TaskRegistry({"math_opensource": task})
    assert registry.names == ("math_opensource",)
    with pytest.raises(ValueError, match="registered tasks"):
        registry.resolve("unsupported/task")


@pytest.mark.parametrize("aggregation", ["first", "majority_vote", "any_correct"])
def test_mc_registry_uses_one_observation_for_item_aggregation(
    tmp_path: Path, aggregation: str
) -> None:
    def scorer(**kwargs: object) -> ScorerResult:
        del kwargs
        return ScorerResult(
            metrics={
                "acc": 1.0,
                "acc_norm": 1.0,
                "acc_bytes": 1.0,
                "exact_match": 1.0,
            },
            observations={
                "acc": [1.0],
                "acc_norm": [1.0],
                "acc_bytes": [1.0],
                "exact_match": [1.0],
            },
            per_item=[{"correct": True, "aggregation": aggregation}],
            sample_count=1,
            effective_sample_count=1,
        )

    context = EvaluationContext(
        eval_dataset=[{"answer": "A", "gen": ["A", "B"]}],
        task_name="mc_opensource/test",
        label_key="answer",
        response_key="gen",
        cache_path=tmp_path / f"{aggregation}.jsonl",
        max_workers=1,
        timeout=1,
        exec_timeout=1.0,
        seed=1,
        mc_aggregation=aggregation,
        bootstrap_samples=10,
    )

    result = MCTask(scorer, scorer).score(context)

    assert result.metrics["acc"].value == 1.0
    assert result.metrics["acc"].count == 1


def test_code_registry_preserves_scorer_failure_classification(tmp_path: Path) -> None:
    def scorer(**_: object) -> ScorerResult:
        return ScorerResult(
            metrics={"pass@1": 0.0},
            observations={"pass@1": [0.0]},
            per_item=[{"passed": False, "result": "failed: AssertionError"}],
            sample_count=1,
            effective_sample_count=1,
            failed_count=0,
        )

    context = EvaluationContext(
        eval_dataset=[{"answer": "tests", "gen": ["wrong"]}],
        task_name="code_opensource/test",
        label_key="answer",
        response_key="gen",
        cache_path=tmp_path / "code.jsonl",
        max_workers=1,
        timeout=1,
        exec_timeout=1.0,
        seed=1,
        allow_unsafe_code=True,
    )

    result = CodeTask(scorer).score(context)

    assert result.metrics["pass@1"].value == 0.0
    assert result.failed_count == 0
    assert result.effective_sample_count == 1
