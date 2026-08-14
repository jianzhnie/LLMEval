"""Regression tests for P1 registry and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

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


def test_evaluation_result_summary_excludes_records(tmp_path: Path) -> None:
    result = EvaluationResult(
        task_name="mc_opensource/mmlu",
        metrics={"acc": MetricValue(1.0, 1)},
        sample_count=2,
        effective_sample_count=1,
        excluded_count=1,
        records=[{"prompt": "large generated content"}],
    )

    assert "records" not in result.to_dict()

    persist_evaluation_result(result, tmp_path / "score.json")
    summary = json.loads((tmp_path / "score.json").read_text())
    assert "records" not in summary
    assert not any("version" in key for key in summary)
    assert summary["metrics"]["acc"]["value"] == 1.0
    assert summary["excluded_count"] == 1
    assert "metric_values" not in summary


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


@pytest.mark.parametrize("n_resamples", [0, 1])
def test_disabled_bootstrap_omits_uncertainty(n_resamples: int) -> None:
    metric = metric_from_samples([0.0, 1.0], 7, n_resamples=n_resamples)

    assert metric.value == 0.5
    assert metric.stderr is None
    assert metric.ci_low is None
    assert metric.ci_high is None


@pytest.mark.parametrize("count", [1.5, True])
def test_scorer_result_rejects_non_integer_counts(count: object) -> None:
    with pytest.raises(TypeError, match="counts must be integers"):
        ScorerResult(
            metrics={"accuracy": 0.0},
            observations={"accuracy": []},
            sample_count=count,  # type: ignore[arg-type]
            effective_sample_count=count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("excluded_count", [1.5, True])
def test_scorer_result_rejects_non_integer_excluded_count(
    excluded_count: object,
) -> None:
    with pytest.raises(TypeError, match="counts must be integers"):
        ScorerResult(
            metrics={"accuracy": 0.0},
            observations={"accuracy": [0.0]},
            sample_count=1,
            effective_sample_count=1,
            excluded_count=excluded_count,  # type: ignore[arg-type]
        )


def test_scorer_result_accepts_failed_and_excluded_partition() -> None:
    result = ScorerResult(
        metrics={"accuracy": 1.0},
        observations={"accuracy": [1.0]},
        sample_count=3,
        effective_sample_count=1,
        failed_count=1,
        excluded_count=1,
    )

    assert result.effective_sample_count == 1


def test_scorer_result_rejects_negative_excluded_count() -> None:
    with pytest.raises(ValueError, match="counts must be non-negative"):
        ScorerResult(
            metrics={"accuracy": 0.0},
            observations={"accuracy": []},
            excluded_count=-1,
        )


def test_scorer_result_rejects_failed_and_excluded_above_sample_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed sample count"):
        ScorerResult(
            metrics={"accuracy": 0.0},
            observations={"accuracy": []},
            sample_count=1,
            effective_sample_count=0,
            failed_count=1,
            excluded_count=1,
        )


def test_scorer_result_requires_effective_count_to_exclude_all_non_metrics() -> None:
    with pytest.raises(ValueError, match="excluded_count"):
        ScorerResult(
            metrics={"accuracy": 0.0},
            observations={"accuracy": []},
            sample_count=2,
            effective_sample_count=2,
            excluded_count=1,
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
            metrics={"acc": 1.0},
            observations={"acc": [1.0]},
            records=[{"correct": True, "aggregation": aggregation}],
            sample_count=1,
            effective_sample_count=1,
        )

    context = EvaluationContext(
        eval_dataset=[{"answer": "A", "gen": ["A", "B"]}],
        task_name="mc_opensource/test",
        label_key="answer",
        response_key="gen",
        result_path=tmp_path / f"{aggregation}.json",
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


def test_mc_registry_rejects_non_list_logprobs() -> None:
    scorer = MagicMock()
    task = MCTask(scorer, scorer)

    with pytest.raises(ValueError, match="invalid logprobs"):
        task._mc_schema([{"logprobs": None}])


def test_code_registry_preserves_scorer_failure_classification(tmp_path: Path) -> None:
    def scorer(**_: object) -> ScorerResult:
        return ScorerResult(
            metrics={"pass@1": 0.0},
            observations={"pass@1": [0.0]},
            records=[{"passed": False, "result": "failed: AssertionError"}],
            sample_count=1,
            effective_sample_count=1,
            failed_count=0,
        )

    context = EvaluationContext(
        eval_dataset=[{"answer": "tests", "gen": ["wrong"]}],
        task_name="code_opensource/test",
        label_key="answer",
        response_key="gen",
        result_path=tmp_path / "code.json",
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


def test_code_registry_uses_problem_observations_for_metric_count(
    tmp_path: Path,
) -> None:
    def scorer(**_: object) -> ScorerResult:
        return ScorerResult(
            metrics={"pass@1": 1.0},
            observations={"pass@1": [1.0]},
            sample_count=64,
            effective_sample_count=64,
        )

    context = EvaluationContext(
        eval_dataset=[{"answer": "tests", "gen": ["code"]}],
        task_name="code_opensource/test",
        label_key="answer",
        response_key="gen",
        result_path=tmp_path / "code.json",
        max_workers=1,
        timeout=1,
        exec_timeout=1.0,
        seed=1,
        allow_unsafe_code=True,
    )

    result = CodeTask(scorer).score(context)

    assert result.sample_count == 64
    assert result.metrics["pass@1"].count == 1
