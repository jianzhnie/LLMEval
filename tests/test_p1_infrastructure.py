"""Regression tests for P1 registry, metrics, and caches."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from llmeval.cache import ContentAddressedCache
from llmeval.inference.mc import FewShotFormatter
from llmeval.tasks.registry import (
    CodeTask,
    EvaluationContext,
    MathTask,
    MCTask,
    TaskRegistry,
    evaluate_registered_task,
    write_per_item_results,
    write_structured_summary,
)
from llmeval.tasks.results import (
    EvaluationResult,
    MetricValue,
    ScorerResult,
    aggregate_metric_values,
    metric_from_samples,
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

    write_structured_summary(result, tmp_path / "score.jsonl")
    summary = json.loads((tmp_path / "score.summary.json").read_text())
    assert summary["per_item"] == result.per_item
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
    write_per_item_results(result, compact_path)
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
    write_per_item_results(result, debug_path, output_schema="debug")
    assert json.loads(debug_path.read_text()) == record


def test_cache_tracks_hits_misses_corruption_and_lifecycle(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path, "evaluation")
    key = cache.key({"request": 1})

    assert cache.get(key) is None
    cache.set(key, {"value": 1})
    assert cache.get(key) == {"value": 1}
    (tmp_path / "evaluation" / f"{key}.json").write_text("broken", encoding="utf-8")
    assert cache.get(key) is None
    stats = cache.stats()
    assert stats.to_dict() == {"hits": 1, "misses": 2, "corrupt": 1, "writes": 1}

    cache.set(key, {"value": 2})
    assert cache.clear() == 1
    assert cache.get(key) is None


def test_cache_rank_isolation(tmp_path: Path) -> None:
    key_payload = {"model_name": "m", "seed": 7}
    rank_zero = ContentAddressedCache(tmp_path, "inference", rank=0)
    rank_one = ContentAddressedCache(tmp_path, "inference", rank=1)
    key = rank_zero.key(key_payload)

    rank_zero.set(key, {"rank": 0})
    rank_one.set(key, {"rank": 1})
    assert rank_zero.get(key) == {"rank": 0}
    assert rank_one.get(key) == {"rank": 1}
    assert (tmp_path / "inference" / "rank-0" / f"{key}.json").exists()
    assert (tmp_path / "inference" / "rank-1" / f"{key}.json").exists()


def test_cache_atomic_writes_are_valid_across_processes(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path, "inference")
    key = cache.key({"request": "concurrent"})
    script = (
        "from llmeval.cache import ContentAddressedCache; "
        "import sys; "
        "cache=ContentAddressedCache(sys.argv[1], 'inference'); "
        "cache.set(sys.argv[3], {'value': sys.argv[2]})"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(index), key],
            cwd=Path(__file__).parents[1],
        )
        for index in range(8)
    ]
    assert all(process.wait(timeout=30) == 0 for process in processes)
    assert cache.get(key) in ({"value": str(index)} for index in range(8))


def test_cache_cleanup_cli_clears_namespace(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path, "evaluation")
    key = cache.key({"request": "cleanup"})
    cache.set(key, {"value": 1})

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmeval.cache",
            "clear",
            "--root",
            str(tmp_path),
            "--namespace",
            "evaluation",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["removed"] == 1
    assert not (tmp_path / "evaluation" / f"{key}.json").exists()


def test_content_cache_round_trip_corruption_and_key_isolation(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path, "evaluation")
    key = cache.key({"task": "math", "seed": 1})
    other_key = cache.key({"task": "math", "seed": 2})
    assert key != other_key
    cache.set(key, {"value": 1})
    assert cache.get(key) == {"value": 1}
    assert cache.get(other_key) is None

    path = tmp_path / "evaluation" / f"{key}.json"
    path.write_text("not json", encoding="utf-8")
    assert cache.get(key) is None


def test_content_cache_read_only_and_force_recompute(tmp_path: Path) -> None:
    writable = ContentAddressedCache(tmp_path, "inference")
    key = writable.key({"request": 1})
    writable.set(key, {"answer": "cached"})
    read_only = ContentAddressedCache(tmp_path, "inference", read_only=True)
    assert read_only.get(key) == {"answer": "cached"}
    read_only.set(key, {"answer": "ignored"})
    assert read_only.get(key) == {"answer": "cached"}
    forced = ContentAddressedCache(tmp_path, "inference", force_recompute=True)
    assert forced.get(key) is None


def test_content_cache_serializes_non_finite_numbers_as_json_strings(
    tmp_path: Path,
) -> None:
    cache = ContentAddressedCache(tmp_path, "inference")
    key = cache.key({"request": "first-token"})
    cache.set(key, {"scores": [float("-inf"), float("inf"), float("nan")]})

    raw = (tmp_path / "inference" / f"{key}.json").read_text(encoding="utf-8")
    assert "-Infinity" not in raw
    assert cache.get(key) == {"scores": ["-inf", "inf", "nan"]}


def test_bootstrap_and_aggregation_are_deterministic() -> None:
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
    assert aggregate_metric_values([first], mode="micro").value == pytest.approx(0.5)
    assert aggregate_metric_values([first], mode="macro").value == pytest.approx(0.5)


def test_few_shot_sampling_is_per_document_and_seeded(tmp_path: Path) -> None:
    source = tmp_path / "dev.jsonl"
    examples = [
        {
            "doc_id": f"dev:{index}",
            "prompt": f"Question {index}?\nA. one\nB. two\nAnswer:",
            "answer": "A",
        }
        for index in range(5)
    ]
    source.write_text(
        "\n".join(json.dumps(example) for example in examples), encoding="utf-8"
    )
    formatter = FewShotFormatter(2, str(source), seed=9)
    formatter.load()
    first = formatter.get_prefix("test prompt", "test:0")
    repeat = formatter.get_prefix("test prompt", "test:0")
    other = formatter.get_prefix("test prompt", "test:1")
    assert first == repeat
    assert first != other


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


def test_evaluation_cache_avoids_recomputing_registered_task(tmp_path: Path) -> None:
    calls = 0

    def scorer(**kwargs: object) -> ScorerResult:
        nonlocal calls
        calls += 1
        eval_dataset = kwargs["eval_dataset"]
        assert isinstance(eval_dataset, list)
        for item in eval_dataset:
            assert isinstance(item, dict)
            item.update({"accuracy": 1.0, "raw_gen": "4"})
        return ScorerResult(
            metrics={"accuracy": 1.0},
            observations={"accuracy": [1.0]},
            per_item=eval_dataset,
            sample_count=1,
            effective_sample_count=1,
        )

    task = MathTask(scorer)
    registry = TaskRegistry({"math_opensource": task})
    dataset = [{"prompt": "2+2", "answer": "4", "gen": ["4"]}]

    def context() -> EvaluationContext:
        return EvaluationContext(
            eval_dataset=[dict(item) for item in dataset],
            task_name="math_opensource/test",
            label_key="answer",
            response_key="gen",
            cache_path=tmp_path / "legacy.jsonl",
            max_workers=1,
            timeout=1,
            exec_timeout=1.0,
            seed=42,
            content_cache_dir=tmp_path / "content",
            bootstrap_samples=10,
        )

    first = evaluate_registered_task(context(), registry)
    second = evaluate_registered_task(context(), registry)
    assert calls == 1
    assert first.to_dict() == second.to_dict()


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
