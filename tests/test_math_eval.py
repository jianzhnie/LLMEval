"""Tests for llmeval.tasks.math_eval.math_score.

These tests exercise the real math-verify scorer (no mocking).  They are
skipped automatically when math-verify / pebble are not installed so that
minimal CI environments keep passing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_DEPS_AVAILABLE = (
    importlib.util.find_spec("math_verify") is not None
    and importlib.util.find_spec("pebble") is not None
)
pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE, reason="math-verify / pebble not installed"
)


def _math_item(
    answer: str, gen: list[str], task: str = "math_opensource/aime24"
) -> dict:
    return {
        "prompt": "q",
        "answer": answer,
        "gen": gen,
        "task": task,
    }


# ===========================================================================
# ProcessingStats
# ===========================================================================


class TestProcessingStats:
    def test_rates(self) -> None:
        from llmeval.tasks.math_eval.math_score import ProcessingStats

        stats = ProcessingStats(total=10, correct=5, timeout=2, error=1)
        assert stats.effective == 7
        assert stats.correct_rate == pytest.approx(5 / 7 * 100)
        assert stats.timeout_rate == pytest.approx(20.0)
        assert stats.error_rate == pytest.approx(10.0)

    def test_zero_total_no_division_error(self) -> None:
        from llmeval.tasks.math_eval.math_score import ProcessingStats

        stats = ProcessingStats()
        assert stats.correct_rate == 0.0
        assert stats.timeout_rate == 0.0
        assert stats.error_rate == 0.0


# ===========================================================================
# process_answers (single-item worker)
# ===========================================================================


class TestProcessAnswers:
    def test_worker_timeout_is_verification_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.math_eval.math_score as math_mod

        monkeypatch.setattr(
            math_mod, "_verify_func", MagicMock(side_effect=TimeoutError())
        )

        result = math_mod.process_answers(
            (0, _math_item("5", ["$\\boxed{5}$"]), "answer", "gen")
        )

        assert result is not None
        assert result.failure_stage == "verification"
        assert result.failure_reason == "timeout"
        assert result.predicted == "Timeout"

    def test_value_error_fallback_is_marked_without_error_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import llmeval.tasks.math_eval.math_score as math_mod

        monkeypatch.setattr(
            math_mod, "_verify_func", MagicMock(side_effect=ValueError("symbolic"))
        )
        monkeypatch.setattr(math_mod, "_math_text_equiv", lambda *_: True)

        result = math_mod.process_answers(
            (0, _math_item("p - q", ["p - q"]), "answer", "gen")
        )

        assert result is not None
        assert result.fallback_matched is True
        assert not [record for record in caplog.records if record.levelname == "ERROR"]

    def test_correct_answer(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("5", ["The answer is $\\boxed{5}$"])
        idx, grade, pred, gold = process_answers((0, item, "answer", "gen"))
        assert idx == 0
        assert grade == 1.0
        assert pred is not None
        assert gold is not None

    def test_wrong_answer(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("5", ["The answer is $\\boxed{6}$"])
        _, grade, _, _ = process_answers((0, item, "answer", "gen"))
        assert grade == 0.0

    def test_latex_equivalence(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("\\frac{1}{2}", ["The answer is $\\boxed{0.5}$"])
        _, grade, _, _ = process_answers((0, item, "answer", "gen"))
        assert grade == 1.0

    def test_gsm8k_gold_parsing(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item(
            "2+2=4\n#### 4", ["The answer is 4"], task="math_opensource/gsm8k"
        )
        _, grade, _, _ = process_answers((0, item, "answer", "gen"))
        assert grade == 1.0

    def test_missing_gen(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = {"prompt": "q", "answer": "5", "task": "math_opensource/aime24"}
        idx, grade, pred, gold = process_answers((3, item, "answer", "gen"))
        assert (idx, grade, pred, gold) == (3, 0.0, None, None)

    def test_empty_gen_list(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("5", [])
        _, grade, _, _ = process_answers((0, item, "answer", "gen"))
        assert grade == 0.0

    def test_bare_string_gen_tolerated(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("5", ["$\\boxed{5}$"])
        item["gen"] = "$\\boxed{5}$"  # str instead of list
        _, grade, _, _ = process_answers((0, item, "answer", "gen"))
        assert grade == 1.0

    def test_invalid_task_format(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = {"prompt": "q", "answer": "5", "gen": ["5"], "task": "noslash"}
        idx, grade, pred, gold = process_answers((7, item, "answer", "gen"))
        assert (idx, grade, pred, gold) == (7, 0.0, None, None)

    def test_invalid_math_task_prefix_is_rejected(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("5", ["5"], task="math_opensource_evil/task")
        idx, grade, pred, gold = process_answers((7, item, "answer", "gen"))
        assert (idx, grade, pred, gold) == (7, 0.0, None, None)

    def test_family_only_task_uses_generic_parser(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("5", ["The answer is $\\boxed{5}$"], task="math_opensource")
        idx, grade, pred, gold = process_answers((8, item, "answer", "gen"))
        assert idx == 8
        assert grade == 1.0
        assert pred is not None
        assert gold is not None

    def test_missing_task_field(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = {"prompt": "q", "answer": "5", "gen": ["5"]}
        _, grade, _, _ = process_answers((0, item, "answer", "gen"))
        assert grade == 0.0

    def test_unparseable_gold(self) -> None:
        from llmeval.tasks.math_eval.math_score import process_answers

        item = _math_item("", ["5"])  # empty gold → parse error
        idx, grade, _, _ = process_answers((1, item, "answer", "gen"))
        assert (idx, grade) == (1, 0.0)

    def test_fallback_normalizes_final_answer_text(self) -> None:
        from llmeval.tasks.math_eval.math_score import _math_text_equiv

        assert _math_text_equiv(
            "100000",
            "Final Answer: The final answer is $100,000$. I hope it is correct.",
        )


# ===========================================================================
# compute_scores (parallel driver)
# ===========================================================================


class TestComputeScores:
    def test_worker_exception_returns_indexed_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.math_eval.math_score as math_mod

        monkeypatch.setattr(
            math_mod,
            "_process_answers_impl",
            lambda _args: (_ for _ in ()).throw(TypeError("bad schema")),
        )

        result = math_mod.process_answers((3, {}, "answer", "gen"))

        assert result.index == 3
        assert result.failure_stage == "verification"
        assert result.failure_reason == "worker error: bad schema"

    def test_extraction_failure_is_separate_from_wrong_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.math_eval.math_score as math_mod

        def fake_compute_scores(*, eval_dataset, **_kwargs):
            eval_dataset[0].update(
                accuracy=0.0,
                evaluation_status="failed",
                failure_stage="extraction",
            )
            eval_dataset[1].update(
                accuracy=0.0,
                evaluation_status="completed",
                failure_stage="none",
            )
            return 0.0

        monkeypatch.setattr(math_mod, "compute_scores", fake_compute_scores)
        result = math_mod.score_math_result(
            eval_dataset=[
                _math_item("5", ["unparseable"]),
                _math_item("5", ["$\\boxed{6}$"]),
            ],
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=1,
            timeout=60,
        )

        assert result.failed_count == 1
        assert result.effective_sample_count == 1
        assert result.failure_counts["extraction_failed"] == 1
        assert "wrong_answer" not in result.failure_counts
        assert result.details["wrong_answer_count"] == 1

    def test_mixed_accuracy_and_fields(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import compute_scores

        data = [
            _math_item("5", ["$\\boxed{5}$"]),  # correct
            _math_item("4", ["$\\boxed{3}$"]),  # wrong
        ]
        cache = tmp_path / "cache.jsonl"
        acc = compute_scores(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(cache),
            max_workers=2,
            timeout=60,
        )
        assert acc == pytest.approx(0.5)
        # Result fields appended in-place
        assert data[0]["accuracy"] == 1.0
        assert data[1]["accuracy"] == 0.0
        assert "extracted_gold" in data[0]
        assert "extracted_answer" in data[0]
        assert data[0]["raw_gen"] == "$\\boxed{5}$"
        assert data[0]["filtered_gen"] == "$\\boxed{5}$"
        assert data[0]["filter_trace"]["pipeline"] == "math_response"
        summary = json.loads((tmp_path / "cache.summary.json").read_text())
        assert summary["accuracy"] == pytest.approx(0.5)
        # Cache written as valid JSONL
        lines = cache.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["accuracy"] == 1.0

    def test_all_correct(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import compute_scores

        data = [_math_item("2", ["$\\boxed{2}$"]), _math_item("3", ["$\\boxed{3}$"])]
        acc = compute_scores(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
        )
        assert acc == 1.0

    def test_empty_dataset(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import compute_scores

        acc = compute_scores(
            eval_dataset=[],
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
        )
        assert acc == 0.0

    def test_pool_timeout_is_classified_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pool-level timeout (missing worker result) is timeout, not failed."""
        import llmeval.tasks.math_eval.math_score as math_mod

        class _EmptyFuture:
            def result(self) -> object:
                return iter([])

        class _FakePool:
            def __init__(self, max_workers: int) -> None:
                pass

            def __enter__(self) -> _FakePool:
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

            def map(self, *_args: object, **_kwargs: object) -> _EmptyFuture:
                return _EmptyFuture()

        monkeypatch.setattr(math_mod, "ProcessPool", _FakePool)
        result = math_mod.score_math_result(
            eval_dataset=[
                _math_item("5", ["$\\boxed{5}$"]),
                _math_item("4", ["$\\boxed{4}$"]),
            ],
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
        )

        assert result.sample_count == 2
        assert result.timeout_count == 2
        assert result.failed_count == 0
        assert result.effective_sample_count == 0
        assert result.failure_counts["timeout"] == 2
        assert "verification_failed" not in result.failure_counts
        assert all(item["evaluation_status"] == "timeout" for item in result.per_item)

    def test_structured_result_counts_with_skipped_item(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import score_math_result

        data = [
            _math_item("5", ["$\\boxed{5}$"]),
            {
                "prompt": "q",
                "answer": "3",
                "task": "math_opensource/aime24",
            },
        ]
        result = score_math_result(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
        )
        assert result.sample_count == 2
        assert result.effective_sample_count == 1
        assert result.failed_count == 1
        assert result.skipped_count == 0
        assert result.failure_counts["inference_failed"] == 1
        assert result.observations["accuracy"] == [1.0]
        assert result.metrics["accuracy"] == 1.0

    def test_multi_sample_problem_metrics(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import score_math_result

        data = [
            {
                **_math_item("5", ["$\\boxed{5}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("5", ["$\\boxed{6}$"]),
                "doc_id": "aime24:0",
            },
            {
                # Distinct raw text extracting to the same answer as row 1.
                **_math_item("5", ["The answer is $\\boxed{5}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("7", ["$\\boxed{8}$"]),
                "doc_id": "aime24:1",
            },
            {
                **_math_item("7", ["$\\boxed{7}$"]),
                "doc_id": "aime24:1",
            },
            {
                **_math_item("7", ["The answer is $\\boxed{8}$"]),
                "doc_id": "aime24:1",
            },
        ]

        result = score_math_result(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "multi.jsonl"),
            max_workers=2,
            timeout=60,
        )

        assert result.sample_count == 6
        assert result.effective_sample_count == 6
        assert result.metrics["accuracy"] == pytest.approx(0.5)
        assert result.metrics["problem_pass@3"] == pytest.approx(1.0)
        assert result.metrics["problem_majority@3"] == pytest.approx(0.5)
        problems = result.details["problem_level"]
        assert problems[0]["correct_samples"] == 2
        assert problems[0]["sample_count"] == 3
        assert problems[0]["majority_correct"] is True
        assert problems[1]["correct_samples"] == 1
        assert problems[1]["majority_correct"] is False


# ===========================================================================
# atomic JSONL persistence (previously math_score.save_cache)
# ===========================================================================


class TestAtomicWriteJsonl:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        from llmeval.tasks.postprocess import atomic_write_jsonl

        cache = tmp_path / "out.jsonl"
        atomic_write_jsonl(cache, [{"a": 1}, {"a": 2}])
        lines = cache.read_text(encoding="utf-8").strip().split("\n")
        assert [json.loads(line)["a"] for line in lines] == [1, 2]

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        from llmeval.tasks.postprocess import atomic_write_jsonl

        cache = tmp_path / "deep" / "nested" / "out.jsonl"
        atomic_write_jsonl(cache, [{"a": 1}])
        assert cache.exists()

    def test_bare_filename_no_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: os.makedirs('') crashed for cache paths without a
        directory component (e.g. --cache_path results.jsonl)."""
        from llmeval.tasks.postprocess import atomic_write_jsonl

        monkeypatch.chdir(tmp_path)
        atomic_write_jsonl("results.jsonl", [{"a": 1}])
        assert (tmp_path / "results.jsonl").exists()

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        from llmeval.tasks.postprocess import atomic_write_jsonl

        cache = tmp_path / "out.jsonl"
        atomic_write_jsonl(cache, [{"answer": "答案是 5"}])
        record = json.loads(cache.read_text(encoding="utf-8").strip())
        assert record["answer"] == "答案是 5"


# ===========================================================================
# Sample-index protocol (P0-1)
# ===========================================================================


class TestRepeatedMathRows:
    """Repeated rows are independent math-generation samples."""

    def _score(self, data: list[dict], tmp_path: Path):
        from llmeval.tasks.math_eval.math_score import score_math_result

        return score_math_result(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
        )

    def test_repeated_rows_are_grouped_by_document(self, tmp_path: Path) -> None:
        data = [
            {
                **_math_item("5", ["$\\boxed{5}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("5", ["$\\boxed{6}$"]),
                "doc_id": "aime24:0",
            },
        ]
        result = self._score(data, tmp_path)
        problems = result.details["problem_level"]
        assert problems[0]["sample_count"] == 2
        assert problems[0]["correct_samples"] == 1

    def test_multi_generation_row_is_rejected(self, tmp_path: Path) -> None:
        data = [
            {
                **_math_item("5", ["$\\boxed{5}$", "$\\boxed{6}$"]),
                "doc_id": "aime24:0",
            }
        ]
        with pytest.raises(ValueError, match="one generation per row"):
            self._score(data, tmp_path)

    def test_empty_generation_is_recorded(self, tmp_path: Path) -> None:
        data = [
            {
                **_math_item("5", []),
                "doc_id": "aime24:0",
            }
        ]

        result = self._score(data, tmp_path)

        assert result.sample_count == 1
        assert result.failed_count == 1
        assert result.per_item[0]["failure_stage"] == "inference"

    def test_identical_responses_remain_independent_samples(
        self, tmp_path: Path
    ) -> None:
        row = {
            **_math_item("5", ["$\\boxed{5}$"]),
            "doc_id": "aime24:0",
        }
        single = self._score([dict(row)], tmp_path)
        doubled = self._score([dict(row), dict(row)], tmp_path)

        assert single.sample_count == 1
        assert doubled.sample_count == 2
        assert doubled.metrics["accuracy"] == single.metrics["accuracy"] == 1.0
        assert doubled.metrics["problem_pass@2"] == 1.0
        problems = doubled.details["problem_level"]
        assert problems[0]["sample_count"] == 2
        assert problems[0]["correct_samples"] == 2

    def test_conflicting_duplicate_rows_raise(self, tmp_path: Path) -> None:
        """Same doc_id with a different gold answer signals a corrupt resume."""
        data = [
            {
                **_math_item("5", ["$\\boxed{5}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("6", ["$\\boxed{6}$"]),
                "doc_id": "aime24:0",
            },
        ]
        with pytest.raises(ValueError, match="Conflicting 'answer'"):
            self._score(data, tmp_path)

    def test_different_generations_remain_distinct(self, tmp_path: Path) -> None:
        data = [
            {
                **_math_item("5", ["$\\boxed{5}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("5", ["$\\boxed{6}$"]),
                "doc_id": "aime24:0",
            },
        ]
        result = self._score(data, tmp_path)
        assert result.metrics["accuracy"] == pytest.approx(0.5)


# ===========================================================================
# Problem-level completeness gating (P0-4)
# ===========================================================================


def _scored_sample(doc_id: str, *, correct: bool, expected: int = 64) -> dict:
    """Fabricate one scored sample row for _build_problem_level_metrics."""
    return {
        "doc_id": doc_id,
        "accuracy": 1.0 if correct else 0.0,
        "extracted_answer": "5" if correct else "6",
        "evaluation_status": "completed",
        "expected_samples": expected,
    }


class TestProblemCompleteness:
    """@k metrics aggregate problems with the requested row count."""

    def test_partial_problem_excluded_from_pass_at_k(self) -> None:
        """A problem observed at 10/64 samples must not enter problem_pass@64."""
        from llmeval.tasks.math_eval.math_score import _build_problem_level_metrics

        rows = [_scored_sample("p1", correct=i == 0) for i in range(10)]
        problems, metrics, observations = _build_problem_level_metrics(
            rows, expected_samples=64
        )
        assert problems[0]["sample_count"] == 10
        assert problems[0]["complete"] is False
        # Empty cohort: explicit 0.0 with no observations, no division error.
        assert metrics["problem_pass@64"] == 0.0
        assert metrics["problem_majority@64"] == 0.0
        assert observations["problem_pass@64"] == []
        assert observations["problem_majority@64"] == []

    def test_full_coverage_enters_pass_at_k(self) -> None:
        """64 rows aggregate normally."""
        from llmeval.tasks.math_eval.math_score import _build_problem_level_metrics

        rows = [_scored_sample("p1", correct=i < 63) for i in range(64)]
        problems, metrics, _ = _build_problem_level_metrics(rows, expected_samples=64)
        assert problems[0]["complete"] is True
        assert metrics["problem_pass@64"] == pytest.approx(1.0)
        assert metrics["problem_majority@64"] == pytest.approx(1.0)

    def test_identical_rows_still_count_toward_requested_depth(self) -> None:
        from llmeval.tasks.math_eval.math_score import _build_problem_level_metrics

        rows = [_scored_sample("p1", correct=False) for _ in range(64)]
        problems, metrics, _ = _build_problem_level_metrics(rows, expected_samples=64)
        assert problems[0]["sample_count"] == 64
        assert problems[0]["complete"] is True
        assert metrics["problem_pass@64"] == 0.0

    def test_failed_rows_do_not_make_problem_complete(self) -> None:
        from llmeval.tasks.math_eval.math_score import _build_problem_level_metrics

        completed = _scored_sample("p1", correct=True, expected=2)
        timeout = {
            **_scored_sample("p1", correct=False, expected=2),
            "evaluation_status": "timeout",
        }

        problems, metrics, _ = _build_problem_level_metrics([completed, timeout])

        assert problems[0]["sample_count"] == 2
        assert problems[0]["observed_samples"] == 1
        assert problems[0]["complete"] is False
        assert metrics["problem_pass@2"] == 0.0

    def test_extra_rows_keep_problem_complete(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rows beyond the requested depth warn but no longer exclude the problem."""
        import logging

        import llmeval.tasks.math_eval.math_score as math_mod

        rows = [_scored_sample("p1", correct=False) for _ in range(65)]
        math_mod.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="math_score"):
                problems, metrics, _ = math_mod._build_problem_level_metrics(
                    rows, expected_samples=64
                )
        finally:
            math_mod.logger.removeHandler(caplog.handler)
        assert problems[0]["sample_count"] == 65
        assert problems[0]["complete"] is True
        assert metrics["problem_pass@64"] == 0.0
        assert any("65 completed rows" in r.message for r in caplog.records)

    def test_bool_expected_samples_is_ignored(self) -> None:
        """``True``/``False`` are not valid expected sample counts."""
        from llmeval.tasks.math_eval.math_score import _build_problem_level_metrics

        row = {**_scored_sample("p1", correct=True), "expected_samples": True}
        problems, metrics, _ = _build_problem_level_metrics([row])
        assert problems[0]["expected_samples"] == 1  # falls back to sample count
        assert problems[0]["complete"] is True
        assert metrics["problem_pass@1"] == pytest.approx(1.0)

    def test_mixed_problems_denominator_only_complete(self, tmp_path: Path) -> None:
        """With mixed problems, pass@k/majority@k divide by complete problems."""
        from llmeval.tasks.math_eval.math_score import score_math_result

        data = [
            {
                **_math_item("5", ["$\\boxed{5}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("5", ["$\\boxed{6}$"]),
                "doc_id": "aime24:0",
            },
            {
                **_math_item("7", ["$\\boxed{7}$"]),
                "doc_id": "aime24:1",
            },
        ]
        result = score_math_result(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
            expected_samples=2,
        )
        # Only the complete problem enters the @2 cohort.
        assert result.metrics["problem_pass@2"] == pytest.approx(1.0)
        assert result.metrics["problem_majority@2"] == pytest.approx(1.0)
        problems = result.details["problem_level"]
        assert problems[0]["complete"] is True
        assert problems[1]["complete"] is False
        assert result.details["complete_problem_count"] == 1
        assert result.details["incomplete_problem_count"] == 1
        assert result.details["excluded_problem_doc_ids"] == ["doc:aime24:1"]

    def test_all_incomplete_reports_zero_and_summary(self, tmp_path: Path) -> None:
        """Zero complete problems: @k is 0.0 and the summary says so."""
        from llmeval.tasks.math_eval.math_score import score_math_result

        data = [
            {
                **_math_item("7", ["$\\boxed{7}$"]),
                "doc_id": "aime24:0",
            }
        ]
        result = score_math_result(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
            expected_samples=2,
        )
        assert result.metrics["problem_pass@2"] == 0.0
        assert result.metrics["problem_majority@2"] == 0.0
        assert result.details["complete_problem_count"] == 0
        assert result.details["incomplete_problem_count"] == 1
        assert result.details["excluded_problem_doc_ids"] == ["doc:aime24:0"]
