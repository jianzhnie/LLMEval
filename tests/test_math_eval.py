"""Tests for llmeval.tasks.math_eval.math_score.

These tests exercise the real math-verify scorer (no mocking).  They are
skipped automatically when math-verify / pebble are not installed so that
minimal CI environments keep passing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    return {"prompt": "q", "answer": answer, "gen": gen, "task": task}


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
        assert summary["provenance"]["seed"] is None
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

    def test_structured_result_counts_with_skipped_item(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import compute_score_result

        data = [
            _math_item("5", ["$\\boxed{5}$"]),
            {"prompt": "q", "answer": "3", "task": "math_opensource/aime24"},
        ]
        result = compute_score_result(
            eval_dataset=data,
            label_key="answer",
            response_key="gen",
            cache_path=str(tmp_path / "cache.jsonl"),
            max_workers=2,
            timeout=60,
        )
        assert result.sample_count == 2
        assert result.effective_sample_count == 1
        assert result.skipped_count == 1
        assert result.observations["accuracy"] == [1.0]
        assert result.metrics["accuracy"] == 1.0


# ===========================================================================
# save_cache
# ===========================================================================


class TestSaveCache:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import save_cache

        cache = tmp_path / "out.jsonl"
        save_cache([{"a": 1}, {"a": 2}], str(cache))
        lines = cache.read_text(encoding="utf-8").strip().split("\n")
        assert [json.loads(line)["a"] for line in lines] == [1, 2]

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import save_cache

        cache = tmp_path / "deep" / "nested" / "out.jsonl"
        save_cache([{"a": 1}], str(cache))
        assert cache.exists()

    def test_bare_filename_no_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: os.makedirs('') crashed for cache paths without a
        directory component (e.g. --cache_path results.jsonl)."""
        from llmeval.tasks.math_eval.math_score import save_cache

        monkeypatch.chdir(tmp_path)
        save_cache([{"a": 1}], "results.jsonl")
        assert (tmp_path / "results.jsonl").exists()

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        from llmeval.tasks.math_eval.math_score import save_cache

        cache = tmp_path / "out.jsonl"
        save_cache([{"answer": "答案是 5"}], str(cache))
        record = json.loads(cache.read_text(encoding="utf-8").strip())
        assert record["answer"] == "答案是 5"
