"""Tests for llmeval.tasks.mc_eval.mc_score.

Contains golden metric parity checks against the local lm-evaluation-harness,
responsiveness checks for the redistributed scorer tests (moved from the old
mega-file ``tests/test_mc_eval.py``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

task_module = pytest.importorskip("lm_eval.api.task")
ConfigurableTask = task_module.ConfigurableTask


class _HarnessMultipleChoiceTask:
    OUTPUT_TYPE = "multiple_choice"
    config = SimpleNamespace(process_results=None)
    _metric_fn_list: ClassVar[dict[str, None]] = {
        "acc": None,
        "acc_norm": None,
        "acc_bytes": None,
    }
    multiple_input = False
    multiple_target = False

    @staticmethod
    def doc_to_choice(doc: dict[str, object]) -> list[str]:
        choices = doc["choices"]
        assert isinstance(choices, list)
        return [str(choice) for choice in choices]

    @staticmethod
    def doc_to_target(doc: dict[str, object]) -> int:
        return int(doc["gold"])


@pytest.mark.parametrize(
    ("choices", "logprobs", "gold"),
    [
        (["A", "B", "C"], [-2.0, -0.5, -3.0], 1),
        (["AB", "C"], [-1.0, -0.7], 0),
        (["你你", "abc"], [-1.0, -0.8], 0),
    ],
)
def test_mc_metrics_match_local_harness(
    choices: list[str], logprobs: list[float], gold: int
) -> None:
    document: dict[str, object] = {"choices": choices, "gold": gold}
    harness = ConfigurableTask.process_results(
        _HarnessMultipleChoiceTask(),
        document,
        [(logprob, False) for logprob in logprobs],
    )
    llmeval = score_loglikelihood_item(
        {
            "gold": gold,
            "logprobs": logprobs,
            "choice_logprobs": [[logprob] for logprob in logprobs],
            "choice_tokens": choices,
            "choice_char_count": [len(choice) for choice in choices],
            "choice_byte_count": [len(choice.encode("utf-8")) for choice in choices],
        }
    )

    assert float(llmeval["correct"]) == harness["acc"]
    assert float(llmeval["correct_norm"]) == harness["acc_norm"]
    assert float(llmeval["correct_bytes"]) == harness["acc_bytes"]


# ===========================================================================
# Scorer behavior tests (redistributed from tests/test_mc_eval.py)
# ===========================================================================


class TestMCExtractAnswer:
    """Test answer letter extraction from generated text."""

    def test_extract_answer_pattern(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import extract_answer

        assert extract_answer("Some text\nAnswer: B") == "B"
        assert extract_answer("Reasoning...\n答案：D") == "D"
        assert extract_answer("Answer: A") == "A"

    def test_extract_last_letter_fallback(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import extract_answer

        assert extract_answer("The correct option is C.") == "C"
        assert extract_answer("I think A and B but choose D") == "D"

    def test_extract_empty(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import extract_answer

        assert extract_answer("") == ""
        assert extract_answer("no letters here") == ""


class TestScoreLoglikelihood:
    """Test loglikelihood scoring with acc + acc_norm."""

    def test_all_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0], "choices": ["a", "b", "c"]},
            {"gold": 0, "logprobs": [-0.1, -1.0, -3.0], "choices": ["x", "y", "z"]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            assert acc == 1.0
            # Check summary file
            summary_path = Path(cache).with_suffix(".summary.json")
            assert summary_path.exists()
            s = json.loads(summary_path.read_text())
            assert s["acc"] == 1.0
            assert s["acc_norm"] == 1.0
            assert s["acc_bytes"] == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_half_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0]},  # correct: index 1
            {"gold": 0, "logprobs": [-0.5, -0.1, -3.0]},  # wrong: index 1 wins, gold 0
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            assert acc == 0.5
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_acc_norm_length_penalized(self) -> None:
        """acc_norm should prefer shorter choices with same logprob."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        # Choice A: short, Choice B: very long. Same raw logprob.
        items = [
            {
                "gold": 0,
                "logprobs": [-10.0, -10.0],
                "choices": ["A", "BBBBBBBBBB"],
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            # Same logprob: argmax picks index 0 (= gold) → acc correct
            assert acc == 1.0
            summary_path = Path(cache).with_suffix(".summary.json")
            s = json.loads(summary_path.read_text())
            # acc_norm: "A"(len=1) → -10/1=-10.0, "B"×10 → -10/10=-1.0
            # argmax picks index 1 (long) ≠ gold 0 → acc_norm wrong
            assert s["acc_norm"] == 0.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_acc_bytes_uses_utf8_length(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {
                "gold": 0,
                "logprobs": [-10.0, -10.0],
                "choices": ["é", "aa"],
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            score_loglikelihood(items, cache)
            summary_path = Path(cache).with_suffix(".summary.json")
            s = json.loads(summary_path.read_text())
            assert s["acc_norm"] == 0.0
            assert s["acc_bytes"] == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_empty_dataset(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood([], cache)
            assert acc == 0.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)


class TestScoreGenerate:
    """Test generation-based scoring."""

    def test_exact_match(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "B", "gen": ["Some text\nAnswer: B"]},
            {"answer": "A", "gen": ["The answer is A."]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_generate(items, "answer", "gen", cache)
            assert acc == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_mixed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "C", "gen": ["Answer: B"]},
            {"answer": "D", "gen": ["Answer: D"]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_generate(items, "answer", "gen", cache)
            assert acc == 0.5
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    @pytest.mark.parametrize(
        ("aggregation", "expected"),
        [("first", 0.0), ("majority_vote", 1.0), ("any_correct", 1.0)],
    )
    def test_multiple_generation_aggregation(
        self, aggregation: str, expected: float, tmp_path: Path
    ) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        cache = tmp_path / f"{aggregation}.jsonl"
        items = [
            {
                "doc_id": "q0",
                "answer": "B",
                "gen": [generation],
            }
            for generation in ["Answer: A", "Answer: B", "Answer: B"]
        ]
        assert (
            score_generate(items, "answer", "gen", cache, aggregation=aggregation)
            == expected
        )

    def test_per_sample_aggregation_uses_sample_denominator(
        self, tmp_path: Path
    ) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        cache = tmp_path / "per_sample.jsonl"
        items = [
            {
                "doc_id": "q0",
                "answer": "B",
                "gen": [generation],
            }
            for generation in ["Answer: A", "Answer: B"]
        ]
        assert (
            score_generate(items, "answer", "gen", cache, aggregation="per_sample")
            == 0.5
        )
        summary = json.loads(cache.with_suffix(".summary.json").read_text())
        assert summary["total"] == 2


class TestScoreLoglikelihoodItem:
    """Per-item loglikelihood scoring: argmax, gold parsing, guards."""

    def test_argmax_picks_highest(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        rec = score_loglikelihood_item({"gold": 1, "logprobs": [1.0, 3.0, 2.0]})
        assert rec["pred"] == 1
        assert rec["correct"] is True

    def test_single_choice(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        rec = score_loglikelihood_item({"gold": 0, "logprobs": [5.0]})
        assert rec["pred"] == 0
        assert rec["correct"] is True

    def test_choice_logprobs_are_used_without_aggregate_logprobs(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        rec = score_loglikelihood_item(
            {
                "gold": 1,
                "logprobs": [],
                "choice_logprobs": [[-3.0], [-1.0]],
            }
        )

        assert rec["pred"] == 1
        assert rec["correct"] is True

    def test_argmax_over_negative_logprobs(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        rec = score_loglikelihood_item({"gold": 1, "logprobs": [-1.0, -0.5, -2.0]})
        assert rec["pred"] == 1
        assert rec["correct"] is True

    def test_string_gold_coerced(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        rec = score_loglikelihood_item({"gold": "1", "logprobs": [-1.0, -0.5]})
        assert rec["gold"] == 1
        assert rec["correct"] is True


class TestMCScoreEdgeCases:
    """Regression tests for scorer fixes."""

    def test_null_logprob_restores_missing_candidate(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        record = score_loglikelihood_item(
            {"gold": 0, "choices": ["A", "B"], "logprobs": [-0.2, None]}
        )

        assert record["pred"] == 0
        assert record["correct"] is True

    def test_context_length_marker_is_excluded_as_inference_failure(
        self, tmp_path: Path
    ) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_result

        result = score_loglikelihood_result(
            [
                {
                    "doc_id": "mmlu:0",
                    "gold": 0,
                    "choices": ["A", "B"],
                    "logprobs": [],
                    "error": "context_length_exceeded",
                }
            ],
            tmp_path / "context.jsonl",
        )

        assert result.sample_count == 1
        assert result.effective_sample_count == 0
        assert result.failed_count == 1

    def test_generate_empty_gold_and_pred_not_correct(self, tmp_path: Path) -> None:
        """Empty gold + unparseable (empty) pred must NOT count as correct."""
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "", "gen": ["no letter here"]},
            {"answer": "B", "gen": ["Answer: B"]},
        ]
        acc = score_generate(items, "answer", "gen", tmp_path / "c.jsonl")
        assert acc == 1.0  # invalid-gold items are skipped from the denominator
        summary = json.loads((tmp_path / "c.summary.json").read_text())
        assert summary["sample_total"] == 2
        assert summary["effective_sample_count"] == 1
        assert summary["skipped_count"] == 1

    def test_generate_null_gold_is_skipped(self, tmp_path: Path) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": None, "gen": ["Answer: A"]},
            {"answer": "B", "gen": ["Answer: B"]},
        ]
        cache = tmp_path / "null_gold.jsonl"
        assert score_generate(items, "answer", "gen", cache) == 1.0
        summary = json.loads(cache.with_suffix(".summary.json").read_text())
        assert summary["effective_sample_count"] == 1
        assert summary["skipped_count"] == 1

    def test_per_sample_timeout_remains_visible_in_structured_counts(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import (
            MCScoreResult,
            _error_record,
            _to_scorer_result,
        )

        timeout_record = _error_record(
            {"answer": "A", "gen": ["Answer: A", "Answer: B"]},
            "generate",
            "answer",
            "gen",
            "per_sample",
            "timeout",
        )
        # The generation count is backfilled so per_sample weighting keeps the
        # item visible instead of evaporating from every count.
        assert timeout_record["sample_total"] == 2
        result = _to_scorer_result(MCScoreResult(per_item=[timeout_record]))

        assert result.sample_count == 2
        assert result.effective_sample_count == 0
        assert result.timeout_count == 2
        assert result.failure_counts == {"timeout": 2}

    def test_per_sample_timeout_counts_string_generation_as_one_sample(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import (
            MCScoreResult,
            _error_record,
            _to_scorer_result,
        )

        timeout_record = _error_record(
            {"answer": "A", "gen": "Answer: A"},
            "generate",
            "answer",
            "gen",
            "per_sample",
            "timeout",
        )

        assert timeout_record["sample_total"] == 1
        result = _to_scorer_result(MCScoreResult(per_item=[timeout_record]))
        assert result.sample_count == 1
        assert result.timeout_count == 1

    def test_pool_timeout_is_classified_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pool-level timeout (missing worker result) is timeout, not failed."""
        import llmeval.tasks.mc_eval.mc_score as mc_score

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

        monkeypatch.setattr(mc_score, "ProcessPool", _FakePool)
        result = mc_score.score_generate_result(
            [
                {"doc_id": "mmlu:0", "answer": "A", "gen": ["Answer: A"]},
                {"doc_id": "mmlu:1", "answer": "B", "gen": ["Answer: B"]},
            ],
            "answer",
            "gen",
            tmp_path / "pool_timeout.jsonl",
            max_workers=2,
            timeout=60,
        )

        assert result.sample_count == 2
        assert result.timeout_count == 2
        assert result.failed_count == 0
        assert result.effective_sample_count == 0
        assert result.failure_counts == {"timeout": 2}
        assert all(r["evaluation_status"] == "timeout" for r in result.per_item)

    def test_loglikelihood_all_neg_inf_counted_wrong(self, tmp_path: Path) -> None:
        """All -inf logprobs (failed inference) must not be argmax-scored."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 0, "logprobs": [float("-inf")] * 2, "choices": ["a", "b"]},
            {"gold": 1, "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
        ]
        acc = score_loglikelihood(items, tmp_path / "c.jsonl")
        assert acc == 1.0
        summary = json.loads((tmp_path / "c.summary.json").read_text())
        assert summary["effective_sample_count"] == 1
        assert summary["failed_count"] == 1

    def test_acc_norm_uses_choices_when_present(self, tmp_path: Path) -> None:
        """Length normalization flips the argmax when choices differ in length."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        # raw argmax → index 1; normalized: -2.0/4=-0.5 vs -1.0/1=-1.0 → index 0
        items = [{"gold": 0, "logprobs": [-2.0, -1.0], "choices": ["aaaa", "b"]}]
        acc = score_loglikelihood(items, tmp_path / "c.jsonl")
        assert acc == 0.0
        summary = json.loads((tmp_path / "c.summary.json").read_text())
        assert summary["acc_norm"] == 1.0

    def test_acc_norm_prefers_choice_tokens_when_present(self, tmp_path: Path) -> None:
        """Answer-token scoring should not normalize by full option text length."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {
                "gold": 0,
                "logprobs": [-2.0, -1.0],
                "choice_tokens": ["A", "B"],
                "choices": ["aaaa", "b"],
            }
        ]
        score_loglikelihood(items, tmp_path / "c.jsonl")
        summary = json.loads((tmp_path / "c.summary.json").read_text())
        assert summary["acc_norm"] == 0.0

    def test_extract_lowercase_letter(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import extract_answer

        assert extract_answer("the answer is b") == "B"
        assert extract_answer("选 c") == "C"

    def test_non_numeric_gold_treated_invalid(self, tmp_path: Path) -> None:
        """A non-numeric gold is skipped rather than entering the denominator."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": "B", "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
            {"gold": 1, "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
        ]
        acc = score_loglikelihood(items, tmp_path / "c.jsonl")
        assert acc == 1.0

    def test_generate_bare_string_gen_tolerated(self, tmp_path: Path) -> None:
        """A plain-string gen field (schema expects list) is scored as text."""
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [{"answer": "B", "gen": "Answer: B"}]
        acc = score_generate(items, "answer", "gen", tmp_path / "c.jsonl")
        assert acc == 1.0


def test_generate_filter_trace_preserves_raw_response(tmp_path: Path) -> None:
    from llmeval.tasks.mc_eval.mc_score import score_generate

    cache = tmp_path / "mc.jsonl"
    score_generate(
        [
            {
                "answer": "B",
                "gen": ["<think>x</think><answer>B</answer>"],
            }
        ],
        "answer",
        "gen",
        cache,
    )

    record = json.loads(cache.read_text(encoding="utf-8"))
    assert record["raw_gen"] == ["<think>x</think><answer>B</answer>"]
    assert record["filtered_gen"] == ["B"]
    assert record["filter_trace"][0]["pipeline"] == "mc_generation"
    assert [step["name"] for step in record["filter_trace"][0]["filters"]] == [
        "strip_reasoning",
        "extract_answer",
    ]


def test_generate_merges_resumed_rows_before_aggregation(tmp_path: Path) -> None:
    from llmeval.tasks.mc_eval.mc_score import score_generate

    items = [
        {
            "doc_id": "mmlu:0",
            "prompt": "q",
            "answer": "B",
            "gen": ["Answer: A"],
        },
        {
            "doc_id": "mmlu:0",
            "prompt": "q",
            "answer": "B",
            "gen": ["Answer: B"],
        },
        {
            "doc_id": "mmlu:0",
            "prompt": "q",
            "answer": "B",
            "gen": ["Answer: B"],
        },
    ]
    cache = tmp_path / "resumed.jsonl"

    assert (
        score_generate(items, "answer", "gen", cache, aggregation="majority_vote")
        == 1.0
    )
    records = cache.read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["predictions"] == ["A", "B", "B"]
    summary = json.loads(cache.with_suffix(".summary.json").read_text())
    assert summary["question_total"] == 1
    assert summary["sample_total"] == 3
    assert summary["aggregation"] == "majority_vote"


class TestMCRepeatedRows:
    """MC aggregation consumes repeated one-generation rows."""

    def _row(self, gens: list[str], **extra: object) -> dict:
        return {
            "doc_id": "mmlu:0",
            "prompt": "q",
            "answer": "B",
            "gen": gens,
            **extra,
        }

    def test_repeated_rows_are_grouped_in_file_order(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import merge_generate_records

        merged = merge_generate_records(
            [
                self._row(["Answer: A"]),
                self._row(["Answer: B"]),
            ],
            "answer",
            "gen",
        )
        assert len(merged) == 1
        assert merged[0]["gen"] == ["Answer: A", "Answer: B"]

    def test_multi_generation_row_is_rejected(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import merge_generate_records

        with pytest.raises(ValueError, match="one generation per row"):
            merge_generate_records(
                [self._row(["Answer: A", "Answer: B"])],
                "answer",
                "gen",
            )

    def test_identical_generations_remain_distinct_samples(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import merge_generate_records

        merged = merge_generate_records(
            [
                self._row(["Answer: B"]),
                self._row(["Answer: B"]),
            ],
            "answer",
            "gen",
        )
        assert len(merged) == 1
        assert merged[0]["gen"] == ["Answer: B", "Answer: B"]

    def test_different_generations_remain_distinct_samples(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import merge_generate_records

        merged = merge_generate_records(
            [self._row(["Answer: A"]), self._row(["Answer: B"])],
            "answer",
            "gen",
        )
        assert merged[0]["gen"] == ["Answer: A", "Answer: B"]
