"""Tests for llmeval.tasks.mc_eval.mc_score.

Contains golden metric parity checks against the local lm-evaluation-harness,
responsiveness checks for the redistributed scorer tests (moved from the old
mega-file ``tests/test_mc_eval.py``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

task_module = pytest.importorskip("lm_eval.api.task")
ConfigurableTask = task_module.ConfigurableTask


class _HarnessMultipleChoiceTask:
    OUTPUT_TYPE = "multiple_choice"
    config = SimpleNamespace(process_results=None)
    _metric_fn_list: ClassVar[dict[str, None]] = {"acc": None}
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
    """Test answer-token loglikelihood accuracy."""

    def test_all_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_result

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0], "choices": ["a", "b", "c"]},
            {"gold": 0, "logprobs": [-0.1, -1.0, -3.0], "choices": ["x", "y", "z"]},
        ]
        result = score_loglikelihood_result(items)
        assert result.metrics == {"acc": 1.0}

    def test_half_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_result

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0]},  # correct: index 1
            {"gold": 0, "logprobs": [-0.5, -0.1, -3.0]},  # wrong: index 1 wins, gold 0
        ]
        assert score_loglikelihood_result(items).metrics["acc"] == 0.5

    def test_empty_dataset(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_result

        assert score_loglikelihood_result([]).metrics["acc"] == 0.0


class TestScoreGenerate:
    """Test generation-based scoring."""

    def test_exact_match(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {"answer": "B", "gen": ["Some text\nAnswer: B"]},
            {"answer": "A", "gen": ["The answer is A."]},
        ]
        result = score_generate_result(items, "answer", "gen")
        assert result.metrics["acc"] == 1.0

    def test_mixed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {"answer": "C", "gen": ["Answer: B"]},
            {"answer": "D", "gen": ["Answer: D"]},
        ]
        result = score_generate_result(items, "answer", "gen")
        assert result.metrics["acc"] == 0.5

    @pytest.mark.parametrize(
        ("aggregation", "expected"),
        [("first", 0.0), ("majority_vote", 1.0), ("any_correct", 1.0)],
    )
    def test_multiple_generation_aggregation(
        self, aggregation: str, expected: float
    ) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q0",
                "answer": "B",
                "gen": [generation],
            }
            for generation in ["Answer: A", "Answer: B", "Answer: B"]
        ]
        result = score_generate_result(items, "answer", "gen", aggregation=aggregation)
        assert result.metrics["acc"] == expected

    def test_per_sample_aggregation_uses_sample_denominator(
        self,
    ) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q0",
                "answer": "B",
                "gen": [generation],
            }
            for generation in ["Answer: A", "Answer: B"]
        ]
        result = score_generate_result(items, "answer", "gen", aggregation="per_sample")
        assert result.metrics["acc"] == 0.5
        assert result.sample_count == 2

    def test_explicit_inference_error_is_excluded_per_sample(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q0",
                "sample_index": 0,
                "answer": "B",
                "gen": "Answer: B",
            },
            {
                "doc_id": "q0",
                "sample_index": 1,
                "answer": "B",
                "gen": "",
                "error": "context_length_exceeded",
            },
        ]

        per_sample = score_generate_result(
            items, "answer", "gen", aggregation="per_sample"
        )
        first = score_generate_result(items, "answer", "gen", aggregation="first")

        assert per_sample.metrics["acc"] == 1.0
        assert per_sample.sample_count == 2
        assert per_sample.effective_sample_count == 1
        assert per_sample.failed_count == 1
        assert first.metrics["acc"] == 1.0
        assert first.effective_sample_count == 1

    def test_first_aggregation_does_not_skip_failed_first_sample(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q0",
                "sample_index": 0,
                "answer": "B",
                "gen": "",
                "error": "context_length_exceeded",
            },
            {
                "doc_id": "q0",
                "sample_index": 1,
                "answer": "B",
                "gen": "Answer: B",
            },
        ]

        result = score_generate_result(items, "answer", "gen", aggregation="first")

        assert result.metrics["acc"] == 0.0
        assert result.sample_count == 1
        assert result.effective_sample_count == 0
        assert result.failed_count == 1

    @pytest.mark.parametrize("aggregation", ["majority_vote", "any_correct"])
    def test_question_aggregation_excludes_incomplete_samples(
        self, aggregation: str
    ) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q0",
                "sample_index": 0,
                "answer": "B",
                "gen": "Answer: B",
            },
            {
                "doc_id": "q0",
                "sample_index": 1,
                "answer": "B",
                "gen": "",
                "error": "context_length_exceeded",
            },
        ]

        result = score_generate_result(items, "answer", "gen", aggregation=aggregation)

        assert result.metrics["acc"] == 0.0
        assert result.effective_sample_count == 0
        assert result.failed_count == 1

    def test_majority_vote_counts_unparseable_answers_as_wrong_votes(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [
                {"doc_id": "q0", "sample_index": 0, "answer": "B", "gen": "?"},
                {"doc_id": "q0", "sample_index": 1, "answer": "B", "gen": "?"},
                {
                    "doc_id": "q0",
                    "sample_index": 2,
                    "answer": "B",
                    "gen": "Answer: B",
                },
            ],
            "answer",
            "gen",
            aggregation="majority_vote",
        )

        assert result.metrics["acc"] == 0.0
        assert result.effective_sample_count == 1

    def test_failed_record_recomputes_untrusted_sample_total(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _error_record

        record = _error_record(
            {"doc_id": "q1", "gen": ["A"], "sample_total": "stale"},
            "generate",
            "answer",
            "gen",
            "per_sample",
        )

        assert record["sample_total"] == 1


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

    def test_nested_choice_logprobs_do_not_override_aggregate_scores(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        rec = score_loglikelihood_item(
            {
                "gold": 1,
                "logprobs": [],
                "choice_logprobs": [[-3.0], [-1.0]],
            }
        )

        assert rec["pred"] == -1
        assert rec["correct"] is False
        assert rec["evaluation_status"] == "failed"

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

    @pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
    def test_non_finite_logprob_is_failed(self, invalid: float) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        record = score_loglikelihood_item(
            {"gold": 0, "choices": ["A", "B"], "logprobs": [invalid, -1.0]}
        )

        assert record["pred"] == -1
        assert record["evaluation_status"] == "failed"

    def test_boolean_gold_is_failed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

        record = score_loglikelihood_item(
            {"gold": True, "choices": ["A", "B"], "logprobs": [-0.1, -1.0]}
        )

        assert record["pred"] == -1
        assert record["evaluation_status"] == "failed"

    def test_fractional_gold_is_failed(self) -> None:
        record = score_loglikelihood_item(
            {"gold": 1.5, "choices": ["A", "B"], "logprobs": [-1.0, -0.1]}
        )

        assert record["pred"] == -1
        assert record["evaluation_status"] == "failed"

    def test_context_length_marker_is_excluded_as_inference_failure(self) -> None:
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
            ]
        )

        assert result.sample_count == 1
        assert result.effective_sample_count == 0
        assert result.failed_count == 1

    def test_generate_empty_gold_and_pred_not_correct(self) -> None:
        """Empty gold + unparseable (empty) pred must NOT count as correct."""
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {"answer": "", "gen": ["no letter here"]},
            {"answer": "B", "gen": ["Answer: B"]},
        ]
        result = score_generate_result(items, "answer", "gen")
        assert result.metrics["acc"] == 1.0
        assert result.sample_count == 2
        assert result.effective_sample_count == 1
        assert result.failed_count == 1

    def test_generate_null_gold_is_failed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {"answer": None, "gen": ["Answer: A"]},
            {"answer": "B", "gen": ["Answer: B"]},
        ]
        result = score_generate_result(items, "answer", "gen")
        assert result.metrics["acc"] == 1.0
        assert result.effective_sample_count == 1
        assert result.failed_count == 1

    def test_generate_invalid_gold_is_failed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [{"answer": "invalid", "gen": ["Answer: A"]}],
            "answer",
            "gen",
        )

        assert result.sample_count == 1
        assert result.effective_sample_count == 0
        assert result.failed_count == 1

    def test_generate_numeric_gold_resolves_to_option_index(self) -> None:
        """1-based numeric gold resolves to the matching choice letter."""
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q1",
                "answer": "1",
                "choices": ["a", "b"],
                "gen": ["Answer: A"],
            },
            {
                "doc_id": "q2",
                "answer": "2",
                "choices": ["a", "b"],
                "gen": ["Answer: B"],
            },
        ]
        result = score_generate_result(items, "answer", "gen")

        assert result.metrics["acc"] == 1.0
        assert result.failed_count == 0
        assert result.effective_sample_count == 2

    def test_generate_out_of_range_numeric_gold_is_failed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [{"answer": "5", "choices": ["a", "b"], "gen": ["Answer: A"]}],
            "answer",
            "gen",
        )

        assert result.failed_count == 1
        assert result.effective_sample_count == 0

    def test_generate_text_gold_matches_choice_text(self) -> None:
        """Full-text gold equal to a choice's text resolves to that choice."""
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [
            {
                "doc_id": "q1",
                "answer": "Paris",
                "choices": ["London", "Paris"],
                "gen": ["Answer: B"],
            },
        ]
        result = score_generate_result(items, "answer", "gen")

        assert result.metrics["acc"] == 1.0
        assert result.failed_count == 0

    @pytest.mark.parametrize("aggregation", ["first", "per_sample"])
    def test_generate_missing_response_key_is_failed(self, aggregation: str) -> None:
        """A row without the response key is a structural failure, not an
        empty completed answer."""
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [{"doc_id": "q", "answer": "A"}],
            "answer",
            "gen",
            max_workers=1,
            aggregation=aggregation,
        )

        assert result.sample_count == 1
        assert result.effective_sample_count == 0
        assert result.failed_count == 1
        assert result.observations["acc"] == []

    @pytest.mark.parametrize("generation", [None, [None], 123])
    def test_generate_malformed_response_is_failed(self, generation: object) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [{"answer": "A", "gen": generation}],
            "answer",
            "gen",
            max_workers=1,
        )

        assert result.failed_count == 1
        assert result.effective_sample_count == 0

    def test_generate_letter_gold_must_exist_in_choices(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [{"answer": "J", "choices": ["a", "b"], "gen": ["Answer: J"]}],
            "answer",
            "gen",
        )

        assert result.failed_count == 1
        assert result.effective_sample_count == 0

    @pytest.mark.parametrize("answer", ["11", "eleventh"])
    def test_generate_gold_cannot_resolve_past_j(self, answer: str) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        choices = [f"choice {index}" for index in range(1, 11)] + ["eleventh"]
        result = score_generate_result(
            [{"answer": answer, "choices": choices, "gen": ["Answer: J"]}],
            "answer",
            "gen",
        )

        assert result.failed_count == 1
        assert result.effective_sample_count == 0

    def test_per_sample_empty_generation_is_one_completed_wrong_sample(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        result = score_generate_result(
            [{"doc_id": "q", "answer": "A", "gen": []}],
            "answer",
            "gen",
            max_workers=1,
            aggregation="per_sample",
        )

        assert result.sample_count == 1
        assert result.effective_sample_count == 1
        assert result.failed_count == 0
        assert result.observations["acc"] == [0.0]

    def test_per_sample_failure_remains_visible_in_structured_counts(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import (
            MCScoreResult,
            _error_record,
            _to_scorer_result,
        )

        failed_record = _error_record(
            {"answer": "A", "gen": ["Answer: A", "Answer: B"]},
            "generate",
            "answer",
            "gen",
            "per_sample",
        )
        # The generation count is backfilled so per_sample weighting keeps the
        # item visible instead of evaporating from every count.
        assert failed_record["sample_total"] == 2
        result = _to_scorer_result(MCScoreResult(records=[failed_record]))

        assert result.sample_count == 2
        assert result.effective_sample_count == 0
        assert result.failed_count == 2

    def test_per_sample_failure_counts_string_generation_as_one_sample(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import (
            MCScoreResult,
            _error_record,
            _to_scorer_result,
        )

        failed_record = _error_record(
            {"answer": "A", "gen": "Answer: A"},
            "generate",
            "answer",
            "gen",
            "per_sample",
        )

        assert failed_record["sample_total"] == 1
        result = _to_scorer_result(MCScoreResult(records=[failed_record]))
        assert result.sample_count == 1
        assert result.failed_count == 1

    def test_pool_timeout_is_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing worker result is represented as a failed sample."""
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
            max_workers=2,
            timeout=60,
        )

        assert result.sample_count == 2
        assert result.failed_count == 2
        assert result.effective_sample_count == 0
        assert all(r["evaluation_status"] == "failed" for r in result.records)

    def test_loglikelihood_all_neg_inf_counted_wrong(self) -> None:
        """All -inf logprobs (failed inference) must not be argmax-scored."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_result

        items = [
            {"gold": 0, "logprobs": [float("-inf")] * 2, "choices": ["a", "b"]},
            {"gold": 1, "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
        ]
        result = score_loglikelihood_result(items)
        assert result.metrics["acc"] == 1.0
        assert result.effective_sample_count == 1
        assert result.failed_count == 1

    def test_extract_lowercase_letter(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import extract_answer

        assert extract_answer("the answer is b") == "B"
        assert extract_answer("选 c") == "C"

    def test_non_numeric_gold_treated_invalid(self) -> None:
        """A non-numeric gold is failed rather than entering the denominator."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_result

        items = [
            {"gold": "B", "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
            {"gold": 1, "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
        ]
        result = score_loglikelihood_result(items)
        assert result.metrics["acc"] == 1.0

    def test_generate_bare_string_gen_tolerated(self) -> None:
        """A plain-string gen field (schema expects list) is scored as text."""
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        items = [{"answer": "B", "gen": "Answer: B"}]
        result = score_generate_result(items, "answer", "gen")
        assert result.metrics["acc"] == 1.0


def test_generate_filter_trace_preserves_raw_response() -> None:
    from llmeval.tasks.mc_eval.mc_score import score_generate_result

    result = score_generate_result(
        [
            {
                "answer": "B",
                "gen": ["<think>x</think><answer>B</answer>"],
            }
        ],
        "answer",
        "gen",
    )

    record = result.records[0]
    assert record["raw_gen"] == ["<think>x</think><answer>B</answer>"]
    assert "filtered_gen" not in record
    assert record["filter_trace"][0]["pipeline"] == "mc_generation"
    assert [step["name"] for step in record["filter_trace"][0]["filters"]] == [
        "strip_reasoning",
        "extract_answer",
    ]


def test_extract_answer_uses_last_explicit_correction() -> None:
    from llmeval.tasks.mc_eval.mc_score import extract_answer

    assert extract_answer("Answer: A\nCorrection follows.\nAnswer: B") == "B"


def test_generate_merges_resumed_rows_before_aggregation() -> None:
    from llmeval.tasks.mc_eval.mc_score import score_generate_result

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
    result = score_generate_result(
        items,
        "answer",
        "gen",
        aggregation="majority_vote",
    )
    assert result.metrics["acc"] == 1.0
    assert len(result.records) == 1
    assert result.records[0]["predictions"] == ["A", "B", "B"]
    assert result.sample_count == 1
    assert result.records[0]["sample_total"] == 3
    assert result.records[0]["aggregation"] == "majority_vote"


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

    def test_repeated_rows_are_ordered_by_sample_index(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import (
            merge_generate_records,
            score_generate_result,
        )

        rows = [
            self._row(["Answer: A"], sample_index=1),
            self._row(["Answer: B"], sample_index=0),
        ]
        merged = merge_generate_records(rows, "answer", "gen")
        result = score_generate_result(rows, "answer", "gen", aggregation="first")

        assert merged[0]["gen"] == ["Answer: B", "Answer: A"]
        assert result.metrics["acc"] == 1.0

    def test_duplicate_sample_index_is_rejected(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import merge_generate_records

        with pytest.raises(ValueError, match="Duplicate sample_index 0"):
            merge_generate_records(
                [
                    self._row(["Answer: A"], sample_index=0),
                    self._row(["Answer: B"], sample_index=0),
                ],
                "answer",
                "gen",
            )

    def test_multi_generation_row_is_failed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate_result

        for aggregation in ("first", "per_sample"):
            result = score_generate_result(
                [self._row(["Answer: A", "Answer: B"])],
                "answer",
                "gen",
                aggregation=aggregation,
            )

            assert result.sample_count == 1
            assert result.failed_count == 1
            assert result.effective_sample_count == 0

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
