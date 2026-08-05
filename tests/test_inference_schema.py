"""Tests for backend-independent inference request and result schemas."""

from __future__ import annotations

import pytest

from llmeval.inference.schema import (
    ChoiceLoglikelihood,
    LoglikelihoodRequest,
    LoglikelihoodResult,
)


def test_request_moves_trailing_context_whitespace_for_scoring() -> None:
    request = LoglikelihoodRequest("Question:\n ", ("A", "答案"))

    assert request.scoring_context == "Question:"
    assert request.continuation_prefix == "\n "
    assert request.scored_continuation(0) == "\n A"
    assert request.scored_continuation(1) == "\n 答案"


def test_choice_counts_original_text_and_validates_token_alignment() -> None:
    choice = ChoiceLoglikelihood(
        continuation="答案",
        scored_text=" 答案",
        token_logprobs=(-0.4, -0.2),
        token_texts=(" 答", "案"),
        token_ids=(11, 12),
    )

    assert choice.total_logprob == pytest.approx(-0.6)
    assert choice.token_count == 2
    assert choice.char_count == 2
    assert choice.byte_count == 6

    with pytest.raises(ValueError, match="reconstruct"):
        ChoiceLoglikelihood(
            continuation="答案",
            scored_text=" 答案",
            token_logprobs=(-0.4,),
            token_texts=("答案",),
        )

    for invalid_logprob in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            ChoiceLoglikelihood(
                continuation="A",
                scored_text="A",
                token_logprobs=(invalid_logprob,),
                token_texts=("A",),
            )


def test_loglikelihood_cache_round_trip_is_strictly_aligned() -> None:
    request = LoglikelihoodRequest("Q:", ("A", "答案"))
    result = LoglikelihoodResult(
        request=request,
        choices=(
            ChoiceLoglikelihood("A", "A", (-0.1,), ("A",), (1,)),
            ChoiceLoglikelihood("答案", "答案", (-0.2,), ("答案",), None),
        ),
        exact=True,
    )

    restored = LoglikelihoodResult.from_cache_value(request, result.to_cache_value())

    assert restored == result
    with pytest.raises(ValueError, match="cannot be cached"):
        LoglikelihoodResult.failure(request, "backend_error").to_cache_value()

    invalid_cache_value = result.to_cache_value()
    invalid_cache_value["exact"] = "true"
    with pytest.raises(ValueError, match="must be exact"):
        LoglikelihoodResult.from_cache_value(request, invalid_cache_value)

    invalid_cache_value = result.to_cache_value()
    invalid_cache_value["choices"][0]["token_logprobs"] = ["-0.1"]
    with pytest.raises(ValueError, match="must be numeric"):
        LoglikelihoodResult.from_cache_value(request, invalid_cache_value)
