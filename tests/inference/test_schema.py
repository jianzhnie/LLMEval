"""Tests for backend-independent inference request and result schemas."""

from __future__ import annotations

import pytest

from llmeval.inference.schema import (
    ChoiceLoglikelihood,
    LoglikelihoodRequest,
)

TemplateLM = pytest.importorskip("lm_eval.api.model").TemplateLM


class _CharacterHarnessLM(TemplateLM):
    """Minimal tokenizer backend used to exercise harness pair splitting."""

    @property
    def eot_token_id(self) -> int:
        return 0

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **_: object
    ) -> list[int]:
        return [ord(character) for character in string]

    def _loglikelihood_tokens(
        self, requests: list[object], **_: object
    ) -> list[object]:
        return []

    def loglikelihood_rolling(self, requests: list[object]) -> list[float]:
        return []

    def generate_until(self, requests: list[object]) -> list[str]:
        return []


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

    assert sum(choice.token_logprobs) == pytest.approx(-0.6)
    assert len(choice.token_logprobs) == 2
    assert len(choice.continuation) == 2
    assert len(choice.continuation.encode("utf-8")) == 6

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


def test_request_boundary_matches_harness_encode_pair() -> None:
    context = "Question:\n "
    continuation = "答案"
    harness = _CharacterHarnessLM()
    harness_context_ids, harness_continuation_ids = harness._encode_pair(
        context, continuation
    )
    request = LoglikelihoodRequest(context, (continuation,))

    assert harness_context_ids == harness.tok_encode(request.scoring_context)
    assert harness_continuation_ids == harness.tok_encode(
        request.scored_continuation(0)
    )
