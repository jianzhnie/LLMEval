"""Golden metric parity checks against the local lm-evaluation-harness."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from llmeval.inference.schema import LoglikelihoodRequest
from llmeval.tasks.mc_eval.mc_score import score_loglikelihood_item

task_module = pytest.importorskip("lm_eval.api.task")
ConfigurableTask = task_module.ConfigurableTask
TemplateLM = pytest.importorskip("lm_eval.api.model").TemplateLM


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
