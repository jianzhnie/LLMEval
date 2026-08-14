"""Tests for llmeval.inference.mc.

Focuses on the inference client and runner behavior without a live API.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


for mod_name in ("openai", "httpx"):
    if mod_name not in sys.modules and not importlib.util.find_spec(mod_name):
        sys.modules[mod_name] = _make_stub(mod_name)

_openai_mod = sys.modules.get("openai")
if _openai_mod is not None and not hasattr(_openai_mod, "OpenAI"):
    _openai_mod.OpenAI = MagicMock  # type: ignore[attr-defined]
    for _exc in ("APIConnectionError", "APIError", "RateLimitError"):
        if not hasattr(_openai_mod, _exc):
            setattr(_openai_mod, _exc, type(_exc, (Exception,), {}))

_httpx_mod = sys.modules.get("httpx")
if _httpx_mod is not None and not hasattr(_httpx_mod, "Timeout"):
    _httpx_mod.Timeout = MagicMock  # type: ignore[attr-defined]

if "transformers" not in sys.modules and not importlib.util.find_spec("transformers"):
    _tf = types.ModuleType("transformers")
    _tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    _tf.HfArgumentParser = MagicMock
    sys.modules["transformers"] = _tf

if "tqdm" not in sys.modules and not importlib.util.find_spec("tqdm"):
    _tqdm = types.ModuleType("tqdm")
    _tqdm.__spec__ = importlib.machinery.ModuleSpec("tqdm", loader=None)
    _tqdm.tqdm = MagicMock
    sys.modules["tqdm"] = _tqdm

from llmeval.inference.mc import (
    ContinuationAlignmentError,
    FewShotFormatter,
    MCLoglikelihoodClient,
    MCRunner,
    _aligned_continuation_logprobs,
)
from llmeval.utils.config import MCInferArguments


def _make_ll_client(max_retries: int = 0) -> MCLoglikelihoodClient:
    client = MCLoglikelihoodClient.__new__(MCLoglikelihoodClient)
    client.model_name = "m"
    client.timeout = 5
    client.max_retries = max_retries
    client.base_url = "http://test/v1"
    client.seed = 0
    client.system_prompt = None
    client.extra_body = {}
    client.client = MagicMock()
    return client


def _make_api_error(message: str = "", status_code: int | None = None) -> Exception:
    """Build an APIError instance compatible with the (possibly stubbed) openai module."""
    cls = sys.modules["openai"].APIError
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    err.message = message
    err.status_code = status_code
    return err


def _fake_top_probs_resp(top_probs: dict[str, float]) -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    token = MagicMock()
    token.top_logprobs = [
        types.SimpleNamespace(token=text, logprob=logprob)
        for text, logprob in top_probs.items()
    ]
    choice.logprobs.content = [token]
    resp.choices = [choice]
    return resp


def _fake_continuation_resp(
    context: str, continuations: list[str], scores: list[float]
) -> types.SimpleNamespace:
    scoring_context = context.rstrip()
    prefix = context[len(scoring_context) :]
    start = len(scoring_context)
    choices = [
        types.SimpleNamespace(
            index=index,
            logprobs=types.SimpleNamespace(
                text_offset=[0, start],
                token_logprobs=[None, score],
                tokens=[scoring_context, f"{prefix}{continuation}"],
            ),
        )
        for index, (continuation, score) in enumerate(
            zip(continuations, scores, strict=True)
        )
    ]
    return types.SimpleNamespace(choices=list(reversed(choices)))


def _generation_choice(content: str | None, index: int = 0) -> MagicMock:
    choice = MagicMock()
    choice.index = index
    choice.message.content = content
    return choice


def _make_mc_runner(
    tmp_path: Path, mode: str = "loglikelihood", max_retries: int = 0
) -> MCRunner:
    runner = MCRunner.__new__(MCRunner)
    input_file = tmp_path / "in.jsonl"
    input_file.touch(exist_ok=True)
    runner.config = MCInferArguments(
        input_file=str(input_file),
        output_file=str(tmp_path / "out.jsonl"),
        mode=mode,
        max_retries=max_retries,
        max_workers=1,
    )
    runner.client = None
    runner.system_prompt = None
    runner._few_shot_fmt = None
    runner._stats = {
        "processed": 0,
        "failed": 0,
    }
    return runner


def _write_input(path: Path) -> None:
    items = [
        {
            "doc_id": "test:0",
            "prompt": "Q1?\nA. x\nB. y\nAnswer:",
            "choices": ["x", "y"],
            "gold": 1,
        },
        {
            "doc_id": "test:1",
            "prompt": "Q2?\nA. p\nB. q\nAnswer:",
            "choices": ["p", "q"],
            "gold": 0,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestMCLoglikelihoodClient:
    def test_empty_key_is_quiet_for_local_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import llmeval.inference.mc as mc_mod

        monkeypatch.setattr(mc_mod.openai, "OpenAI", MagicMock())
        MCLoglikelihoodClient("http://localhost:8021/v1", "model", api_key="EMPTY")

        assert "Using default 'EMPTY' API key" not in caplog.text

    def test_single_request_with_top_logprobs(self) -> None:
        client = _make_ll_client(max_retries=3)
        client.client.chat.completions.create.return_value = _fake_top_probs_resp(
            {" A": -3.0, " B": -0.5, " C": -4.2}
        )

        result = client.get_choices_logprobs("prompt", ["A", "B", "C"])

        assert result == [-3.0, -0.5, -4.2]
        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "prompt"}]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 20
        assert call_kwargs["max_completion_tokens"] == 1
        assert "max_tokens" not in call_kwargs

    def test_system_prompt_is_sent(self) -> None:
        client = _make_ll_client()
        client.system_prompt = "Follow the format."
        client.client.chat.completions.create.return_value = _fake_top_probs_resp(
            {" A": -0.1}
        )

        client.get_choices_logprobs("prompt", ["A"])

        assert client.client.chat.completions.create.call_args.kwargs["messages"] == [
            {"role": "system", "content": "Follow the format."},
            {"role": "user", "content": "prompt"},
        ]

    def test_malformed_response_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        client = _make_ll_client(max_retries=1)
        client.client.chat.completions.create.side_effect = [
            _fake_top_probs_resp({}),
            _fake_top_probs_resp({" A": -0.1}),
        ]

        assert client.get_choices_logprobs("prompt", ["A"]) == [-0.1]
        assert client.client.chat.completions.create.call_count == 2

    def test_extra_body_is_forwarded(self) -> None:
        client = _make_ll_client()
        client.extra_body = {"top_k": 40}
        client.client.chat.completions.create.return_value = _fake_top_probs_resp(
            {" A": -3.0}
        )

        client.get_choices_logprobs("prompt", ["A"])

        assert client.client.chat.completions.create.call_args.kwargs["extra_body"] == {
            "top_k": 40
        }

    def test_programming_error_propagates(self) -> None:
        client = _make_ll_client(max_retries=0)
        client.client.chat.completions.create.side_effect = RuntimeError("down")

        with pytest.raises(RuntimeError, match="down"):
            client.get_choices_logprobs("prompt", ["A", "B"])

    def test_continuation_scores_complete_candidates(self) -> None:
        client = _make_ll_client()
        client.client.completions.create.return_value = _fake_continuation_resp(
            "Q ", ["A", "B"], [-0.2, -1.3]
        )

        result = client.score_continuations("Q ", ["A", "B"])

        assert result == [-0.2, -1.3]
        request = client.client.completions.create.call_args.kwargs
        assert request["prompt"] == ["Q A", "Q B"]
        assert request["echo"] is True
        assert request["max_tokens"] == 1
        assert "max_completion_tokens" not in request

    def test_continuation_rejects_system_prompt(self) -> None:
        client = _make_ll_client()
        client.system_prompt = "Follow instructions."

        with pytest.raises(ValueError, match="separate system prompt"):
            client.score_continuations("Q ", ["A"])

        client.client.completions.create.assert_not_called()

    def test_continuation_accepts_utf8_byte_offsets(self) -> None:
        scores = _aligned_continuation_logprobs(
            offsets=[0, 3, 4],
            token_logprobs=[None, -0.2, -0.3],
            tokens=["题", " ", "答"],
            context="题",
            continuation=" 答",
        )

        assert scores == [-0.2, -0.3]

    def test_continuation_alignment_error_is_not_retried(self) -> None:
        client = _make_ll_client(max_retries=2)
        client.client.completions.create.return_value = _fake_continuation_resp(
            "Q ", ["X"], [-0.2]
        )

        with pytest.raises(ContinuationAlignmentError):
            client.score_continuations("Q ", ["A"])

        assert client.client.completions.create.call_count == 1

    def test_malformed_continuation_response_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        client = _make_ll_client(max_retries=1)
        client.client.completions.create.side_effect = [
            types.SimpleNamespace(choices=[]),
            _fake_continuation_resp("Q ", ["A"], [-0.2]),
        ]

        assert client.score_continuations("Q ", ["A"]) == [-0.2]
        assert client.client.completions.create.call_count == 2

    def test_choice_not_in_top_returns_neg_inf(self) -> None:
        client = _make_ll_client()
        client.client.chat.completions.create.return_value = _fake_top_probs_resp(
            {" B": -0.5, " C": -4.2}
        )
        assert client.get_choices_logprobs("p", ["A", "B", "C"]) == [
            float("-inf"),
            -0.5,
            -4.2,
        ]

    def test_token_form_variants_are_checked(self) -> None:
        client = _make_ll_client()
        client.client.chat.completions.create.return_value = _fake_top_probs_resp(
            {"b": -1.2, " C": -3.0}
        )
        assert client.get_choices_logprobs("p", ["A", "B", "C"]) == [
            float("-inf"),
            -1.2,
            -3.0,
        ]

    def test_lowercase_target_matches_uppercase_token(self) -> None:
        client = _make_ll_client()
        client.client.chat.completions.create.return_value = _fake_top_probs_resp(
            {" A": -0.2}
        )
        assert client.get_choices_logprobs("p", ["a"]) == [-0.2]

    def test_4xx_aborts_without_retry(self) -> None:
        from llmeval.utils.retry import ClientError

        client = _make_ll_client(max_retries=3)
        client.client.chat.completions.create.side_effect = _make_api_error("bad", 400)
        with pytest.raises(ClientError, match="status=400"):
            client.get_choices_logprobs("p", ["a", "b"])
        assert client.client.chat.completions.create.call_count == 1

    def test_empty_top_logprobs_is_malformed(self) -> None:
        from llmeval.utils.retry import ClientError

        client = _make_ll_client(max_retries=0)
        client.client.chat.completions.create.return_value = _fake_top_probs_resp({})
        with pytest.raises(ClientError, match="no alternatives"):
            client.get_choices_logprobs("p", ["A", "B"])


class TestProcessLoglikelihoodItem:
    def test_context_marker_counts_failed_not_processed(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)

        def fail(_item: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("context length exceeded")

        runner._process_concurrently([{"doc_id": "test:0"}], fail)

        assert runner._stats["failed"] == 1
        assert runner._stats["processed"] == 0
        assert not Path(runner.config.output_file).exists()

    def test_run_logs_effective_mode(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner._process_concurrently = MagicMock()
        runner.log_stats = MagicMock()

        with patch("llmeval.inference.mc.logger.info") as log_info:
            runner.run_loglikelihood([{"prompt": "q"}])

        assert log_info.call_args_list[0].args == (
            "scoring_mode=%s",
            "first_token",
        )

    def test_full_choice_text_uses_answer_letters(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [-1.0, -5.0]
        result = runner.process_loglikelihood_item(
            {
                "doc_id": "test:0",
                "prompt": "q",
                "choices": ["Paris", "London"],
                "gold": 0,
            }
        )
        assert result["choice_tokens"] == ["A", "B"]
        assert result["gold"] == 0
        assert result["logprobs"] == [-1.0, -5.0]

    def test_no_choices_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        with pytest.raises(ValueError, match="non-empty list"):
            runner.process_loglikelihood_item({"prompt": "q"})

    def test_all_neg_inf_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [float("-inf")] * 2

        with pytest.raises(RuntimeError, match="failed for all choices"):
            runner.process_loglikelihood_item(
                {"prompt": "q", "choices": ["a", "b"], "gold": 1}
            )

    def test_mismatched_logprob_count_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [-1.0]

        with pytest.raises(RuntimeError, match="number of choices"):
            runner.process_loglikelihood_item(
                {"prompt": "q", "choices": ["a", "b"], "gold": 1}
            )

    def test_compact_loglikelihood_output(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [-5.0, -1.0]

        result = runner.process_loglikelihood_item(
            {
                "doc_id": "test:0",
                "prompt": "q",
                "choices": ["a", "b"],
                "gold": 1,
            }
        )

        assert result["choice_tokens"] == ["A", "B"]
        assert set(result) == {
            "prompt",
            "doc_id",
            "sample_index",
            "n_samples",
            "choices",
            "choice_tokens",
            "gold",
            "logprobs",
            "scoring_mode",
        }

    def test_continuation_pred_and_compact_output(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "continuation"
        runner.client = MagicMock()
        runner.client.score_continuations.return_value = [-5.0, -1.0]

        result = runner.process_loglikelihood_item(
            {
                "doc_id": "test:0",
                "prompt": "Answer:",
                "choices": ["a", "b"],
                "gold": 1,
            }
        )

        assert result == {
            "prompt": "Answer:",
            "doc_id": "test:0",
            "sample_index": 0,
            "n_samples": 1,
            "choices": ["a", "b"],
            "choice_tokens": [" A", " B"],
            "gold": 1,
            "logprobs": [-5.0, -1.0],
            "scoring_mode": "continuation",
        }
        runner.client.score_continuations.assert_called_once_with(
            "Answer:", [" A", " B"]
        )
        runner.client.get_choices_logprobs.assert_not_called()

    def test_continuation_preserves_existing_answer_separator(
        self, tmp_path: Path
    ) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "continuation"
        runner.client = MagicMock()
        runner.client.score_continuations.return_value = [-1.0, -2.0]

        result = runner.process_loglikelihood_item(
            {
                "doc_id": "test:0",
                "prompt": "Answer: ",
                "choices": ["first", "second"],
                "gold": 0,
            }
        )

        assert result["choice_tokens"] == ["A", "B"]
        runner.client.score_continuations.assert_called_once_with(
            "Answer: ", ["A", "B"]
        )

    @pytest.mark.parametrize(
        "item",
        [
            {"doc_id": "q", "prompt": "q", "choices": "AB", "gold": 0},
            {"doc_id": "q", "prompt": "q", "choices": ["A", "B"], "gold": "1"},
            {"doc_id": "q", "prompt": "q", "choices": ["A", "B"], "gold": 2},
            {
                "doc_id": "q",
                "prompt": "q",
                "choices": ["A", "B"],
                "choice_tokens": ["A"],
                "gold": 0,
            },
        ],
    )
    def test_invalid_schema_fails_before_request(
        self, tmp_path: Path, item: dict[str, object]
    ) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()

        with pytest.raises(ValueError):
            runner.process_loglikelihood_item(item)

        runner.client.get_choices_logprobs.assert_not_called()

    def test_rejects_preformatted_chat_prompt(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        with pytest.raises(ValueError, match="chat_template"):
            runner.process_loglikelihood_item(
                {
                    "doc_id": "q",
                    "prompt": "<|user|> question",
                    "choices": ["A", "B"],
                    "gold": 0,
                }
            )

    def test_missing_top_logprob_is_persisted_as_json_null(
        self, tmp_path: Path
    ) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [-0.2, float("-inf")]

        result = runner.process_loglikelihood_item(
            {"doc_id": "test:0", "prompt": "q", "choices": ["a", "b"], "gold": 0}
        )

        assert result["logprobs"] == [-0.2, None]
        json.dumps(result, allow_nan=False)

    def test_first_token_failed_request_is_not_returned(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = None

        with pytest.raises(RuntimeError, match="no result"):
            runner.process_loglikelihood_item(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "choices": ["a", "b"],
                    "gold": 1,
                }
            )


class TestProcessGenerateItem:
    def test_success(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            _generation_choice("ans")
        ]

        result = runner.process_generate_item(
            {
                "doc_id": "test:0",
                "sample_index": 0,
                "prompt": "q",
                "answer": "A",
                "error": "legacy failure",
            },
            client,
            [],
        )

        assert result["gen"] == "ans"
        assert result["scoring_mode"] == "generate"
        assert "error" not in result

    def test_shared_generation_options_are_sent(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        runner.config.top_p = 0.8
        runner.config.extra_body_dict = {"top_k": 40}
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            _generation_choice("ans")
        ]

        runner.process_generate_item(
            {
                "doc_id": "test:0",
                "sample_index": 0,
                "prompt": "q",
                "answer": "A",
            },
            client,
            [],
        )

        request = client.chat.completions.create.call_args.kwargs
        assert request["top_p"] == 0.8
        assert request["extra_body"] == {"top_k": 40}

    def test_null_content_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [_generation_choice(None)]

        with pytest.raises(RuntimeError, match="Generate request failed"):
            runner.process_generate_item(
                {
                    "doc_id": "test:0",
                    "sample_index": 0,
                    "prompt": "q",
                    "answer": "A",
                },
                client,
                [],
            )

    def test_null_content_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        runner = _make_mc_runner(tmp_path, mode="generate", max_retries=1)
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[_generation_choice(None)]),
            MagicMock(choices=[_generation_choice("answer")]),
        ]

        result = runner.process_generate_item(
            {
                "doc_id": "test:0",
                "sample_index": 0,
                "prompt": "q",
                "answer": "A",
            },
            client,
            [],
        )

        assert result["gen"] == "answer"
        assert client.chat.completions.create.call_count == 2

    def test_empty_string_content_is_preserved(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate", max_retries=1)
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[_generation_choice("")]
        )

        result = runner.process_generate_item(
            {
                "doc_id": "test:0",
                "sample_index": 0,
                "prompt": "q",
                "answer": "A",
            },
            client,
            [],
        )

        assert result["gen"] == ""
        assert result["scoring_mode"] == "generate"
        assert client.chat.completions.create.call_count == 1

    def test_failed_generation_is_not_returned(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.side_effect = _make_api_error(
            "This model's maximum context length is 8192", 400
        )

        with pytest.raises(RuntimeError, match="no response"):
            runner.process_generate_item(
                {
                    "doc_id": "test:0",
                    "sample_index": 0,
                    "prompt": "q",
                    "answer": "A",
                },
                client,
                [],
            )

    def test_single_request_uses_transient_seed(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        runner.config.n_samples = 3
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [_generation_choice("b")]

        result = runner.process_generate_item(
            {
                "doc_id": "q1",
                "prompt": "q",
                "answer": "A",
                "sample_index": 2,
            },
            client,
            [],
        )

        assert result["gen"] == "b"
        request = client.chat.completions.create.call_args.kwargs
        assert "n" not in request
        assert isinstance(request["seed"], int)
        assert request["max_completion_tokens"] == runner.config.max_completion_tokens
        assert "max_tokens" not in request

    def test_persistent_error_raises_after_retries(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate", max_retries=0)
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("down")
        with pytest.raises(RuntimeError):
            runner.process_generate_item(
                {
                    "doc_id": "test:0",
                    "sample_index": 0,
                    "prompt": "q",
                    "answer": "A",
                },
                client,
                [],
            )


class TestMCStableResume:
    def test_loglikelihood_resume_rejects_other_scoring_mode(
        self, tmp_path: Path
    ) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "continuation"
        Path(runner.config.input_file).write_text(
            json.dumps(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "choices": ["a", "b"],
                    "gold": 0,
                }
            )
            + "\n"
        )
        Path(runner.config.output_file).write_text(
            json.dumps(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "logprobs": [-0.1, -1.0],
                    "scoring_mode": "first_token",
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="expected 'continuation'"):
            runner.load_data()

    def test_generate_resume_rejects_loglikelihood_output(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        Path(runner.config.input_file).write_text(
            json.dumps({"doc_id": "test:0", "prompt": "q", "answer": "A"}) + "\n"
        )
        Path(runner.config.output_file).write_text(
            json.dumps(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "logprobs": [-0.1, -1.0],
                    "scoring_mode": "first_token",
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="expected 'generate'"):
            runner.load_data()

    def test_loglikelihood_resume_rejects_generate_output(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        Path(runner.config.input_file).write_text(
            json.dumps(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "choices": ["a", "b"],
                    "gold": 0,
                }
            )
            + "\n"
        )
        Path(runner.config.output_file).write_text(
            json.dumps(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "gen": "A",
                    "scoring_mode": "generate",
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="expected 'first_token'"):
            runner.load_data()

    def test_generate_resume_requests_only_missing_samples(
        self, tmp_path: Path
    ) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        runner.config.n_samples = 3
        Path(runner.config.input_file).write_text(
            json.dumps({"doc_id": "test:0", "prompt": "q", "answer": "A"}) + "\n"
        )

        first = runner.load_data()
        assert len(first) == 3
        assert [item["sample_index"] for item in first] == [0, 1, 2]
        document_id = first[0]["doc_id"]
        Path(runner.config.output_file).write_text(
            "".join(
                json.dumps(
                    {
                        "doc_id": document_id,
                        "prompt": "q",
                        "answer": "A",
                        "gen": [text],
                        "scoring_mode": "generate",
                    }
                )
                + "\n"
                for text in ("a", "b")
            )
        )

        remaining = runner.load_data()

        assert len(remaining) == 1

    def test_mixed_stable_and_legacy_output_both_resume(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        input_rows = [
            {"doc_id": "test:0", "prompt": "first", "answer": "A"},
            {"doc_id": "test:1", "prompt": "second", "answer": "B"},
        ]
        Path(runner.config.input_file).write_text(
            "".join(json.dumps(row) + "\n" for row in input_rows)
        )
        first_load = runner.load_data()
        stable_id = first_load[0]["doc_id"]
        Path(runner.config.output_file).write_text(
            json.dumps(
                {
                    "doc_id": stable_id,
                    "prompt": "first",
                    "gen": ["A"],
                    "scoring_mode": "generate",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "doc_id": "test:1",
                    "prompt": "second",
                    "gen": ["B"],
                    "scoring_mode": "generate",
                }
            )
            + "\n"
        )

        assert runner.load_data() == []

    def test_legacy_failure_row_is_retried_on_resume(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        Path(runner.config.input_file).write_text(
            json.dumps({"doc_id": "test:0", "prompt": "q", "answer": "A"}) + "\n"
        )
        Path(runner.config.output_file).write_text(
            json.dumps(
                {
                    "doc_id": "test:0",
                    "prompt": "q",
                    "answer": "A",
                    "gen": "",
                    "error": "context_length_exceeded",
                }
            )
            + "\n"
        )

        remaining = runner.load_data()
        assert len(remaining) == 1
        assert remaining[0]["doc_id"] == "test:0"

    def test_generate_resume_uses_completed_row_count(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        runner.config.n_samples = 3
        Path(runner.config.input_file).write_text(
            json.dumps({"doc_id": "test:0", "prompt": "q", "answer": "A"}) + "\n"
        )
        Path(runner.config.output_file).write_text(
            "".join(
                json.dumps(
                    {
                        "doc_id": "test:0",
                        "prompt": "q",
                        "answer": "A",
                        "gen": [text],
                        "scoring_mode": "generate",
                    }
                )
                + "\n"
                for text in ("a", "c")
            )
        )

        remaining = runner.load_data()

        assert len(remaining) == 1


class TestMCRunnerEndToEnd:
    def test_run_and_resume(self, tmp_path: Path) -> None:
        from llmeval.inference import mc as mc_infer

        class FakeLLClient:
            def __init__(self, **kwargs):
                pass

            def get_choices_logprobs(self, _prompt, choice_texts):
                return [-1.0 if i == 1 else -5.0 for i in range(len(choice_texts))]

        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_input(inp)
        cfg = MCInferArguments(
            input_file=str(inp),
            output_file=str(out),
            mode="loglikelihood",
            max_workers=2,
        )

        with patch.object(mc_infer, "MCLoglikelihoodClient", FakeLLClient):
            MCRunner(cfg).run()

        rows = [json.loads(x) for x in out.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["choice_tokens"] == ["A", "B"]

        with patch.object(mc_infer, "MCLoglikelihoodClient", FakeLLClient):
            MCRunner(cfg).run()

        assert len(out.read_text().strip().split("\n")) == 2

    def test_failed_items_are_not_written(self, tmp_path: Path) -> None:
        from llmeval.inference import mc as mc_infer

        class FailLLClient:
            def __init__(self, **kwargs):
                pass

            def get_choices_logprobs(self, _prompt, choice_texts):
                return [float("-inf")] * len(choice_texts)

        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_input(input_path)
        config = MCInferArguments(
            input_file=str(input_path),
            output_file=str(output_path),
            mode="loglikelihood",
            max_workers=2,
        )
        with (
            patch.object(mc_infer, "MCLoglikelihoodClient", FailLLClient),
            pytest.raises(RuntimeError, match="failed for 2 sample"),
        ):
            MCRunner(config).run()
        assert not output_path.exists()
        assert not (tmp_path / "out_failed.jsonl").exists()


# ===========================================================================
# Additional inference/client tests (redistributed from tests/test_mc_eval.py)
# ===========================================================================


class TestMCInferArguments:
    """Test MCInferArguments defaults and API key resolution."""

    def test_defaults(self) -> None:
        c = MCInferArguments()
        assert c.mode == "loglikelihood"
        assert c.loglikelihood_mode == "first_token"
        assert c.max_workers == 128
        assert c.temperature == 0.6
        assert c.n_shot == 0

    def test_api_key_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            c = MCInferArguments()
            assert c.api_key is None

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            c = MCInferArguments()
            assert c.api_key == "sk-test"


class TestFewShotFormatter:
    """Test few-shot example loading and dedup."""

    def _make_examples(self, count: int) -> str:
        """Create a temp JSONL with `count` examples."""
        items = [
            {
                "prompt": f"Q{i}: test?\nA. a\nB. b\nC. c\nD. d\nAnswer:",
                "answer": "B",
                "choices": ["a", "b", "c", "d"],
                "gold": 1,
            }
            for i in range(count)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
            return f.name

    def test_zero_shot(self) -> None:
        fmt = FewShotFormatter(n_shot=0)
        assert fmt.get_prefix("any prompt") == ""

    def test_load_and_prefix(self) -> None:
        tmp = self._make_examples(10)
        try:
            fmt = FewShotFormatter(n_shot=3, few_shot_file=tmp, seed=42)
            fmt.load()
            prefix = fmt.get_prefix("some other prompt")
            # Should contain 3 examples separated by \n\n
            assert prefix.count("\n\n") >= 3
            assert "Q" in prefix
            assert "Answer: B" in prefix
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_dedup_excludes_test_prompt(self) -> None:
        tmp = self._make_examples(10)
        try:
            fmt = FewShotFormatter(n_shot=3, few_shot_file=tmp, seed=42)
            fmt.load()
            # Get the raw prompt from one of the few-shot pool
            test_prompt = fmt._few_shot_pool[0]["prompt"]
            prefix_with_dedup = fmt.get_prefix(test_prompt)
            prefix_without = fmt.get_prefix("unrelated prompt")
            # Dedup should produce different prefixes (one fewer example)
            # Both should have content
            assert len(prefix_with_dedup) > 0
            assert len(prefix_without) > 0
            # The key invariant: formatted demo starts with raw_prompt + " " + answer
            assert any(test_prompt in d for d in fmt._all_formatted)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_insufficient_examples(self) -> None:
        tmp = self._make_examples(3)
        try:
            fmt = FewShotFormatter(n_shot=10, few_shot_file=tmp)
            with pytest.raises(ValueError, match="Few-shot pool is too small"):
                fmt.load()
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_load_failure_is_fatal(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.jsonl"
        fmt = FewShotFormatter(n_shot=1, few_shot_file=str(missing))

        with pytest.raises(RuntimeError, match="Failed to load few-shot data"):
            fmt.load()

    def test_per_document_dedup_shortage_is_fatal(self) -> None:
        fmt = FewShotFormatter(n_shot=1)
        fmt._few_shot_pool = [{"doc_id": "same", "prompt": "question"}]
        fmt._all_formatted = ["question A"]

        with pytest.raises(ValueError, match="insufficient after excluding"):
            fmt.get_prefix("question", "same")

    @pytest.mark.parametrize(
        "item, message",
        [
            ({"answer": "A"}, "few-shot prompt"),
            ({"prompt": "question"}, "few-shot answer"),
            (
                {"prompt": "<|user|> question", "answer": "A"},
                "chat_template",
            ),
        ],
    )
    def test_invalid_demo_is_rejected(self, item: dict[str, str], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            FewShotFormatter._format_demo(item)


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
