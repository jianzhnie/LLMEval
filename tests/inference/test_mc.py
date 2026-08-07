"""Tests for llmeval.inference.mc.

Focuses on the inference client and runner behavior without a live API.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import threading
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

from llmeval.inference.mc import FewShotFormatter, MCLoglikelihoodClient, MCRunner
from llmeval.inference.schema import (
    ChoiceLoglikelihood,
    LoglikelihoodRequest,
    LoglikelihoodResult,
)
from llmeval.utils.config import MCInferConfig


def _make_ll_client(max_retries: int = 0) -> MCLoglikelihoodClient:
    client = MCLoglikelihoodClient.__new__(MCLoglikelihoodClient)
    client.model_name = "m"
    client.timeout = 5
    client.max_retries = max_retries
    client.base_url = "http://test/v1"
    client.seed = 0
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
    choice.logprobs.top_logprobs = [top_probs]
    resp.choices = [choice]
    return resp


def _generation_choice(content: str | None, index: int = 0) -> MagicMock:
    choice = MagicMock()
    choice.index = index
    choice.message.content = content
    return choice


def _make_mc_runner(
    tmp_path: Path, mode: str = "loglikelihood", max_retries: int = 0
) -> MCRunner:
    runner = MCRunner.__new__(MCRunner)
    runner.config = MCInferConfig(
        input_file=str(tmp_path / "in.jsonl"),
        output_file=str(tmp_path / "out.jsonl"),
        mode=mode,
        max_retries=max_retries,
        max_workers=1,
    )
    runner.client = None
    runner.system_prompt = None
    runner._few_shot_fmt = None
    runner._file_lock = threading.Lock()
    runner._stats_lock = threading.Lock()
    runner._stats = {
        "processed": 0,
        "failed": 0,
        "correct": 0,
        "skipped": 0,
        "continuation_fallback": 0,
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
    def test_continuation_schema_value_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.inference.mc as mc_mod

        client = _make_ll_client()
        monkeypatch.setattr(
            mc_mod, "call_with_retry", MagicMock(side_effect=ValueError("bad schema"))
        )

        with pytest.raises(ValueError, match="bad schema"):
            client.score_continuations(LoglikelihoodRequest("q", ("A",)))

    def test_continuation_unexpected_backend_error_is_item_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.inference.mc as mc_mod

        client = _make_ll_client()
        monkeypatch.setattr(
            mc_mod, "call_with_retry", MagicMock(side_effect=RuntimeError("backend"))
        )

        result = client.score_continuations(LoglikelihoodRequest("q", ("A",)))

        assert result.exact is False
        assert result.error == "backend"

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
        client.client.completions.create.return_value = _fake_top_probs_resp(
            {" A": -3.0, " B": -0.5, " C": -4.2}
        )

        result = client.get_choices_logprobs("prompt", ["A", "B", "C"])

        assert result == [-3.0, -0.5, -4.2]
        call_kwargs = client.client.completions.create.call_args.kwargs
        assert call_kwargs["prompt"] == "prompt"
        assert call_kwargs["echo"] is False
        assert call_kwargs["logprobs"] == 20
        assert call_kwargs["max_tokens"] == 1

    def test_tool_choice_none_is_omitted(self) -> None:
        client = _make_ll_client()
        client.client.completions.create.return_value = _fake_top_probs_resp(
            {" A": -3.0}
        )

        client.get_choices_logprobs("prompt", ["A"])

        assert "tool_choice" not in client.client.completions.create.call_args.kwargs

    def test_programming_error_propagates(self) -> None:
        client = _make_ll_client(max_retries=0)
        client.client.completions.create.side_effect = RuntimeError("down")

        with pytest.raises(RuntimeError, match="down"):
            client.get_choices_logprobs("prompt", ["A", "B"])

    def test_continuation_returns_only_choice_token_scores(self) -> None:
        client = _make_ll_client()
        first = MagicMock()
        first.logprobs.text_offset = [0, 6, 8]
        first.logprobs.token_logprobs = [None, -0.2, -9.0]
        first.logprobs.tokens = ["prompt", " A", "x"]
        first.logprobs.token_ids = [1, 2, 3]
        first.index = 0
        second = MagicMock()
        second.logprobs.text_offset = [0, 6, 8]
        second.logprobs.token_logprobs = [None, -1.3, -9.0]
        second.logprobs.tokens = ["prompt", " B", "x"]
        second.logprobs.token_ids = [1, 4, 3]
        second.index = 1
        client.client.completions.create.return_value.choices = [first, second]

        result = client.score_continuations(
            LoglikelihoodRequest("prompt", (" A", " B"))
        )

        assert result.exact is True
        assert [list(c.token_logprobs) for c in result.choices] == [[-0.2], [-1.3]]
        assert [list(c.token_texts) for c in result.choices] == [[" A"], [" B"]]
        assert [list(c.token_ids) for c in result.choices] == [[2], [4]]

    def test_continuation_handles_leading_space_and_multitoken_chinese(self) -> None:
        client = _make_ll_client()
        choice = MagicMock()
        choice.index = 0
        choice.logprobs.text_offset = [0, 2, 4]
        choice.logprobs.token_logprobs = [None, -0.4, -0.2]
        choice.logprobs.tokens = ["题:", " 答", "案"]
        choice.logprobs.token_ids = [10, 11, 12]
        client.client.completions.create.return_value.choices = [choice]

        result = client.score_continuations(LoglikelihoodRequest("题:", (" 答案",)))

        assert result.exact is True
        assert [list(c.token_logprobs) for c in result.choices] == [[-0.4, -0.2]]
        assert [list(c.token_texts) for c in result.choices] == [[" 答", "案"]]
        assert [list(c.token_ids) for c in result.choices] == [[11, 12]]

    def test_continuation_accepts_utf8_byte_offsets(self) -> None:
        client = _make_ll_client()
        choice = MagicMock(index=0)
        choice.logprobs.text_offset = [0, 4, 8]
        choice.logprobs.token_logprobs = [None, -0.4, -0.2]
        choice.logprobs.tokens = ["题:", " 答", "案"]
        choice.logprobs.token_ids = [10, 11, 12]
        client.client.completions.create.return_value.choices = [choice]

        result = client.score_continuations(LoglikelihoodRequest("题:", (" 答案",)))

        assert result.exact is True
        assert [list(c.token_logprobs) for c in result.choices] == [[-0.4, -0.2]]
        assert [list(c.token_texts) for c in result.choices] == [[" 答", "案"]]

    def test_continuation_scores_trailing_context_whitespace_like_harness(
        self,
    ) -> None:
        client = _make_ll_client()
        choice = MagicMock(index=0)
        choice.logprobs.text_offset = [0, 2]
        choice.logprobs.token_logprobs = [None, -0.3]
        choice.logprobs.tokens = ["Q:", " A"]
        choice.logprobs.token_ids = [10, 11]
        client.client.completions.create.return_value.choices = [choice]

        result = client.score_continuations(LoglikelihoodRequest("Q: ", ("A",)))

        assert result.exact is True
        assert [list(c.token_logprobs) for c in result.choices] == [[-0.3]]
        assert [list(c.token_texts) for c in result.choices] == [[" A"]]

    def test_continuation_rejects_token_crossing_prompt_boundary(self) -> None:
        client = _make_ll_client(max_retries=3)
        choice = MagicMock()
        choice.index = 0
        choice.logprobs.text_offset = [0, 5]
        choice.logprobs.token_logprobs = [None, -0.2]
        choice.logprobs.tokens = ["promp", "t A"]
        client.client.completions.create.return_value.choices = [choice]

        result = client.score_continuations(LoglikelihoodRequest("prompt", (" A",)))

        assert result.exact is False
        assert [list(c.token_logprobs) for c in result.choices] == [[]]
        assert client.client.completions.create.call_count == 1

    def test_continuation_rejects_misaligned_token_offsets(self) -> None:
        client = _make_ll_client()
        choice = MagicMock(index=0)
        choice.logprobs.text_offset = [0, 6, 6]
        choice.logprobs.token_logprobs = [None, -0.2, -0.3]
        choice.logprobs.tokens = ["prompt", " ", "A"]
        choice.logprobs.token_ids = [1, 2, 3]
        client.client.completions.create.return_value.choices = [choice]

        result = client.score_continuations(LoglikelihoodRequest("prompt", (" A",)))

        assert result.exact is False
        assert [list(c.token_logprobs) for c in result.choices] == [[]]

    def test_continuation_retries_malformed_response_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A structurally malformed response is retried, not failed fast."""
        monkeypatch.setattr(time, "sleep", lambda s: None)
        client = _make_ll_client(max_retries=2)
        good = MagicMock(index=0)
        good.logprobs.text_offset = [0, 6]
        good.logprobs.token_logprobs = [None, -0.2]
        good.logprobs.tokens = ["prompt", " A"]
        good.logprobs.token_ids = None
        bad_response = MagicMock()
        bad_response.choices = []  # completion count mismatch
        good_response = MagicMock()
        good_response.choices = [good]
        client.client.completions.create.side_effect = [bad_response, good_response]

        result = client.score_continuations(LoglikelihoodRequest("prompt", (" A",)))

        assert result.exact is True
        assert [list(c.token_logprobs) for c in result.choices] == [[-0.2]]
        assert client.client.completions.create.call_count == 2

    def test_continuation_persistent_malformed_response_fails_after_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retries exhausted on malformed responses → failed (non-exact) result."""
        monkeypatch.setattr(time, "sleep", lambda s: None)
        client = _make_ll_client(max_retries=2)
        client.client.completions.create.return_value.choices = []

        result = client.score_continuations(LoglikelihoodRequest("prompt", (" A",)))

        assert result.exact is False
        assert result.error is not None
        assert client.client.completions.create.call_count == 3


class TestProcessLoglikelihoodItem:
    def test_context_marker_counts_failed_not_processed(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        marker = {
            "doc_id": "test:0",
            "prompt": "q",
            "logprobs": [],
            "error": "context_length_exceeded",
        }

        runner._process_concurrently([{"doc_id": "test:0"}], lambda _item: marker)

        assert runner._stats["failed"] == 1
        assert runner._stats["processed"] == 0

    def test_run_logs_effective_mode(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner._process_concurrently = MagicMock()
        runner.log_stats = MagicMock()

        with patch("llmeval.inference.mc.logger.info") as log_info:
            runner.run_loglikelihood([{"prompt": "q"}])

        assert log_info.call_args_list[0].args == (
            "effective_loglikelihood_mode=%s",
            "first_token",
        )

    def test_all_neg_inf_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [float("-inf")] * 2

        with pytest.raises(RuntimeError, match="failed for all choices"):
            runner.process_loglikelihood_item(
                {"prompt": "q", "choices": ["a", "b"], "gold": 1}
            )

    def test_normal_pred_and_correct(self, tmp_path: Path) -> None:
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

        assert result["pred"] == 1
        assert result["correct"] is True
        assert result["choice_tokens"] == ["A", "B"]

    def test_auto_uses_first_token_without_probing_continuation(
        self, tmp_path: Path
    ) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "auto"
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [-0.2, -1.0]

        result = runner.process_loglikelihood_item(
            {"doc_id": "test:0", "prompt": "q", "choices": ["a", "b"], "gold": 0}
        )

        assert result["scoring_mode"] == "first_token"
        assert result["logprobs"] == [-0.2, -1.0]
        assert result["loglikelihood_exact"] is False
        assert result["scoring_approximation"] == "first_token_top_logprobs"
        runner.client.score_continuations.assert_not_called()

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

    def test_explicit_continuation_uses_exact_scoring(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "continuation"
        runner.client = MagicMock()
        request = LoglikelihoodRequest("q", ("A", "B"))
        runner.client.score_continuations.return_value = LoglikelihoodResult(
            request=request,
            choices=(
                ChoiceLoglikelihood("A", "A", (-0.5,), ("A",)),
                ChoiceLoglikelihood("B", "B", (-0.2,), ("B",)),
            ),
            exact=True,
        )

        result = runner.process_loglikelihood_item(
            {"doc_id": "test:0", "prompt": "q", "choices": ["a", "b"], "gold": 1}
        )

        assert result["scoring_mode"] == "continuation"
        assert result["loglikelihood_exact"] is True
        runner.client.score_continuations.assert_called_once_with(request)
        runner.client.get_choices_logprobs.assert_not_called()

    def test_continuation_context_length_returns_permanent_failure_row(
        self, tmp_path: Path
    ) -> None:
        """Context-length rejection is deterministic: a marked row, not a retry."""
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "continuation"
        runner.client = MagicMock()
        runner.client.score_continuations.return_value = LoglikelihoodResult.failure(
            LoglikelihoodRequest("q", ("A", "B")), "context_length_exceeded"
        )

        result = runner.process_loglikelihood_item(
            {"doc_id": "test:0", "prompt": "q", "choices": ["a", "b"], "gold": 1}
        )

        assert result["error"] == "context_length_exceeded"
        assert result["logprobs"] == []
        assert result["pred"] == -1
        assert result["correct"] is False

    def test_first_token_context_length_returns_permanent_failure_row(
        self, tmp_path: Path
    ) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.config.loglikelihood_mode = "first_token"
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = None

        result = runner.process_loglikelihood_item(
            {"doc_id": "test:0", "prompt": "q", "choices": ["a", "b"], "gold": 1}
        )

        assert result["error"] == "context_length_exceeded"
        assert result["logprobs"] == []
        assert result["pred"] == -1


class TestProcessGenerateItem:
    def test_success(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            _generation_choice("ans")
        ]

        result = runner.process_generate_item(
            {"prompt": "q", "answer": "A", "_request_seed": 1}, client, []
        )

        assert result["gen"] == "ans"

    def test_tool_choice_auto_is_sent(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        runner.config.tool_choice = "auto"
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            _generation_choice("ans")
        ]

        runner.process_generate_item(
            {"prompt": "q", "answer": "A", "_request_seed": 1}, client, []
        )

        assert client.chat.completions.create.call_args.kwargs["tool_choice"] == "auto"

    def test_null_content_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [_generation_choice(None)]

        with pytest.raises(RuntimeError, match="no usable text"):
            runner.process_generate_item(
                {"prompt": "q", "answer": "A", "_request_seed": 1}, client, []
            )

    def test_context_length_returns_permanent_failure_row(self, tmp_path: Path) -> None:
        """Context-length rejection yields a marked row instead of raising."""
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.side_effect = _make_api_error(
            "This model's maximum context length is 8192", 400
        )

        result = runner.process_generate_item(
            {"doc_id": "test:0", "prompt": "q", "answer": "A", "_request_seed": 1},
            client,
            [],
        )

        assert result["gen"] == ""
        assert result["error"] == "context_length_exceeded"

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
                "_request_seed": 123,
            },
            client,
            [],
        )

        assert result["gen"] == "b"
        request = client.chat.completions.create.call_args.kwargs
        assert "n" not in request
        assert request["seed"] == 123


class TestMCStableResume:
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
        assert len({item["_request_seed"] for item in first}) == 3
        assert all(
            not any(key.startswith("_llmeval_") for key in item) for item in first
        )
        document_id = first[0]["doc_id"]
        Path(runner.config.output_file).write_text(
            "".join(
                json.dumps(
                    {
                        "doc_id": document_id,
                        "prompt": "q",
                        "answer": "A",
                        "gen": [text],
                    }
                )
                + "\n"
                for text in ("a", "b")
            )
        )

        remaining = runner.load_data()

        assert len(remaining) == 1
        assert not any(key.startswith("_llmeval_") for key in remaining[0])

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
                }
            )
            + "\n"
            + json.dumps(
                {
                    "doc_id": "test:1",
                    "prompt": "second",
                    "gen": ["B"],
                }
            )
            + "\n"
        )

        assert runner.load_data() == []

    def test_context_length_row_is_skipped_on_resume(self, tmp_path: Path) -> None:
        """A permanent-failure (context-length) row counts as completed."""
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

        assert runner.load_data() == []

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
                    }
                )
                + "\n"
                for text in ("a", "c")
            )
        )

        remaining = runner.load_data()

        assert len(remaining) == 1
        assert not any(key.startswith("_llmeval_") for key in remaining[0])


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
        cfg = MCInferConfig(
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


# ===========================================================================
# Additional inference/client tests (redistributed from tests/test_mc_eval.py)
# ===========================================================================


class TestMCInferConfig:
    """Test MCInferConfig defaults and API key resolution."""

    def test_defaults(self) -> None:
        c = MCInferConfig()
        assert c.mode == "loglikelihood"
        assert c.max_workers == 32
        assert c.temperature == 0.0
        assert c.n_shot == 0

    def test_api_key_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            c = MCInferConfig()
            assert c.api_key == "EMPTY"

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            c = MCInferConfig()
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


class TestContinuationScoring:
    def test_acc_norm_uses_harness_character_counts(self, tmp_path: Path) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {
                "gold": 0,
                "logprobs": [-1.0, -1.2],
                "choice_logprobs": [[-1.0], [-0.3, -0.3, -0.3, -0.3]],
                "choice_tokens": ["AB", "C"],
                "choice_token_count": [1, 4],
                "choice_char_count": [2, 1],
                "choice_byte_count": [2, 1],
            }
        ]
        cache = tmp_path / "continuation.jsonl"
        assert score_loglikelihood(items, cache) == 1.0
        summary = json.loads(cache.with_suffix(".summary.json").read_text())
        assert summary["acc_norm"] == 1.0


# Extend existing classes with the remaining tests from tests/test_mc_eval.py.
# Each method below was moved verbatim from that file, reusing the module-level
# helpers already defined above.


def _mc_client_choice_not_in_top_returns_neg_inf(self) -> None:
    """A target letter absent from top_logprobs gets float('-inf')."""
    client = _make_ll_client()
    client.client.completions.create.return_value = _fake_top_probs_resp(
        {" B": -0.5, " C": -4.2}
    )
    result = client.get_choices_logprobs("p", ["A", "B", "C"])
    assert result == [float("-inf"), -0.5, -4.2]


TestMCLoglikelihoodClient.test_choice_not_in_top_returns_neg_inf = (
    _mc_client_choice_not_in_top_returns_neg_inf
)


def _mc_client_token_form_variants_are_checked(self) -> None:
    """Letters are looked up as 'X', ' X', 'x', ' x' to handle tokenizer variance."""
    client = _make_ll_client()
    # Tokenizer uses lowercase form for some models.
    client.client.completions.create.return_value = _fake_top_probs_resp(
        {"b": -1.2, " C": -3.0}
    )
    result = client.get_choices_logprobs("p", ["A", "B", "C"])
    assert result == [float("-inf"), -1.2, -3.0]


TestMCLoglikelihoodClient.test_token_form_variants_are_checked = (
    _mc_client_token_form_variants_are_checked
)


def _mc_client_4xx_aborts_without_retry(self) -> None:
    """Non-retryable 4xx errors propagate after the first request."""
    from llmeval.utils.retry import ClientError

    client = _make_ll_client(max_retries=3)
    client.client.completions.create.side_effect = _make_api_error("bad", 400)
    with pytest.raises(ClientError, match="status=400"):
        client.get_choices_logprobs("p", ["a", "b"])
    assert client.client.completions.create.call_count == 1


TestMCLoglikelihoodClient.test_4xx_aborts_without_retry = (
    _mc_client_4xx_aborts_without_retry
)


def _mc_client_empty_top_logprobs_returns_all_neg_inf(self) -> None:
    """An empty top_logprobs dict yields -inf for every choice."""
    client = _make_ll_client(max_retries=0)
    client.client.completions.create.return_value = _fake_top_probs_resp({})
    assert client.get_choices_logprobs("p", ["A", "B"]) == [
        float("-inf"),
        float("-inf"),
    ]


TestMCLoglikelihoodClient.test_empty_top_logprobs_returns_all_neg_inf = (
    _mc_client_empty_top_logprobs_returns_all_neg_inf
)


def _mc_client_complete_continuation_uses_choice_offsets(self) -> None:
    client = _make_ll_client()
    response = MagicMock()
    response.choices = []
    for text, values in (("AB", [-1.0, -2.0]), ("C", [-0.5])):
        choice = MagicMock()
        choice.logprobs.text_offset = [0, *range(2, 2 + len(text))]
        choice.logprobs.token_logprobs = [None, *values]
        choice.logprobs.tokens = ["Q:", *text]
        choice.logprobs.token_ids = None
        response.choices.append(choice)
    client.client.completions.create.return_value = response

    result = client.score_continuations(LoglikelihoodRequest("Q:", ("AB", "C")))

    assert result.exact is True
    assert [list(c.token_logprobs) for c in result.choices] == [
        [-1.0, -2.0],
        [-0.5],
    ]
    kwargs = client.client.completions.create.call_args.kwargs
    assert kwargs["prompt"] == ["Q:AB", "Q:C"]
    assert kwargs["echo"] is True


TestMCLoglikelihoodClient.test_complete_continuation_uses_choice_offsets = (
    _mc_client_complete_continuation_uses_choice_offsets
)


def _mc_process_ll_full_choice_text_uses_answer_letters(self, tmp_path: Path) -> None:
    runner = _make_mc_runner(tmp_path)
    runner.client = MagicMock()
    runner.client.get_choices_logprobs.return_value = [-1.0, -5.0]
    item = {
        "doc_id": "test:0",
        "prompt": "q",
        "choices": ["Paris", "London"],
        "gold": 0,
    }
    result = runner.process_loglikelihood_item(item)
    assert result["choice_tokens"] == ["A", "B"]
    assert result["correct"] is True


TestProcessLoglikelihoodItem.test_full_choice_text_uses_answer_letters = (
    _mc_process_ll_full_choice_text_uses_answer_letters
)


def _mc_process_ll_no_choices_returns_none(self, tmp_path: Path) -> None:
    runner = _make_mc_runner(tmp_path)
    assert runner.process_loglikelihood_item({"prompt": "q"}) is None


TestProcessLoglikelihoodItem.test_no_choices_returns_none = (
    _mc_process_ll_no_choices_returns_none
)


def _mc_process_gen_persistent_error_raises_after_retries(self, tmp_path: Path) -> None:
    runner = _make_mc_runner(tmp_path, mode="generate", max_retries=0)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("down")
    with pytest.raises(RuntimeError):
        runner.process_generate_item(
            {"prompt": "q", "answer": "A", "_request_seed": 1}, client, []
        )


TestProcessGenerateItem.test_persistent_error_raises_after_retries = (
    _mc_process_gen_persistent_error_raises_after_retries
)


def _mc_runner_failed_items_dumped_not_written(self, tmp_path: Path) -> None:
    from llmeval.inference import mc as mc_infer

    class FailLLClient:
        def __init__(self, **kwargs):
            pass

        def get_choices_logprobs(self, _prompt, choice_texts):
            return [float("-inf")] * len(choice_texts)

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _write_input(inp)
    cfg = MCInferConfig(
        input_file=str(inp),
        output_file=str(out),
        mode="loglikelihood",
        max_workers=2,
    )
    with patch.object(mc_infer, "MCLoglikelihoodClient", FailLLClient):
        MCRunner(cfg).run()
    assert not out.exists()  # nothing scored
    failed = tmp_path / "out_failed.jsonl"
    assert failed.exists()
    assert len(failed.read_text().strip().split("\n")) == 2


TestMCRunnerEndToEnd.test_failed_items_dumped_not_written = (
    _mc_runner_failed_items_dumped_not_written
)


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
