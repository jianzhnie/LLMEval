"""Tests for llmeval.inference.online.

Focuses on request behavior, thread-safe writing, and concurrent grouping --
all testable without a live vLLM server.  The shared data-loading / resume
helpers are covered by tests/test_inference_common.py.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Mock heavy dependencies ──
# Only stub modules that are genuinely absent — a bare ModuleType stub has
# __spec__ = None, which crashes importlib.util.find_spec (called inside the
# transformers import chain) with "ValueError: <pkg>.__spec__ is None".
def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


for mod_name in ("openai", "httpx"):
    if mod_name not in sys.modules and not importlib.util.find_spec(mod_name):
        sys.modules[mod_name] = _make_stub(mod_name)

# Provide stubs for openai exceptions used at module level (real openai
# already has them; only the stub needs patching)
_openai_mod = sys.modules.get("openai")
if _openai_mod is not None:
    for _exc in ("APIConnectionError", "APIError", "RateLimitError"):
        if not hasattr(_openai_mod, _exc):
            setattr(_openai_mod, _exc, type(_exc, (Exception,), {}))

# Only stub transformers if genuinely absent — a partial stub (just
# HfArgumentParser) would pollute sys.modules for other test modules needing
# the real AutoTokenizer.
if "transformers" not in sys.modules and not importlib.util.find_spec("transformers"):
    _tf = types.ModuleType("transformers")
    _tf.HfArgumentParser = MagicMock
    sys.modules["transformers"] = _tf

# tqdm is likely available, but just in case
if "tqdm" not in sys.modules and not importlib.util.find_spec("tqdm"):
    _tqdm = types.ModuleType("tqdm")
    _tqdm.tqdm = MagicMock
    sys.modules["tqdm"] = _tqdm

from llmeval.inference.common import expand_data_with_resume
from llmeval.inference.online import (
    InferenceRunner,
)

# ── helpers ───────────────────────────────────────────────────────


def _make_runner(tmp_path: Path, **overrides: Any) -> InferenceRunner:
    """Build an InferenceRunner with minimal setup for unit testing."""
    defaults: dict[str, Any] = {
        "input_file": str(tmp_path / "input.jsonl"),
        "output_file": str(tmp_path / "output.jsonl"),
        "base_url": "http://127.0.0.1:8090/v1",
        "model_name": "test-model",
        "n_samples": 1,
        "max_tokens": 128,
        "max_workers": 2,
        "input_key": "prompt",
        "label_key": "answer",
        "response_key": "gen",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "enable_thinking": False,
    }
    defaults.update(overrides)

    # Write minimal input
    inp = Path(defaults["input_file"])
    if not inp.exists():
        inp.parent.mkdir(parents=True, exist_ok=True)
        with open(inp, "w") as f:
            f.write(json.dumps({"prompt": "2+2", "answer": "4"}) + "\n")

    # Bypass __init__ entirely; set attributes manually
    runner = InferenceRunner.__new__(InferenceRunner)
    args = MagicMock()
    for k, v in defaults.items():
        setattr(args, k, v)
    runner.args = args
    runner.system_prompt = None
    runner._file_lock = threading.Lock()
    runner._stats_lock = threading.Lock()
    runner._stats = {"processed": 0, "failed": 0, "skipped": 0}
    return runner


# ── InferenceRunner._write_result ────────────────────────────────


class TestWriteResult:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        runner = _make_runner(tmp_path, output_file=str(out))
        runner._write_result({"prompt": "q", "gen": ["answer"]})

        lines = out.read_text().strip().split("\n")
        data = [json.loads(line) for line in lines]
        assert len(data) == 1
        assert data[0]["gen"] == ["answer"]

    def test_thread_safety(self, tmp_path: Path) -> None:
        """Multiple concurrent writes should not interleave JSON lines."""
        import concurrent.futures

        out = tmp_path / "concurrent.jsonl"
        runner = _make_runner(tmp_path, output_file=str(out))

        items = [{"prompt": f"q{i}", "gen": [f"a{i}"]} for i in range(20)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(runner._write_result, items))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 20
        for line in lines:
            data = json.loads(line)
            assert "prompt" in data
            assert "gen" in data


# ── InferenceRunner.process_item ────────────────────────────────


class TestProcessItem:
    def test_missing_query_returns_none(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        result = runner.process_item({"answer": "a"})
        assert result is None
        assert runner._stats["skipped"] == 1

    def test_non_dict_item_returns_none(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        result = runner.process_item("not a dict")
        assert result is None
        assert runner._stats["failed"] == 1

    def test_process_item_creates_independent_gen_list(self, tmp_path: Path) -> None:
        """Verify process_item doesn't mutate the input item."""
        runner = _make_runner(tmp_path)
        # Mock the client to return a simple response
        mock_client = MagicMock()
        mock_client.get_content.return_value = "test response"
        runner.client = mock_client

        item = {"prompt": "q", "answer": "a", "gen": ["existing"]}
        original_gen = item["gen"].copy()
        result = runner.process_item(item)

        # The original item's gen list should not be modified
        assert item["gen"] == original_gen
        assert result is not None
        assert result["gen"] == ["existing", "test response"]


# ── InferenceClient request behavior ──────────────────────────────


def _make_api_error(message: str = "", status_code: int | None = None) -> Exception:
    """Build an APIError instance compatible with the (possibly stubbed) openai module."""
    cls = sys.modules["openai"].APIError
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    err.message = message
    err.status_code = status_code
    return err


def _make_client(max_retries: int = 0):
    """InferenceClient bypassing __init__ (works with stubbed openai module)."""
    from llmeval.inference.online import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client.timeout = 5
    client.max_retries = max_retries
    client.tool_choice = "none"
    client.client = MagicMock()
    return client


def _fake_completion(contents: list[str | None]) -> MagicMock:
    """Completion whose choices carry the given contents."""
    completion = MagicMock()
    choices = []
    for content in contents:
        choice = MagicMock()
        choice.message.content = content
        choices.append(choice)
    completion.choices = choices
    return completion


class TestGetContent:
    def test_null_content_normalized_to_empty(self) -> None:
        """Reasoning model truncation returns content=None → ""."""
        client = _make_client()
        client.client.chat.completions.create.return_value = _fake_completion([None])
        result = client.get_content("q", None, "m", 8, 0.0, 1.0, 40, False)
        assert result == ""

    def test_context_length_returns_empty(self) -> None:
        client = _make_client()
        client.client.chat.completions.create.side_effect = _make_api_error(
            "This model's maximum context length is 8192"
        )
        assert client.get_content("q", None, "m", 8, 0.0, 1.0, 40, False) == ""

    def test_4xx_fails_fast_without_retry(self) -> None:
        client = _make_client(max_retries=3)
        client.client.chat.completions.create.side_effect = _make_api_error(
            "invalid", 400
        )
        from llmeval.utils.retry import ClientError

        with pytest.raises(ClientError, match="non-retryable"):
            client.get_content("q", None, "m", 8, 0.0, 1.0, 40, False)
        assert client.client.chat.completions.create.call_count == 1

    def test_5xx_retries_then_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(time, "sleep", lambda s: None)
        client = _make_client(max_retries=1)
        client.client.chat.completions.create.side_effect = _make_api_error("boom", 500)
        from llmeval.utils.retry import ClientError

        with pytest.raises(ClientError):
            client.get_content("q", None, "m", 8, 0.0, 1.0, 40, False)
        assert client.client.chat.completions.create.call_count == 2


class TestInferenceClientInit:
    def test_api_key_argument_takes_priority(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llmeval.inference import online as online_mod
        from llmeval.inference.online import InferenceClient

        fake_openai = MagicMock()
        monkeypatch.setattr(online_mod.openai, "OpenAI", fake_openai)
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        InferenceClient("http://example.com/v1", 5, api_key="cli-key")

        assert fake_openai.call_args.kwargs["api_key"] == "cli-key"


class TestGetContents:
    def test_n_parameter_sent_and_list_returned(self) -> None:
        client = _make_client()
        client.client.chat.completions.create.return_value = _fake_completion(
            ["a", "b", None]
        )
        result = client.get_contents("q", None, "m", 8, 0.6, 1.0, 40, False, n=3)
        assert result == ["a", "b", ""]  # null normalized
        assert client.client.chat.completions.create.call_args.kwargs["n"] == 3

    def test_single_sample_omits_n(self) -> None:
        client = _make_client()
        client.client.chat.completions.create.return_value = _fake_completion(["a"])
        client.get_contents("q", None, "m", 8, 0.6, 1.0, 40, False, n=1)
        assert "n" not in client.client.chat.completions.create.call_args.kwargs

    def test_context_length_returns_empty_list(self) -> None:
        client = _make_client()
        client.client.chat.completions.create.side_effect = _make_api_error(
            "This model's maximum context length is 8192"
        )
        assert client.get_contents("q", None, "m", 8, 0.6, 1.0, 40, False, n=4) == []

    def test_invalid_n_raises(self) -> None:
        client = _make_client()
        with pytest.raises(ValueError, match="n must be"):
            client.get_contents("q", None, "m", 8, 0.6, 1.0, 40, False, n=0)


# ── InferenceRunner.process_item_group (batched n-parameter path) ──


class TestProcessItemGroup:
    def test_batch_writes_one_line_per_copy(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=3)
        runner.client = MagicMock()
        runner.client.get_contents.return_value = ["s1", "s2", "s3"]

        items = [{"prompt": "q", "answer": "a"} for _ in range(3)]
        runner.process_item_group(items)

        runner.client.get_contents.assert_called_once()
        assert runner.client.get_contents.call_args.kwargs["n"] == 3
        lines = (tmp_path / "output.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        gens = sorted(json.loads(x)["gen"][0] for x in lines)
        assert gens == ["s1", "s2", "s3"]
        assert runner._stats["processed"] == 3

    def test_empty_sample_counted_failed(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=3)
        runner.client = MagicMock()
        runner.client.get_contents.return_value = ["s1", "", "s3"]

        runner.process_item_group([{"prompt": "q", "answer": "a"}] * 3)

        assert runner._stats["processed"] == 2
        assert runner._stats["failed"] == 1

    def test_short_batch_counts_missing_failed(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=3)
        runner.client = MagicMock()
        runner.client.get_contents.return_value = ["only-one"]

        runner.process_item_group([{"prompt": "q", "answer": "a"}] * 3)

        assert runner._stats["processed"] == 1
        assert runner._stats["failed"] == 2

    def test_empty_responses_fail_whole_group(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=4)
        runner.client = MagicMock()
        runner.client.get_contents.return_value = []  # e.g. context length

        runner.process_item_group([{"prompt": "q", "answer": "a"}] * 4)

        assert runner._stats["failed"] == 4
        assert runner._stats["processed"] == 0
        assert not (tmp_path / "output.jsonl").exists()

    def test_singleton_delegates_to_process_item(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=1)
        runner.client = MagicMock()
        runner.client.get_content.return_value = "solo"

        runner.process_item_group([{"prompt": "q", "answer": "a"}])

        assert runner.client.get_content.called
        assert not runner.client.get_contents.called
        assert runner._stats["processed"] == 1


# ── InferenceRunner._process_concurrently grouping ────────────────


class TestConcurrentGrouping:
    def test_same_prompt_grouped_into_one_batch(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=3)
        runner.client = MagicMock()
        runner.client.get_contents.return_value = ["s1", "s2", "s3"]

        expanded = expand_data_with_resume(
            [{"prompt": "q", "answer": "a"}], {}, "prompt", 3
        )
        assert len(expanded) == 3
        runner._process_concurrently(expanded)

        runner.client.get_contents.assert_called_once()
        assert runner._stats["processed"] == 3

    def test_non_str_prompt_does_not_crash_run(self, tmp_path: Path) -> None:
        """Non-hashable/non-str prompts form singleton groups instead of raising."""
        runner = _make_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_content.return_value = "ok"

        items = [
            {"prompt": ["multimodal"], "answer": "a"},
            {"prompt": "q", "answer": "a"},
        ]
        runner._process_concurrently(items)

        total = sum(runner._stats.values())
        assert total == 2  # run survives; each item accounted for

    def test_failed_tasks_file_records_prompt(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, n_samples=2)
        runner.client = MagicMock()
        runner.client.get_contents.side_effect = RuntimeError("server down")

        runner._process_concurrently([{"prompt": "qq", "answer": "a"}] * 2)

        failed_file = tmp_path / "output_failed.jsonl"
        assert failed_file.exists()
        record = json.loads(failed_file.read_text().strip().split("\n")[0])
        assert record["prompt"] == "qq"
        assert record["samples"] == 2
        assert "server down" in record["error"]
