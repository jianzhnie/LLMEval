"""Tests for llmeval.inference.mc.

Focuses on the inference client and runner behavior without a live API.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import threading
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

from llmeval.inference.mc import MCLoglikelihoodClient, MCRunner
from llmeval.utils.config import MCInferConfig


def _make_ll_client(max_retries: int = 0) -> MCLoglikelihoodClient:
    client = MCLoglikelihoodClient.__new__(MCLoglikelihoodClient)
    client.model_name = "m"
    client.timeout = 5
    client.max_retries = max_retries
    client.client = MagicMock()
    return client


def _fake_top_probs_resp(top_probs: dict[str, float]) -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.logprobs.top_logprobs = [top_probs]
    resp.choices = [choice]
    return resp


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
    runner._stats = {"processed": 0, "failed": 0, "correct": 0, "skipped": 0}
    return runner


def _write_input(path: Path) -> None:
    items = [
        {"prompt": "Q1?\nA. x\nB. y\nAnswer:", "choices": ["x", "y"], "gold": 1},
        {"prompt": "Q2?\nA. p\nB. q\nAnswer:", "choices": ["p", "q"], "gold": 0},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestMCLoglikelihoodClient:
    def test_single_request_with_top_logprobs(self) -> None:
        client = _make_ll_client()
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

    def test_failure_returns_all_neg_inf(self) -> None:
        client = _make_ll_client(max_retries=0)
        client.client.completions.create.side_effect = RuntimeError("down")

        assert client.get_choices_logprobs("prompt", ["A", "B"]) == [
            float("-inf"),
            float("-inf"),
        ]


class TestProcessLoglikelihoodItem:
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
            {"prompt": "q", "choices": ["a", "b"], "gold": 1}
        )

        assert result["pred"] == 1
        assert result["correct"] is True
        assert result["choice_tokens"] == ["A", "B"]


class TestProcessGenerateItem:
    def test_success(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "ans"

        result = runner.process_generate_item(
            {"prompt": "q", "answer": "A"}, client, []
        )

        assert result["gen"] == ["ans"]

    def test_null_content_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = None

        with pytest.raises(RuntimeError, match="no usable text"):
            runner.process_generate_item({"prompt": "q", "answer": "A"}, client, [])


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
