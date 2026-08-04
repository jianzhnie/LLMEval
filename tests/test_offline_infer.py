"""Tests for llmeval.inference.offline without requiring vLLM installed."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_vllm_absent = importlib.util.find_spec("vllm") is None
if _vllm_absent:
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.outputs"] = types.ModuleType("vllm.outputs")
    sys.modules["vllm"].__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)
    sys.modules["vllm.outputs"].__spec__ = importlib.machinery.ModuleSpec(
        "vllm.outputs", loader=None
    )
    sys.modules["vllm"].LLM = MagicMock  # type: ignore[attr-defined]
    sys.modules["vllm"].SamplingParams = MagicMock  # type: ignore[attr-defined]
    sys.modules["vllm.outputs"].RequestOutput = MagicMock  # type: ignore[attr-defined]

if "transformers" not in sys.modules and not importlib.util.find_spec("transformers"):
    _tf = types.ModuleType("transformers")
    _tf.HfArgumentParser = MagicMock
    sys.modules["transformers"] = _tf

from llmeval.inference.offline import OfflineInferenceRunner


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "input_file": str(tmp_path / "input.jsonl"),
        "output_file": str(tmp_path / "output.jsonl"),
        "cache_dir": str(tmp_path / "cache"),
        "input_key": "prompt",
        "label_key": "answer",
        "response_key": "gen",
        "system_prompt_type": "empty",
        "n_samples": 1,
        "batch_size": 2,
        "model_name_or_path": "test-model",
        "max_model_len": 4096,
        "rope_scaling": "{}",
        "rope_scaling_dict": None,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 2048,
        "seed": 123,
        "trust_remote_code": False,
        "dtype": "float16",
        "device": "cuda",
        "quantization": "awq",
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runner(tmp_path: Path, **overrides: object) -> OfflineInferenceRunner:
    runner = OfflineInferenceRunner.__new__(OfflineInferenceRunner)
    runner.args = _args(tmp_path, **overrides)
    runner._file_lock = threading.Lock()
    runner.llm = None
    runner.sampling_params = None
    runner.system_prompt = None
    return runner


class TestOfflineInferenceRunner:
    def test_convert_to_messages_format(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)

        messages = runner.convert_to_messages_format({"prompt": "2+2?", "answer": "4"})

        assert messages == [{"role": "user", "content": "2+2?"}]

    def test_convert_to_messages_format_falls_back_to_prompt(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path, input_key="question")

        messages = runner.convert_to_messages_format({"prompt": "2+2?", "answer": "4"})

        assert messages == [{"role": "user", "content": "2+2?"}]

    def test_convert_rejects_applied_chat_template(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)

        with pytest.raises(ValueError, match="chat_template"):
            runner.convert_to_messages_format({"prompt": "<|im_start|>user\nq"})

    def test_load_data_applies_resume(self, tmp_path: Path) -> None:
        args = _args(tmp_path, n_samples=2)
        Path(args.input_file).write_text(
            json.dumps({"prompt": "q", "answer": "a"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        Path(args.output_file).write_text(
            json.dumps({"prompt": "q", "gen": ["one"]}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        runner = _runner(tmp_path, n_samples=2)

        remaining = runner.load_data()

        assert len(remaining) == 1
        assert remaining[0]["prompt"] == "q"

    def test_write_batch_results_appends_generation(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        output = MagicMock()
        output.outputs = [MagicMock(text="answer text")]

        runner._write_batch_results([{"prompt": "q", "answer": "a"}], [output])

        rows = [
            json.loads(line)
            for line in Path(runner.args.output_file).read_text().splitlines()
        ]
        assert rows == [{"prompt": "q", "answer": "a", "gen": ["answer text"]}]

    def test_setup_vllm_engine_passes_configured_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.inference.offline as offline_mod

        fake_llm = MagicMock(return_value="llm")
        fake_sampling = MagicMock(return_value="sampling")
        monkeypatch.setattr(offline_mod, "LLM", fake_llm)
        monkeypatch.setattr(offline_mod, "SamplingParams", fake_sampling)
        runner = _runner(tmp_path)

        llm, sampling = runner.setup_vllm_engine()

        assert llm == "llm"
        assert sampling == "sampling"
        llm_kwargs = fake_llm.call_args.kwargs
        assert llm_kwargs["model"] == "test-model"
        assert llm_kwargs["device"] == "cuda"
        assert llm_kwargs["quantization"] == "awq"
        assert llm_kwargs["max_num_batched_tokens"] == 2048
        sampling_kwargs = fake_sampling.call_args.kwargs
        assert sampling_kwargs["max_tokens"] == 128
        assert sampling_kwargs["temperature"] == 0.2
