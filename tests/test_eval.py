"""Tests for llmeval.evaluator helpers.

These tests target _process_item and evaluate_task logic without
requiring pebble / math-verify to be installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Mock heavy dependencies ──
# Only stub modules that are genuinely absent.  If pebble/math-verify are
# installed, leave the real modules in place so that other test modules
# running in the same process keep working (a global MagicMock on
# pebble.ProcessPool would break test_mc_eval's parallel path).
_pebble_absent = "pebble" not in sys.modules and not importlib.util.find_spec("pebble")
_math_verify_absent = "math_verify" not in sys.modules and not importlib.util.find_spec(
    "math_verify"
)

if _pebble_absent:
    sys.modules["pebble"] = types.ModuleType("pebble")
if _math_verify_absent:
    for mod_name in ("math_verify", "math_verify.metric", "math_verify.parser"):
        sys.modules[mod_name] = types.ModuleType(mod_name)

if _pebble_absent:
    sys.modules["pebble"].ProcessPool = MagicMock  # type: ignore[attr-defined]
if _math_verify_absent:
    sys.modules["math_verify"].metric = types.ModuleType("metric")
    sys.modules["math_verify"].parser = types.ModuleType("parser")
    sys.modules["math_verify.metric"].math_metric = MagicMock(return_value=MagicMock())
    sys.modules["math_verify.parser"].ExprExtractionConfig = MagicMock
    sys.modules["math_verify.parser"].LatexExtractionConfig = MagicMock

# Only stub transformers if it is genuinely absent — a partial stub (just
# HfArgumentParser) would otherwise pollute sys.modules for other test modules
# that need the real AutoTokenizer (e.g. test_verifier_infer_helpers).
if "transformers" not in sys.modules and not importlib.util.find_spec("transformers"):
    _tf = types.ModuleType("transformers")
    _tf.HfArgumentParser = MagicMock
    sys.modules["transformers"] = _tf

from llmeval.evaluator import _process_item, evaluate_task


class TestProcessItem:
    def test_copies_and_adds_task(self) -> None:
        item = {"prompt": "q", "answer": "a", "gen": ["response"]}
        result = _process_item(item, "math_opensource/aime24")
        assert result["task"] == "math_opensource/aime24"
        assert result["gen"] == ["response"]
        # Original not modified
        assert "task" not in item

    def test_custom_keys(self) -> None:
        item = {"input": "q", "label": "a", "output": ["r"]}
        result = _process_item(
            item, "math_opensource/math500", label_key="label", response_key="output"
        )
        assert result["task"] == "math_opensource/math500"

    def test_missing_label_key_raises(self) -> None:
        with pytest.raises(ValueError, match="label"):
            _process_item({"prompt": "q"}, "task")

    def test_missing_response_key_raises(self) -> None:
        with pytest.raises(ValueError, match="response"):
            _process_item({"prompt": "q", "answer": "a"}, "task")

    def test_non_dict_raises(self) -> None:
        with pytest.raises(TypeError, match="dictionary"):
            _process_item("not a dict", "task")

    def test_does_not_mutate_original(self) -> None:
        item = {"prompt": "q", "answer": "a", "gen": ["r"]}
        original_keys = set(item.keys())
        _process_item(item, "task")
        assert set(item.keys()) == original_keys
        assert "task" not in item


class TestEvaluateTask:
    def test_empty_dataset_returns_none(self) -> None:
        assert evaluate_task([], "task", "a", "g", "/tmp/cache", 4) is None

    def test_unsupported_task_returns_none(self, tmp_path: Path) -> None:
        data = [{"prompt": "q", "answer": "a", "gen": ["r"], "task": "t"}]
        result = evaluate_task(
            data, "unsupported/task", "answer", "gen", str(tmp_path / "cache"), 4
        )
        assert result is None
