"""Tests for the evaluator entry points.

These tests exercise registry-backed evaluation without requiring a live model
service.
"""

from __future__ import annotations

import importlib.machinery
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
        module = types.ModuleType(mod_name)
        module.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
        sys.modules[mod_name] = module

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

from llmeval.evaluator import (
    evaluate_task,
)
from llmeval.tasks.results import ScorerResult


def _scorer_result(name: str, value: float) -> ScorerResult:
    return ScorerResult(
        metrics={name: value},
        observations={name: [value]},
        sample_count=1,
        effective_sample_count=1,
    )


class TestEvaluateTask:
    def test_empty_dataset_returns_zero_and_validates_task(self, tmp_path: Path) -> None:
        assert evaluate_task(
            [], "mc_opensource/task", "a", "g", tmp_path / "empty.jsonl", 4
        ) == 0.0
        assert (tmp_path / "empty.summary.json").exists()

    def test_unsupported_task_returns_none(self, tmp_path: Path) -> None:
        data = [{"prompt": "q", "answer": "a", "gen": ["r"], "task": "t"}]
        result = evaluate_task(
            data, "unsupported/task", "answer", "gen", str(tmp_path / "cache"), 4
        )
        assert result is None

    def test_mc_loglikelihood_dispatch(self, tmp_path: Path) -> None:
        """Items with 'logprobs' route to score_loglikelihood."""
        data = [
            {
                "gold": 1,
                "logprobs": [-1.0, -0.5, -2.0],
                "choices": ["a", "b", "c"],
                "task": "mc_opensource/mmlu",
            }
        ]
        acc = evaluate_task(
            data, "mc_opensource/mmlu", "answer", "gen", tmp_path / "mc.jsonl", 1
        )
        assert acc == 1.0
        assert (tmp_path / "mc.summary.json").exists()

    def test_mc_generate_dispatch(self, tmp_path: Path) -> None:
        """Items without 'logprobs' route to score_generate."""
        data = [{"answer": "B", "gen": ["Answer: B"], "task": "mc_opensource/mmlu"}]
        acc = evaluate_task(
            data, "mc_opensource/mmlu", "answer", "gen", tmp_path / "mc.jsonl", 1
        )
        assert acc == 1.0

    def test_mc_mixed_schema_returns_none(self, tmp_path: Path) -> None:
        data = [
            {"gold": 1, "logprobs": [-1.0, -0.5], "task": "mc_opensource/mmlu"},
            {"answer": "B", "gen": ["Answer: B"], "task": "mc_opensource/mmlu"},
        ]
        acc = evaluate_task(
            data, "mc_opensource/mmlu", "answer", "gen", tmp_path / "mc.jsonl", 1
        )
        assert acc is None

    def test_mc_error_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scorer exception is caught and reported as None."""
        import llmeval.evaluator as ev

        def _boom(**_: object) -> float:
            raise RuntimeError("scorer exploded")

        monkeypatch.setattr(ev, "score_generate_result", _boom)
        data = [{"answer": "B", "gen": ["Answer: B"], "task": "mc_opensource/mmlu"}]
        acc = evaluate_task(
            data, "mc_opensource/mmlu", "answer", "gen", tmp_path / "mc.jsonl", 1
        )
        assert acc is None

    def test_code_dispatch(self, tmp_path: Path) -> None:
        data = [
            {
                "task_id": "t0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a + b"],
                "task": "code_opensource/humaneval",
            }
        ]
        acc = evaluate_task(
            data,
            "code_opensource/humaneval",
            "answer",
            "gen",
            tmp_path / "code.jsonl",
            1,  # max_workers=1 → serial path
            allow_unsafe_code=True,
        )
        assert acc == 1.0

    def test_code_error_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.evaluator as ev

        def _boom(**_: object) -> float:
            raise RuntimeError("scorer exploded")

        monkeypatch.setattr(ev, "score_code_result", _boom)
        data = [
            {
                "prompt": "def f():\n",
                "answer": "\nassert f() == 1\n",
                "gen": ["    return 1"],
                "task": "code_opensource/humaneval",
            }
        ]
        acc = evaluate_task(
            data, "code_opensource/humaneval", "answer", "gen", tmp_path / "c.jsonl", 1
        )
        assert acc is None

    def test_math_dispatch(self, tmp_path: Path) -> None:
        import llmeval.evaluator as ev

        called: dict[str, object] = {}

        def _fake_compute_scores(**kwargs: object) -> ScorerResult:
            called.update(kwargs)
            return _scorer_result("accuracy", 1.0)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ev, "compute_score_result", _fake_compute_scores)
        try:
            data = [
                {
                    "answer": "5",
                    "gen": ["The answer is $\\boxed{5}$"],
                    "task": "math_opensource/aime24",
                }
            ]
            acc = evaluate_task(
                data,
                "math_opensource/aime24",
                "answer",
                "gen",
                tmp_path / "m.jsonl",
                2,
            )
        finally:
            monkeypatch.undo()

        assert acc == 1.0
        assert called["eval_dataset"] == data
        assert called["label_key"] == "answer"
        assert called["response_key"] == "gen"

    def test_math_error_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.evaluator as ev

        def _boom(**_: object) -> float:
            raise RuntimeError("scorer exploded")

        monkeypatch.setattr(ev, "compute_score_result", _boom)
        data = [{"answer": "5", "gen": ["5"], "task": "math_opensource/aime24"}]
        acc = evaluate_task(
            data, "math_opensource/aime24", "answer", "gen", tmp_path / "m.jsonl", 1
        )
        assert acc is None
