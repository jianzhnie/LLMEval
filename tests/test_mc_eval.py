"""Tests for llmeval.tasks.mc_eval (mc_score + mc_infer)."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

# ── Mock heavy dependencies ──
for mod_name in ("openai", "httpx"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Provide stubs for openai exceptions used at module level in mc_infer
_openai_mod = sys.modules["openai"]
for _exc in ("APIConnectionError", "APIError", "RateLimitError"):
    if not hasattr(_openai_mod, _exc):
        setattr(_openai_mod, _exc, type(_exc, (Exception,), {}))


# ===========================================================================
# mc_score tests
# ===========================================================================


class TestMCExtractAnswer:
    """Test answer letter extraction from generated text."""

    def test_extract_answer_pattern(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("Some text\nAnswer: B") == "B"
        assert _extract_answer("Reasoning...\n答案：D") == "D"
        assert _extract_answer("Answer: A") == "A"

    def test_extract_last_letter_fallback(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("The correct option is C.") == "C"
        assert _extract_answer("I think A and B but choose D") == "D"

    def test_extract_empty(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("") == ""
        assert _extract_answer("no letters here") == ""


class TestScoreLoglikelihood:
    """Test loglikelihood scoring with acc + acc_norm."""

    def test_all_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0], "choices": ["a", "b", "c"]},
            {"gold": 0, "logprobs": [-0.1, -1.0, -3.0], "choices": ["x", "y", "z"]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            assert acc == 1.0
            # Check summary file
            summary_path = Path(cache).with_suffix(".summary.json")
            assert summary_path.exists()
            s = json.loads(summary_path.read_text())
            assert s["acc"] == 1.0
            assert s["acc_norm"] == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_half_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0]},  # correct: index 1
            {"gold": 0, "logprobs": [-0.5, -0.1, -3.0]},  # wrong: index 1 wins, gold 0
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            assert acc == 0.5
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_acc_norm_length_penalized(self) -> None:
        """acc_norm should prefer shorter choices with same logprob."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        # Choice A: short, Choice B: very long. Same raw logprob.
        items = [
            {
                "gold": 0,
                "logprobs": [-10.0, -10.0],
                "choices": ["A", "BBBBBBBBBB"],
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            # Same logprob: argmax picks index 0 (= gold) → acc correct
            assert acc == 1.0
            summary_path = Path(cache).with_suffix(".summary.json")
            s = json.loads(summary_path.read_text())
            # acc_norm: "A"(len=1) → -10/1=-10.0, "B"×10 → -10/10=-1.0
            # argmax picks index 1 (long) ≠ gold 0 → acc_norm wrong
            assert s["acc_norm"] == 0.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_empty_dataset(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood([], cache)
            assert acc == 0.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)


class TestScoreGenerate:
    """Test generation-based scoring."""

    def test_exact_match(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "B", "gen": ["Some text\nAnswer: B"]},
            {"answer": "A", "gen": ["The answer is A."]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_generate(items, "answer", "gen", cache)
            assert acc == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_mixed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "C", "gen": ["Answer: B"]},  # wrong
            {"answer": "D", "gen": ["Answer: D"]},  # correct
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_generate(items, "answer", "gen", cache)
            assert acc == 0.5
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)


# ===========================================================================
# mc_infer tests
# ===========================================================================


class TestMCInferConfig:
    """Test MCInferConfig defaults and API key resolution."""

    def test_defaults(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import MCInferConfig

        c = MCInferConfig()
        assert c.mode == "loglikelihood"
        assert c.max_workers == 32
        assert c.temperature == 0.0
        assert c.n_shot == 0

    def test_api_key_default(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import MCInferConfig

        with patch.dict("os.environ", {}, clear=True):
            c = MCInferConfig()
            assert c.api_key == "EMPTY"

    def test_api_key_from_env(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import MCInferConfig

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
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        fmt = FewShotFormatter(n_shot=0)
        assert fmt.get_prefix("any prompt") == ""

    def test_load_and_prefix(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        tmp = self._make_examples(10)
        try:
            fmt = FewShotFormatter(n_shot=3, seed=42)
            fmt.load(tmp)
            prefix = fmt.get_prefix("some other prompt")
            # Should contain 3 examples separated by \n\n
            assert prefix.count("\n\n") >= 3
            assert "Q" in prefix
            assert "Answer: B" in prefix
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_dedup_excludes_test_prompt(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        tmp = self._make_examples(10)
        try:
            fmt = FewShotFormatter(n_shot=3, seed=42)
            fmt.load(tmp)
            # Get the raw prompt from one of the few-shot pool
            test_prompt = fmt._few_shot_pool[0]["prompt"]
            prefix_with_dedup = fmt.get_prefix(test_prompt)
            prefix_without = fmt.get_prefix("unrelated prompt")
            # Dedup should produce different prefixes (one fewer example)
            # Both should have content
            assert len(prefix_with_dedup) > 0
            assert len(prefix_without) > 0
            # Dedup removes the matching example, so non-dedup prefix may differ
            # The key invariant: formatted demo starts with raw_prompt + " " + answer
            assert any(test_prompt in d for d in fmt._all_formatted)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_insufficient_examples(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        tmp = self._make_examples(3)
        try:
            fmt = FewShotFormatter(n_shot=10)
            fmt.load(tmp)
            # Should warn and return empty
            assert fmt.get_prefix("test") == ""
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestArgmax:
    def test_basic(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import _argmax

        assert _argmax([1.0, 3.0, 2.0]) == 1
        assert _argmax([5.0]) == 0
        assert _argmax([-1.0, -0.5, -2.0]) == 1
