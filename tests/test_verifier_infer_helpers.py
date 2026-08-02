"""Tests for llmeval.inference.verifier helper functions.

These tests target the pure-utility functions (extract_tagged_answer, process_judgment,
etc.) that don't require a vLLM engine.  We mock the heavy imports so the module
can be loaded without vllm/openai installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock

import pytest

# ── Mock heavy dependencies before importing the module under test ──
_vllm_absent = importlib.util.find_spec("vllm") is None
if _vllm_absent:
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.outputs"] = types.ModuleType("vllm.outputs")

# Provide stubs that the module's top-level imports need
if _vllm_absent:
    sys.modules["vllm"].LLM = MagicMock  # type: ignore[attr-defined]
    sys.modules["vllm"].SamplingParams = MagicMock  # type: ignore[attr-defined]
    sys.modules["vllm"].RequestOutput = MagicMock  # type: ignore[attr-defined]
    sys.modules["vllm.outputs"].RequestOutput = MagicMock  # type: ignore[attr-defined]

# transformers may be available; if not, mock it.  Guarded with find_spec so
# the real module (with AutoTokenizer) is never shadowed in sys.modules.
if "transformers" not in sys.modules and not importlib.util.find_spec("transformers"):
    _tf = types.ModuleType("transformers")
    _tf.AutoTokenizer = MagicMock
    _tf.HfArgumentParser = MagicMock
    sys.modules["transformers"] = _tf

from llmeval.inference.verifier import (
    _last_n_strs,
    extract_tagged_answer,
    process_judgment,
    process_judgment_cursor,
)

# ── _last_n_strs ──────────────────────────────────────────────────


class TestLastNStrs:
    def test_basic(self) -> None:
        assert _last_n_strs("a b c d", 2) == "c d"

    def test_n_larger_than_tokens(self) -> None:
        assert _last_n_strs("hello", 10) == "hello"

    def test_empty_string(self) -> None:
        assert _last_n_strs("", 3) == ""


# ── extract_tagged_answer ──────────────────────────────────────────


class TestExtractAnswer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("<answer>42</answer>", "42"),
            ("reasoning <answer>\\frac{1}{2}</answer> extra", "\\frac{1}{2}"),
            ("multi\n<answer>\nline\n</answer>", "line"),
        ],
    )
    def test_answer_tag_extraction(self, text: str, expected: str) -> None:
        assert extract_tagged_answer(text) == expected

    def test_fallback_after_think_tag(self) -> None:
        text = "Some thinking\n</think >\n\nThe result is 7"
        assert extract_tagged_answer(text) == "The result is 7"

    def test_think_tag_case_insensitive(self) -> None:
        text = "Think...\n</THINK>\nvalue"
        assert extract_tagged_answer(text) == "value"

    def test_fallback_last_n_tokens(self) -> None:
        text = "alpha beta gamma"
        result = extract_tagged_answer(text, fallback_tokens=2)
        assert "beta gamma" in result

    def test_empty_string_returns_empty(self) -> None:
        assert extract_tagged_answer("") == ""

    def test_none_returns_empty(self) -> None:
        assert extract_tagged_answer(None) == ""

    def test_non_string_returns_empty(self) -> None:
        assert extract_tagged_answer(123) == ""

    def test_empty_answer_tag_falls_through(self) -> None:
        text = "<answer></answer>some content"
        result = extract_tagged_answer(text)
        assert result != ""

    def test_answer_tag_takes_priority_over_think(self) -> None:
        text = "</think />\nwrong <answer>correct</answer>"
        assert extract_tagged_answer(text) == "correct"


# ── process_judgment ──────────────────────────────────────────────


class TestProcessJudgment:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("\\boxed{A}", "A"),
            ("\\boxed{B}", "B"),
            ("\\boxed{C}", "C"),
            ("\\boxed{D}", "D"),
            ("A", "A"),
            ("B", "B"),
        ],
    )
    def test_boxed_extraction(self, input_str: str, expected: str) -> None:
        assert process_judgment(input_str) == expected

    def test_last_boxed_wins(self) -> None:
        assert process_judgment("\\boxed{A} and \\boxed{B}") == "B"

    def test_final_judgment_section(self) -> None:
        result = process_judgment("Final Judgment: (C)")
        assert result == "C"

    def test_fallback_any_letter(self) -> None:
        result = process_judgment("The answer is D")
        assert result == "D"

    def test_empty_returns_empty(self) -> None:
        assert process_judgment("") == ""

    def test_none_returns_empty(self) -> None:
        assert process_judgment(None) == ""

    def test_non_string_returns_empty(self) -> None:
        assert process_judgment(123) == ""


# ── process_judgment_cursor ───────────────────────────────────────


class TestProcessJudgmentCursor:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("\\boxed{A}", "A"),
            ("\\boxed{  c }", "C"),
            ("\\boxed{b}", "B"),
        ],
    )
    def test_boxed_extraction(self, input_str: str, expected: str) -> None:
        assert process_judgment_cursor(input_str) == expected

    def test_paren_fallback(self) -> None:
        assert process_judgment_cursor("Result: (D)") == "D"

    def test_standalone_letter_fallback(self) -> None:
        assert process_judgment_cursor("Grade = B") == "B"

    def test_does_not_match_letters_in_words(self) -> None:
        result = process_judgment_cursor("BAD word")
        assert result != "A"

    def test_case_insensitive(self) -> None:
        assert process_judgment_cursor("\\boxed{c}") == "C"

    def test_empty_returns_empty(self) -> None:
        assert process_judgment_cursor("") == ""

    def test_none_returns_empty(self) -> None:
        assert process_judgment_cursor(None) == ""

    def test_last_boxed_wins(self) -> None:
        assert process_judgment_cursor("\\boxed{A} \\boxed{C}") == "C"
