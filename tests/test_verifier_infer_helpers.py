"""Tests for llmeval.inference.verifier helper functions.

These tests target the pure-utility functions (extract_tagged_answer, process_judgment,
etc.) that don't require a vLLM engine.  We mock the heavy imports so the module
can be loaded without vllm/openai installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
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

from llmeval.cache import ContentAddressedCache
from llmeval.inference.verifier import (
    VerifierOfflineInferenceRunner,
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


class TestVerifierResume:
    def _runner(self, tmp_path: Path) -> VerifierOfflineInferenceRunner:
        runner = VerifierOfflineInferenceRunner.__new__(VerifierOfflineInferenceRunner)
        args = MagicMock()
        args.input_key = "prompt"
        args.label_key = "answer"
        args.response_key = "gen"
        args.keep_origin_data = False
        args.output_file = str(tmp_path / "verifier.jsonl")
        args.input_file = str(tmp_path / "input.jsonl")
        args.n_samples = 1
        args.verifier_prompt_type = "fdd_prompt_cursor"
        args.model_name_or_path = "verifier-model"
        args.model_revision = "revision-1"
        args.task = "math_opensource/test"
        args.max_tokens = 64
        args.temperature = 0.0
        args.top_p = 1.0
        args.top_k = -1
        args.repetition_penalty = 1.0
        args.seed = 7
        runner.args = args
        runner._file_lock = threading.Lock()
        runner.cache = None
        runner._git_hash = "test-git"
        runner.verifier_prompt = "{question}\n{gold_answer}\n{llm_response}"
        runner.llm = None
        runner.tokenizer = None
        runner.sampling_params = None
        return runner

    def test_sampling_params_are_independent_and_honor_decoding_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.inference.verifier as verifier_mod

        constructed: list[dict[str, object]] = []

        def _sampling_params(**kwargs: object) -> SimpleNamespace:
            constructed.append(kwargs)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(verifier_mod, "SamplingParams", _sampling_params)
        runner = self._runner(tmp_path)
        runner.args.do_sample = False
        runner.args.skip_special_tokens = False
        items = [
            {"doc_id": "doc:1", "prompt": "q", "sample_index": 0},
            {"doc_id": "doc:1", "prompt": "q", "sample_index": 1},
        ]

        runner._sampling_params_for_items(items)

        first, second = constructed
        assert first["seed"] != second["seed"]
        assert first["temperature"] == second["temperature"] == 0.0
        assert first["skip_special_tokens"] is False

    def test_resume_id_survives_compacted_output(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        item = {"prompt": "q", "answer": "4", "gen": ["4"]}
        result = runner._prepare_result_item(item, "\\boxed{A}")

        assert result["prompt"] == ""
        assert result["gen"] == ""
        assert result["llmeval_verifier_id"]

        Path(runner.args.output_file).write_text(
            json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = runner.count_completed_samples()

        assert counts[result["llmeval_verifier_id"]] == 1

    def test_load_data_skips_completed_resume_id(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        item = {"prompt": "q", "answer": "4", "gen": ["4"]}
        Path(runner.args.input_file).write_text(
            json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        result = runner._prepare_result_item(item, "\\boxed{A}")
        Path(runner.args.output_file).write_text(
            json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        assert runner.load_data() == []

    def test_response_with_unparsed_judgment_is_still_complete(
        self, tmp_path: Path
    ) -> None:
        runner = self._runner(tmp_path)
        item = {"prompt": "q", "answer": "4", "gen": ["4"]}
        Path(runner.args.input_file).write_text(json.dumps(item) + "\n")
        result = runner._prepare_result_item(item, "unclassifiable response")
        result["Verifier_judgment"] = ""
        Path(runner.args.output_file).write_text(json.dumps(result) + "\n")

        assert runner.count_completed_samples()[result["llmeval_verifier_id"]] == 1
        assert runner.load_data() == []

    def test_resume_preserves_missing_sample_index(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner.args.n_samples = 3
        item = {"doc_id": "doc:1", "prompt": "q", "answer": "4", "gen": ["4"]}
        Path(runner.args.input_file).write_text(json.dumps(item) + "\n")
        rows = []
        for index in (0, 2):
            row = runner._prepare_result_item(item, f"response-{index}")
            row["sample_index"] = index
            rows.append(row)
        Path(runner.args.output_file).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

        remaining = runner.load_data()

        assert len(remaining) == 1
        assert remaining[0]["sample_index"] == 1

    def test_verifier_content_cache_reuses_raw_response(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner.cache = ContentAddressedCache(tmp_path / "content", "inference")
        runner.sampling_params = object()
        runner.tokenizer = MagicMock()
        runner.tokenizer.apply_chat_template.return_value = "rendered verifier prompt"
        output = MagicMock()
        output.outputs = [MagicMock(text="\\boxed{A}")]
        runner.llm = MagicMock()
        runner.llm.generate.return_value = [output]
        item = {
            "doc_id": "math:1",
            "prompt": "2+2",
            "answer": "4",
            "gen": ["4"],
        }

        first_key = runner._cache_key(item, "rendered verifier prompt")
        runner.process_and_write_batch([item])
        runner.process_and_write_batch([item])

        assert runner.llm.generate.call_count == 1
        assert runner.cache.stats().to_dict() == {
            "hits": 1,
            "misses": 1,
            "corrupt": 0,
            "writes": 1,
        }
        rows = Path(runner.args.output_file).read_text().strip().splitlines()
        assert len(rows) == 2
        assert json.loads(rows[-1])["Verifier_judgment"] == "A"

        runner.args.temperature = 0.5
        assert runner._cache_key(item, "rendered verifier prompt") != first_key

    def test_verifier_empty_response_is_not_cached(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner.cache = ContentAddressedCache(tmp_path / "content", "inference")
        runner.sampling_params = object()
        runner.tokenizer = MagicMock()
        runner.tokenizer.apply_chat_template.return_value = "rendered verifier prompt"
        output = MagicMock()
        output.outputs = [MagicMock(text="")]
        runner.llm = MagicMock()
        runner.llm.generate.return_value = [output]
        item = {"doc_id": "math:1", "prompt": "2+2", "answer": "4", "gen": ["4"]}

        runner.process_and_write_batch([item])
        runner.process_and_write_batch([item])

        assert runner.llm.generate.call_count == 2
        assert runner.cache.stats().writes == 0
