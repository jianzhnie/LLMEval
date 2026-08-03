"""Tests for llmeval.tasks.code_eval — execution sandbox and scoring.

These tests only exercise the **serial** scoring path and the pure utility
functions so that heavy dependencies (``pebble``, ``multiprocessing``) are
stubbed or avoided entirely.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Stub pebble only if it is genuinely absent ────────────────────────
# If pebble is installed (as it is in dev environments), leave the real
# module in place.  An unconditional global MagicMock on pebble.ProcessPool
# would pollute other test modules (e.g. test_mc_eval) running in the same
# process.
if "pebble" not in sys.modules and not importlib.util.find_spec("pebble"):
    sys.modules["pebble"] = types.ModuleType("pebble")
    sys.modules["pebble"].ProcessPool = MagicMock

from llmeval.tasks.code_eval.code_score import (
    CodeScoreResult,
    _failure_code_record,
    _strip_think_tags,
    estimate_pass_at_k,
    extract_code,
    score_code,
    write_cache,
)
from llmeval.tasks.code_eval.execute import (
    TimeoutException,
    check_correctness,
    reliability_guard,
    reliability_restore,
    swallow_io,
    time_limit,
    unsafe_execute,
)

# ═══════════════════════════════════════════════════════════════════════
# extract_code
# ═══════════════════════════════════════════════════════════════════════


class TestExtractCode:
    def test_fenced_python_block(self) -> None:
        text = "here is code:\n```python\ndef foo():\n    return 42\n```\nend"
        assert extract_code(text) == "def foo():\n    return 42"

    def test_fenced_no_lang_block(self) -> None:
        text = "```\nprint(1)\n```"
        assert extract_code(text) == "print(1)"

    def test_fenced_block_with_whitespace(self) -> None:
        text = "text\n```python\n\ndef foo():\n    pass\n\n```\nmore"
        assert extract_code(text) == "def foo():\n    pass"

    def test_function_fallback_no_fence(self) -> None:
        """No fenced block — should pick up from first ``def`` line."""
        text = "explanation text\ndef add(a, b):\n    return a + b"
        assert extract_code(text) == "def add(a, b):\n    return a + b"

    def test_from_import_with_separate_def(self) -> None:
        """Imports and helper functions should be preserved together."""
        text = "some text\nfrom typing import List\n\ndef foo(x: List):\n    return x"
        assert (
            extract_code(text)
            == "from typing import List\n\ndef foo(x: List):\n    return x"
        )

    def test_preserves_multiple_top_level_defs(self) -> None:
        """Helper classes/functions can be required by the candidate."""
        text = "def foo():\n    return 1\n\nclass Bar:\n    pass"
        assert extract_code(text) == "def foo():\n    return 1\n\nclass Bar:\n    pass"

    def test_trims_trailing_prose(self) -> None:
        text = "def foo():\n    return 1\n\nThis solves the problem."
        assert extract_code(text) == "def foo():\n    return 1"

    def test_empty_input(self) -> None:
        assert extract_code("") == ""
        assert extract_code(None) == ""

    def test_non_string(self) -> None:
        assert extract_code(123) == ""


# ═══════════════════════════════════════════════════════════════════════
# estimate_pass_at_k
# ═══════════════════════════════════════════════════════════════════════


class TestEstimatePassAtK:
    def test_k1_all_correct(self) -> None:
        assert estimate_pass_at_k(1, 1, 1) == 1.0

    def test_k1_all_wrong(self) -> None:
        assert estimate_pass_at_k(1, 0, 1) == 0.0

    def test_k1_half_correct(self) -> None:
        # 2 samples, 1 correct → pass@1 = 0.5
        result = estimate_pass_at_k(2, 1, 1)
        assert result == pytest.approx(0.5)

    def test_k1_all_correct_many_samples(self) -> None:
        assert estimate_pass_at_k(10, 10, 1) == 1.0

    def test_k2_all_correct(self) -> None:
        # n=c=2, k=2 → pass@2 = 1.0
        result = estimate_pass_at_k(2, 2, 2)
        assert result == pytest.approx(1.0)

    def test_k2_one_correct_two_samples(self) -> None:
        # n=2, c=1, k=2 → 1 − C(1,2)/C(2,2) = 1−0/1 = 1.0
        result = estimate_pass_at_k(2, 1, 2)
        assert result == pytest.approx(1.0)

    def test_k2_none_correct(self) -> None:
        result = estimate_pass_at_k(3, 0, 2)
        assert result == 0.0

    def test_k_equals_samples(self) -> None:
        # k=n → always 1.0 as long as c > 0
        assert estimate_pass_at_k(5, 1, 5) == 1.0
        assert estimate_pass_at_k(5, 0, 5) == 0.0

    def test_zero_samples(self) -> None:
        assert estimate_pass_at_k(0, 0, 1) == 0.0

    def test_negative_correct(self) -> None:
        # clamped to 0 internally
        assert estimate_pass_at_k(5, -1, 1) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# check_correctness (real execution — no mocking)
# ═══════════════════════════════════════════════════════════════════════


def _add_program() -> str:
    return "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n"


def _check_wrapper_program() -> str:
    return (
        "def candidate(a, b):\n    return a + b\n\n"
        "def check(fn):\n    assert fn(2, 3) == 5\n"
        "check(candidate)\n"
    )


class TestCheckCorrectness:
    def test_passing_program(self) -> None:
        result = check_correctness(_add_program(), 3.0, "t1")
        assert result["passed"] is True
        assert result["result"] == "passed"

    def test_failing_program(self) -> None:
        program = "def add(a, b):\n    return a * b\n\nassert add(2, 3) == 5\n"
        result = check_correctness(program, 3.0, "t2")
        assert result["passed"] is False
        assert "AssertionError" in result["result"]

    def test_syntax_error(self) -> None:
        result = check_correctness("def add(:\n    return", 3.0, "t3")
        assert result["passed"] is False
        assert "SyntaxError" in result["result"]

    def test_name_error(self) -> None:
        result = check_correctness("assert foo(1) == 2\n", 3.0, "t4")
        assert result["passed"] is False
        assert "NameError" in result["result"]

    def test_timeout(self) -> None:
        program = "import time\ntime.sleep(5)\n"
        result = check_correctness(program, 1.0, "t5")
        assert result["passed"] is False
        assert result["result"] == "timed out"

    def test_long_timeout_still_fires(self) -> None:
        """A long timeout still works correctly for normal code."""
        result = check_correctness(_add_program(), 30.0, "t6")
        assert result["passed"] is True

    def test_with_check_wrapper(self) -> None:
        """HumanEval-style check(candidate) convention."""
        result = check_correctness(_check_wrapper_program(), 3.0, "t7")
        assert result["passed"] is True


# ═══════════════════════════════════════════════════════════════════════
# unsafe_execute (direct, no multiprocess wrapper)
# ═══════════════════════════════════════════════════════════════════════


class TestUnsafeExecute:
    def test_passing(self) -> None:
        status, _ = unsafe_execute(_add_program(), 3.0)
        assert status == "passed"

    def test_failing(self) -> None:
        program = "assert 1 == 2\n"
        status, stderr = unsafe_execute(program, 3.0)
        assert status == "failed: AssertionError"
        assert stderr != ""


# ═══════════════════════════════════════════════════════════════════════
# score_code (serial path only)
# ═══════════════════════════════════════════════════════════════════════


class TestScoreCode:
    def test_all_pass(self) -> None:
        items = [
            {
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
            },
            {
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a - b"],
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code(
                items, "answer", "gen", cache_path, max_workers=0, exec_timeout=3.0
            )
            assert acc == 1.0

            # verify cache files
            assert cache_path.exists()
            summary_path = cache_path.with_suffix(".summary.json")
            assert summary_path.exists()
            summary = json.loads(summary_path.read_text())
            assert summary["pass_at_1"] == 1.0
            assert summary["correct"] == 2
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)

    def test_mixed_results(self) -> None:
        items = [
            {
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
            },  # correct
            {
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a * b"],
            },  # wrong
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code(
                items, "answer", "gen", cache_path, max_workers=0, exec_timeout=3.0
            )
            assert acc == 0.5
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)

    def test_empty_dataset(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code([], "answer", "gen", cache_path, max_workers=0)
            assert acc == 0.0
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)

    def test_empty_generation_marked_failed(self) -> None:
        items = [
            {"prompt": "def foo():\n", "answer": "\nassert foo() == 1\n", "gen": [""]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code(items, "answer", "gen", cache_path, max_workers=0)
            assert acc == 0.0
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)

    def test_multi_sample_pass_at_k_summary(self, tmp_path: Path) -> None:
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a * b", "    return a + b"],
            }
        ]
        cache_path = tmp_path / "code.jsonl"
        acc = score_code(
            items,
            "answer",
            "gen",
            cache_path,
            max_workers=0,
            exec_timeout=3.0,
            k_values=(1, 2),
        )

        summary = json.loads(cache_path.with_suffix(".summary.json").read_text())
        assert acc == pytest.approx(0.5)
        assert summary["pass_at_k"]["pass@1"] == 0.5
        assert summary["pass_at_k"]["pass@2"] == 1.0
        assert summary["total"] == 2
        assert summary["problems"] == 1

    def test_failure_record(self) -> None:
        """_failure_code_record marks every item as failed."""
        rec = _failure_code_record({"task_id": "HumanEval/99"})
        assert rec["passed"] is False
        assert rec["result"] == "scoring error"
        assert rec["task_id"] == "HumanEval/99"


# ═══════════════════════════════════════════════════════════════════════
# CodeScoreResult dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestCodeScoreResult:
    def test_defaults(self) -> None:
        csr = CodeScoreResult()
        assert csr.pass_at_1 == 0.0
        assert csr.total == 0
        assert csr.correct == 0
        assert csr.per_item == []

    def test_populated(self) -> None:
        csr = CodeScoreResult(pass_at_1=0.5, total=4, correct=2, per_item=[])
        assert csr.pass_at_1 == 0.5


# ═══════════════════════════════════════════════════════════════════════
# _strip_think_tags
# ═══════════════════════════════════════════════════════════════════════


class TestStripThinkTags:
    def test_answer_tag_preferred(self) -> None:
        text = "<think>plan</think>junk <answer>def f():\n    return 1</answer> tail"
        assert _strip_think_tags(text) == "def f():\n    return 1"

    def test_think_tag_fallback(self) -> None:
        assert _strip_think_tags("reasoning</think>code here") == "code here"

    def test_plain_text_unchanged(self) -> None:
        assert _strip_think_tags("def f(): pass") == "def f(): pass"


# ═══════════════════════════════════════════════════════════════════════
# reliability_guard / reliability_restore
# ═══════════════════════════════════════════════════════════════════════


class TestReliabilityGuard:
    def test_guard_disables_and_restore_reenables(self) -> None:
        import builtins
        import os

        original_system = os.system
        original_exit = builtins.exit
        try:
            reliability_guard()
            assert os.system is None
            assert builtins.exit is None
        finally:
            reliability_restore()
        assert os.system is original_system
        assert builtins.exit is original_exit

    def test_blocked_modules_raise_import_error(self) -> None:
        import sys

        try:
            reliability_guard()
            assert sys.modules["subprocess"] is None
            with pytest.raises(ImportError):
                import subprocess
        finally:
            reliability_restore()
        import subprocess  # real import works again

        assert subprocess is sys.modules["subprocess"]


# ═══════════════════════════════════════════════════════════════════════
# swallow_io / time_limit
# ═══════════════════════════════════════════════════════════════════════


class TestSwallowIO:
    def test_stdout_captured_and_restored(self, capsys: pytest.CaptureFixture) -> None:
        import sys

        with swallow_io():
            print("hidden output")
            assert sys.stdout.getvalue() == "hidden output\n"
        # After the block, stdout works normally again
        print("visible")
        assert "visible" in capsys.readouterr().out


class TestTimeLimit:
    def test_timeout_fires(self) -> None:
        import time

        with pytest.raises(TimeoutException), time_limit(0.2):
            time.sleep(2)

    def test_fast_code_not_interrupted(self) -> None:
        with time_limit(5.0):
            pass  # no exception


# ═══════════════════════════════════════════════════════════════════════
# score_code with reasoning-model wrappers
# ═══════════════════════════════════════════════════════════════════════


class TestScoreCodeThinkTags:
    def test_think_wrapped_generation_passes(self, tmp_path: Path) -> None:
        items = [
            {
                "task_id": "t0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": [
                    "<think>I should add them</think>\n```python\n    return a + b\n```"
                ],
            }
        ]
        acc = score_code(
            items, "answer", "gen", tmp_path / "cache.jsonl", max_workers=0
        )
        assert acc == 1.0

    def test_answer_tag_wrapped_generation_passes(self, tmp_path: Path) -> None:
        items = [
            {
                "task_id": "t1",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["<answer>```python\n    return a + b\n```</answer>"],
            }
        ]
        acc = score_code(
            items, "answer", "gen", tmp_path / "cache.jsonl", max_workers=0
        )
        assert acc == 1.0


# ═══════════════════════════════════════════════════════════════════════
# write_cache
# ═══════════════════════════════════════════════════════════════════════


class TestWriteCache:
    def test_writes_both_files(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            records = [
                {"task_id": "t1", "passed": True, "result": "passed"},
                {"task_id": "t2", "passed": False, "result": "failed: AssertionError"},
            ]
            csr = CodeScoreResult(pass_at_1=0.5, total=2, correct=1, per_item=records)
            write_cache(csr, cache_path)

            # JSONL
            lines = cache_path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0])["task_id"] == "t1"

            # Summary
            summary_path = cache_path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text())
            assert summary["pass_at_1"] == 0.5
            assert summary["total"] == 2
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)
