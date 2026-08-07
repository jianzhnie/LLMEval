"""Tests for llmeval.tasks.code_eval — execution sandbox and scoring.

These tests only exercise the **serial** scoring path and the pure utility
functions so that heavy dependencies (``pebble``, ``multiprocessing``) are
stubbed or avoided entirely.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import signal
import sys
import tempfile
import types
from contextlib import ExitStack
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
    _code_record_status,
    _failure_code_record,
    _is_code_infrastructure_failure,
    _process_code_item,
    estimate_pass_at_k,
    extract_code,
    score_code,
    score_code_result,
    write_cache,
)
from llmeval.tasks.postprocess import strip_reasoning_wrappers


def test_code_failure_record_contains_filter_trace() -> None:
    _, record = _process_code_item(
        (
            0,
            {"task_id": "t", "prompt": "def f():\n", "test": "", "gen": [""]},
            "test",
            "gen",
            1.0,
            False,
        )
    )

    assert record["raw_gen"] == ""
    assert record["filtered_gen"] == ""
    assert record["filter_trace"]["pipeline"] == "code_generation"


from llmeval.tasks.code_eval import execute as code_execute
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
        result = check_correctness(_add_program(), 3.0, "t1", allow_unsafe_code=True)
        assert result["passed"] is True
        assert result["result"] == "passed"

    def test_failing_program(self) -> None:
        program = "def add(a, b):\n    return a * b\n\nassert add(2, 3) == 5\n"
        result = check_correctness(program, 3.0, "t2", allow_unsafe_code=True)
        assert result["passed"] is False
        assert "AssertionError" in result["result"]

    def test_syntax_error(self) -> None:
        result = check_correctness(
            "def add(:\n    return", 3.0, "t3", allow_unsafe_code=True
        )
        assert result["passed"] is False
        assert "SyntaxError" in result["result"]

    def test_name_error(self) -> None:
        result = check_correctness(
            "assert foo(1) == 2\n", 3.0, "t4", allow_unsafe_code=True
        )
        assert result["passed"] is False
        assert "NameError" in result["result"]

    def test_timeout(self) -> None:
        program = "import time\ntime.sleep(5)\n"
        result = check_correctness(program, 1.0, "t5", allow_unsafe_code=True)
        assert result["passed"] is False
        assert result["result"] == "timed out"

    def test_long_timeout_still_fires(self) -> None:
        """A long timeout still works correctly for normal code."""
        result = check_correctness(_add_program(), 30.0, "t6", allow_unsafe_code=True)
        assert result["passed"] is True

    def test_with_check_wrapper(self) -> None:
        """HumanEval-style check(candidate) convention."""
        result = check_correctness(
            _check_wrapper_program(), 3.0, "t7", allow_unsafe_code=True
        )
        assert result["passed"] is True


# ═══════════════════════════════════════════════════════════════════════
# multiprocessing start-method resolution (P0-2: default fork)
# ═══════════════════════════════════════════════════════════════════════


class TestResolveMpMethod:
    def test_default_is_fork_when_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLMEVAL_MP_METHOD", raising=False)
        monkeypatch.setattr(
            code_execute.multiprocessing,
            "get_all_start_methods",
            lambda: ["fork", "spawn", "forkserver"],
        )
        assert code_execute._resolve_mp_method() == "fork"

    def test_spawn_fallback_when_fork_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Platforms without fork fall back to spawn, with a logged warning
        and a one-shot info line naming the effective method."""
        monkeypatch.delenv("LLMEVAL_MP_METHOD", raising=False)
        monkeypatch.setattr(
            code_execute.multiprocessing,
            "get_all_start_methods",
            lambda: ["spawn"],
        )
        # Fresh log-once cache so this test observes the emission itself.
        monkeypatch.setattr(code_execute, "_LOGGED_MP_METHODS", set())

        # init_logger sets propagate=False, so attach caplog's handler
        # directly to capture this logger's records.
        code_execute.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.DEBUG, logger="code_execute"):
                assert code_execute._resolve_mp_method() == "spawn"
        finally:
            code_execute.logger.removeHandler(caplog.handler)

        assert any(
            r.levelno == logging.WARNING and "falling back to spawn" in r.message
            for r in caplog.records
        )
        assert any(
            "code execution mp method: spawn" in r.message for r in caplog.records
        )

    def test_env_override_spawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMEVAL_MP_METHOD", "spawn")
        monkeypatch.setattr(
            code_execute.multiprocessing,
            "get_all_start_methods",
            lambda: ["fork", "spawn"],
        )
        assert code_execute._resolve_mp_method() == "spawn"

    def test_env_override_spawn_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit ``LLMEVAL_MP_METHOD=spawn`` still drives real execution."""
        monkeypatch.setenv("LLMEVAL_MP_METHOD", "spawn")
        result = check_correctness(
            _add_program(), 10.0, "t_spawn", allow_unsafe_code=True
        )
        assert result["passed"] is True

    def test_invalid_env_override_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMEVAL_MP_METHOD", "bogus")
        with pytest.raises(ValueError, match="invalid LLMEVAL_MP_METHOD"):
            code_execute._resolve_mp_method()


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
    def test_incorrect_program_is_not_infrastructure_failure(
        self, tmp_path: Path
    ) -> None:
        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": ["    return 2"],
                    "sample_index": 0,
                }
            ],
            "answer",
            "gen",
            tmp_path / "incorrect.jsonl",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.metrics["pass@1"] == 0.0
        assert result.failed_count == 0
        assert result.effective_sample_count == 1

    def test_execution_requires_explicit_opt_in(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError, match="executes generated code"):
            score_code(
                [
                    {
                        "prompt": "def f():\n",
                        "answer": "\nassert f() == 1\n",
                        "gen": ["    return 1"],
                    }
                ],
                "answer",
                "gen",
                tmp_path / "blocked.jsonl",
                max_workers=0,
            )

    def test_all_pass(self) -> None:
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
                "sample_index": 0,
            },
            {
                "task_id": "task/1",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a - b"],
                "sample_index": 0,
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code(
                items,
                "answer",
                "gen",
                cache_path,
                max_workers=0,
                exec_timeout=3.0,
                allow_unsafe_code=True,
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
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
                "sample_index": 0,
            },  # correct
            {
                "task_id": "task/1",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a * b"],
                "sample_index": 0,
            },  # wrong
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code(
                items,
                "answer",
                "gen",
                cache_path,
                max_workers=0,
                exec_timeout=3.0,
                allow_unsafe_code=True,
            )
            assert acc == 0.5
        finally:
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)

    def test_parallel_samples_complete_within_timeout(self, tmp_path: Path) -> None:
        """End-to-end timeout regression for the parallel path.

        Two samples are scored by real Pebble pool workers, each forking a
        per-sample child (the default start method since P0-2).  Both must
        complete well within the pool-level timeout — with the old ``spawn``
        default, per-sample interpreter restarts consumed that budget and
        produced spurious scoring timeouts.
        """
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
                "sample_index": 0,
            },
            {
                "task_id": "task/1",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a - b"],
                "sample_index": 0,
            },
        ]
        acc = score_code(
            items,
            "answer",
            "gen",
            tmp_path / "parallel.jsonl",
            max_workers=2,
            timeout=120,
            exec_timeout=5.0,
            allow_unsafe_code=True,
        )
        assert acc == 1.0

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
            {
                "task_id": "task/0",
                "prompt": "def foo():\n",
                "answer": "\nassert foo() == 1\n",
                "gen": [""],
                "sample_index": 0,
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            cache_path = Path(tf.name)
        try:
            acc = score_code(
                items,
                "answer",
                "gen",
                cache_path,
                max_workers=0,
                allow_unsafe_code=True,
            )
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
                "gen": ["    return a * b"],
                "sample_index": 0,
            },
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a + b"],
                "sample_index": 1,
            },
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
            allow_unsafe_code=True,
        )

        summary = json.loads(cache_path.with_suffix(".summary.json").read_text())
        records = [json.loads(line) for line in cache_path.read_text().splitlines()]
        assert acc == pytest.approx(0.5)
        assert summary["pass_at_k"]["pass@1"] == 0.5
        assert summary["pass_at_k"]["pass@2"] == 1.0
        assert summary["total"] == 2
        assert summary["problems"] == 1
        assert [record["sample_index"] for record in records] == [0, 1]

    def test_pass_at_1_reported_when_k_values_exclude_1(self, tmp_path: Path) -> None:
        """pass@1 is computed even when the caller's k_values omit 1."""
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a * b"],
                "sample_index": 0,
            },
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a + b"],
                "sample_index": 1,
            },
        ]
        cache_path = tmp_path / "code.jsonl"
        result = score_code_result(
            items,
            "answer",
            "gen",
            cache_path,
            max_workers=0,
            exec_timeout=3.0,
            k_values=(2,),
            allow_unsafe_code=True,
        )

        assert result.metrics["pass@1"] == pytest.approx(0.5)
        assert result.metrics["pass@2"] == pytest.approx(1.0)
        assert result.observations["pass@1"] == [pytest.approx(0.5)]
        summary = json.loads(cache_path.with_suffix(".summary.json").read_text())
        assert summary["pass_at_1"] == pytest.approx(0.5)

    def test_killed_by_signal_is_completed_observation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker killed by a signal (RLIMIT/OOM) is the model's fault."""
        # The reliability guard blocks os.kill inside candidate code, so
        # simulate the kill at the execution layer and use the fork start
        # method so the patched function reaches the worker process.
        # NOTE: forking this (multi-threaded, due to pytest internals)
        # process may emit a DeprecationWarning on Python ≥ 3.12 — that risk
        # is known and accepted here; production scoring workers are
        # single-threaded at fork time (see ``check_correctness`` docstring).
        monkeypatch.setenv("LLMEVAL_MP_METHOD", "fork")

        def _self_killing_execute(
            check_program: str, timeout: float, exec_globals: object = None
        ) -> tuple[str, str]:
            os.kill(os.getpid(), signal.SIGKILL)
            return ("passed", "")  # pragma: no cover - unreachable

        monkeypatch.setattr(
            "llmeval.tasks.code_eval.execute.unsafe_execute", _self_killing_execute
        )

        items = [
            {
                "task_id": "task/0",
                "prompt": "def f():\n",
                "answer": "\nassert f() == 1\n",
                "gen": ["    return 1"],
                "sample_index": 0,
            }
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            tmp_path / "killed.jsonl",
            max_workers=0,
            exec_timeout=3.0,
            allow_unsafe_code=True,
        )

        record = result.per_item[0]
        assert record["passed"] is False
        assert record["result"].startswith("failed: killed by signal")
        assert record["evaluation_status"] == "completed"
        assert result.failed_count == 0
        assert result.effective_sample_count == 1
        assert result.metrics["pass@1"] == 0.0

    def test_record_status_classification(self) -> None:
        """Signal kills are model failures; missing results are infra failures."""
        killed = {"result": "failed: killed by signal 9"}
        assert _code_record_status(killed) == "completed"
        assert not _is_code_infrastructure_failure(killed)

        missing = {"result": "failed: worker did not produce a result"}
        assert _code_record_status(missing) == "failed"
        assert _is_code_infrastructure_failure(missing)

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
# strip_reasoning_wrappers (code pipeline's reasoning-stripping filter)
# ═══════════════════════════════════════════════════════════════════════


class TestStripThinkTags:
    def test_answer_tag_preferred(self) -> None:
        text = "<think>plan</think>junk <answer>def f():\n    return 1</answer> tail"
        assert strip_reasoning_wrappers(text) == "def f():\n    return 1"

    def test_think_tag_fallback(self) -> None:
        assert strip_reasoning_wrappers("reasoning</think>code here") == "code here"

    def test_plain_text_unchanged(self) -> None:
        assert strip_reasoning_wrappers("def f(): pass") == "def f(): pass"


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
                "sample_index": 0,
            }
        ]
        acc = score_code(
            items,
            "answer",
            "gen",
            tmp_path / "cache.jsonl",
            max_workers=0,
            allow_unsafe_code=True,
        )
        assert acc == 1.0


class TestScoreCodePromptModes:
    def test_mbpp_natural_language_prompt_executes_code_only(
        self, tmp_path: Path
    ) -> None:
        items = [
            {
                "task_id": "1",
                "prompt": (
                    "You are an expert Python programmer.\n\n"
                    "Write a function add that returns the sum of two numbers.\n\n"
                    "[BEGIN]\n"
                ),
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["def add(a, b):\n    return a + b"],
                "sample_index": 0,
            }
        ]

        acc = score_code(
            items,
            "answer",
            "gen",
            tmp_path / "mbpp.jsonl",
            max_workers=0,
            allow_unsafe_code=True,
        )

        assert acc == 1.0

    def test_humaneval_full_function_definition_falls_back(
        self, tmp_path: Path
    ) -> None:
        items = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["def add(a, b):\n    return a + b"],
                "sample_index": 0,
            }
        ]

        acc = score_code(
            items,
            "answer",
            "gen",
            tmp_path / "humaneval.jsonl",
            max_workers=0,
            allow_unsafe_code=True,
        )

        assert acc == 1.0

    def test_answer_tag_wrapped_generation_passes(self, tmp_path: Path) -> None:
        items = [
            {
                "task_id": "t1",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["<answer>```python\n    return a + b\n```</answer>"],
                "sample_index": 0,
            }
        ]
        acc = score_code(
            items,
            "answer",
            "gen",
            tmp_path / "cache.jsonl",
            max_workers=0,
            allow_unsafe_code=True,
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


# ═══════════════════════════════════════════════════════════════════════
# Sample-index protocol (P0-1)
# ═══════════════════════════════════════════════════════════════════════


class TestCodeSampleIndexProtocol:
    """Explicit sample indices are preserved verbatim — never renumbered."""

    _PROMPT = "def add(a, b):\n"
    _TEST = "\nassert add(1, 2) == 3\n"
    _WRONG = "    return a * b"
    _RIGHT = "    return a + b"

    def _item(self, gen: list[str], **extra) -> dict:
        return {
            "task_id": "task/0",
            "prompt": self._PROMPT,
            "answer": self._TEST,
            "gen": gen,
            "sample_index": 0,
            **extra,
        }

    def _score(self, items: list[dict], tmp_path: Path):
        cache_path = tmp_path / "code.jsonl"
        acc = score_code(
            items,
            "answer",
            "gen",
            cache_path,
            max_workers=0,
            exec_timeout=3.0,
            k_values=(1, 2),
            allow_unsafe_code=True,
        )
        records = [json.loads(line) for line in cache_path.read_text().splitlines()]
        return acc, records

    def test_single_sample_rows_keep_explicit_indices(self, tmp_path: Path) -> None:
        """Rows with sample_index 0 and 2 keep the gap — index 1 stays missing."""
        items = [
            self._item([self._WRONG], sample_index=0),
            self._item([self._RIGHT], sample_index=2),
        ]
        acc, records = self._score(items, tmp_path)
        assert [record["sample_index"] for record in records] == [0, 2]
        assert acc == pytest.approx(0.5)

    def test_multi_generation_row_is_rejected(self, tmp_path: Path) -> None:
        items = [self._item([self._WRONG, self._RIGHT])]
        with pytest.raises(ValueError, match="one generation per row"):
            self._score(items, tmp_path)

    def test_multi_sample_row_requires_task_id(self, tmp_path: Path) -> None:
        item = self._item([self._WRONG, self._RIGHT])
        item.pop("task_id")

        with pytest.raises(ValueError, match=r"missing required.*task_id"):
            self._score([item], tmp_path)

    def test_invalid_scalar_sample_index_raises(self, tmp_path: Path) -> None:
        items = [self._item([self._RIGHT], sample_index=-1)]
        with pytest.raises(ValueError, match="sample_index"):
            self._score(items, tmp_path)

    def test_missing_sample_index_raises(self, tmp_path: Path) -> None:
        item = self._item([self._RIGHT])
        item.pop("sample_index")
        with pytest.raises(ValueError, match="sample_index"):
            self._score([item], tmp_path)

    def test_empty_generation_is_recorded(self, tmp_path: Path) -> None:
        items = [self._item([])]

        acc, records = self._score(items, tmp_path)

        assert acc == 0.0
        assert len(records) == 1
        assert records[0]["result"] == "failed: empty generation"

    def test_idempotent_duplicate_merges(self, tmp_path: Path) -> None:
        """Same index + same content is scored once, without error."""
        items = [
            self._item([self._RIGHT], sample_index=0),
            self._item([self._RIGHT], sample_index=0),
        ]
        acc, records = self._score(items, tmp_path)
        assert acc == 1.0
        assert len(records) == 1

    def test_conflicting_duplicate_raises(self, tmp_path: Path) -> None:
        """Same index + different content is a schema conflict."""
        items = [
            self._item([self._WRONG], sample_index=0),
            self._item([self._RIGHT], sample_index=0),
        ]
        with pytest.raises(ValueError, match="Conflicting duplicate"):
            self._score(items, tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# Fork-inherited resource usage (regression: pass@1 collapse)
#
# With the default ``fork`` start method the per-sample worker inherits the
# scoring worker's whole address space and fd table.  Resource limits must
# therefore be growth-relative to the inherited baseline; absolute budgets
# calibrated for a fresh interpreter kill correct solutions (MemoryError /
# EMFILE) and collapse pass@1 to 0 in production.
# ═══════════════════════════════════════════════════════════════════════


class TestForkInheritedResources:
    def test_many_inherited_fds_do_not_break_fork_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fork child born with >64 open fds must still execute and report.

        Before the fix, ``RLIMIT_NOFILE=64`` was already exceeded at birth in
        this situation, so the worker died with EMFILE and the item was
        dropped as "worker did not produce a result".
        """
        pytest.importorskip("resource")
        monkeypatch.setenv("LLMEVAL_MP_METHOD", "fork")
        # NOTE: forking this (multi-threaded, due to pytest internals)
        # process may emit a DeprecationWarning on Python ≥ 3.12 — accepted
        # here, as in ``test_killed_by_signal_is_completed_observation``.
        with ExitStack() as stack:
            for _ in range(80):
                stack.enter_context(tempfile.TemporaryFile())
            result = check_correctness(
                "x = 1 + 1\nassert x == 2\n", 3.0, "t-fd", allow_unsafe_code=True
            )
        assert result["passed"] is True, result

    def _recorded_limits(
        self, monkeypatch: pytest.MonkeyPatch, vsz: int | None, fd_count: int | None
    ) -> dict[int, tuple[int, int]]:
        resource = pytest.importorskip("resource")
        recorded: dict[int, tuple[int, int]] = {}
        monkeypatch.setattr(code_execute, "_current_vsz_bytes", lambda: vsz)
        monkeypatch.setattr(code_execute, "_current_fd_count", lambda: fd_count)
        monkeypatch.setattr(
            resource,
            "setrlimit",
            lambda name, limits: recorded.setdefault(name, limits),
        )
        code_execute._apply_resource_limits(3.0)
        return recorded

    def test_rlimit_as_is_headroom_above_inherited_vsz(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """fork children get the memory budget as growth headroom, not a cap
        already exceeded by the inherited address space."""
        resource = pytest.importorskip("resource")
        monkeypatch.setenv("LLMEVAL_MEMORY_LIMIT_MB", "2048")
        vsz = 3 * 1024**3
        recorded = self._recorded_limits(monkeypatch, vsz=vsz, fd_count=None)
        assert recorded[resource.RLIMIT_AS][0] == vsz + 2048 * 1024 * 1024

    def test_rlimit_as_absolute_when_vsz_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a readable baseline (non-Linux), keep the absolute budget."""
        resource = pytest.importorskip("resource")
        monkeypatch.setenv("LLMEVAL_MEMORY_LIMIT_MB", "2048")
        recorded = self._recorded_limits(monkeypatch, vsz=None, fd_count=None)
        assert recorded[resource.RLIMIT_AS][0] == 2048 * 1024 * 1024

    def test_rlimit_nofile_keeps_headroom_above_inherited_fds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resource = pytest.importorskip("resource")
        recorded = self._recorded_limits(monkeypatch, vsz=None, fd_count=100)
        assert recorded[resource.RLIMIT_NOFILE][0] == 116

    def test_rlimit_nofile_floor_without_inherited_fds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resource = pytest.importorskip("resource")
        recorded = self._recorded_limits(monkeypatch, vsz=None, fd_count=5)
        assert recorded[resource.RLIMIT_NOFILE][0] == 64


# ═══════════════════════════════════════════════════════════════════════
# Timeout classification (candidate timeout = wrong, worker hang = excluded)
# ═══════════════════════════════════════════════════════════════════════


class TestTimeoutClassification:
    def test_status_strings(self) -> None:
        assert _code_record_status({"result": "timed out"}) == "completed"
        assert _code_record_status({"result": "timed out: worker killed"}) == "timeout"

    def test_candidate_infinite_loop_counts_as_wrong(self, tmp_path: Path) -> None:
        """A candidate that dead-loops must stay in the Pass@k denominator."""
        items = [
            {
                "task_id": "task/0",
                "prompt": "def f():\n",
                "answer": "\nf()\n",
                "gen": ["    while True:\n        pass"],
                "sample_index": 0,
            }
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            tmp_path / "loop.jsonl",
            max_workers=0,
            exec_timeout=1.0,
            allow_unsafe_code=True,
        )
        record = result.per_item[0]
        assert record["result"] == "timed out"
        assert record["evaluation_status"] == "completed"
        assert result.metrics["pass@1"] == 0.0
        assert result.timeout_count == 0
        assert result.effective_sample_count == 1


# ═══════════════════════════════════════════════════════════════════════
# End-to-end regression: default parallel path must score correct solutions
# ═══════════════════════════════════════════════════════════════════════


class TestDefaultScoringPath:
    def test_correct_solutions_pass_on_default_parallel_path(
        self, tmp_path: Path
    ) -> None:
        """Default path (Pebble pool + per-sample fork): correct solutions
        must pass, wrong solutions must count as wrong — pass@1 > 0."""
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
                "sample_index": 0,
            },
            {
                "task_id": "task/1",
                "prompt": "def square(x):\n",
                "answer": "\nassert square(3) == 9\nassert square(4) == 16\n",
                "gen": ["    return x * x"],
                "sample_index": 0,
            },
            {
                "task_id": "task/2",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a + b"],  # wrong on purpose
                "sample_index": 0,
            },
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            tmp_path / "default_path.jsonl",
            max_workers=2,
            timeout=60,
            exec_timeout=5.0,
            allow_unsafe_code=True,
        )
        statuses = {r["task_id"]: r["evaluation_status"] for r in result.per_item}
        assert all(status == "completed" for status in statuses.values())
        assert result.metrics["pass@1"] == pytest.approx(2 / 3)
        assert result.failed_count == 0
        assert result.timeout_count == 0
