"""Tests for llmeval.tasks.code_eval.execute — the execution sandbox.

These tests only exercise the **serial** execution path and the pure utility
functions so that heavy dependencies (``pebble``, ``multiprocessing``) are
stubbed or avoided entirely.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
import types
from contextlib import ExitStack
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

    def test_os_exit_is_disabled(self) -> None:
        """``os._exit`` must be blocked so the worker still reports a result.

        Without the guard the candidate would kill the worker before it wrote
        its result file, turning a wrong answer into an infrastructure failure.
        """
        program = "import os\nos._exit(0)\n"
        result = check_correctness(program, 3.0, "t_exit", allow_unsafe_code=True)
        assert result["passed"] is False
        assert result["result"] == "failed: TypeError"  # os._exit is None
        assert result["result"] != "failed: worker did not produce a result"


# ═══════════════════════════════════════════════════════════════════════
# multiprocessing start-method resolution (P0-2: default fork)
# ═══════════════════════════════════════════════════════════════════════


class TestResolveMpMethod:
    def test_linux_default_is_fork_when_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLMEVAL_MP_METHOD", raising=False)
        monkeypatch.setattr(code_execute.sys, "platform", "linux")
        monkeypatch.setattr(
            code_execute.multiprocessing,
            "get_all_start_methods",
            lambda: ["fork", "spawn", "forkserver"],
        )
        assert code_execute._resolve_mp_method() == "fork"

    def test_macos_default_is_spawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLMEVAL_MP_METHOD", raising=False)
        monkeypatch.setattr(code_execute.sys, "platform", "darwin")
        monkeypatch.setattr(
            code_execute.multiprocessing,
            "get_all_start_methods",
            lambda: ["fork", "spawn"],
        )
        assert code_execute._resolve_mp_method() == "spawn"

    def test_spawn_fallback_when_fork_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Platforms without fork fall back to spawn, with a logged warning
        and a one-shot info line naming the effective method."""
        monkeypatch.delenv("LLMEVAL_MP_METHOD", raising=False)
        monkeypatch.setattr(code_execute.sys, "platform", "linux")
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

    def test_main_guard_executes(self) -> None:
        program = 'if __name__ == "__main__":\n    assert False\n'
        status, _ = unsafe_execute(program, 3.0)
        assert status == "failed: AssertionError"

    def test_caught_timeout_is_still_reported(self) -> None:
        program = (
            "import time\ntry:\n    time.sleep(1)\nexcept BaseException:\n    pass\n"
        )
        status, _ = unsafe_execute(program, 0.1)
        assert status == "timed out"

    @pytest.mark.parametrize(
        "program",
        [
            "import os\nos.execv('/bin/true', ['true'])\n",
            "import os\nos.posix_spawn('/bin/true', ['true'], {})\n",
            "import os\nos.popen('true')\n",
        ],
    )
    def test_process_replacement_apis_are_disabled(self, program: str) -> None:
        status, _ = unsafe_execute(program, 3.0)
        assert status == "failed: TypeError"


# ═══════════════════════════════════════════════════════════════════════
# reliability_guard / reliability_restore
# ═══════════════════════════════════════════════════════════════════════


class TestReliabilityGuard:
    def test_guard_disables_and_restore_reenables(self) -> None:
        import builtins

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
