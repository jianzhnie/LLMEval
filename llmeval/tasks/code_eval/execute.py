"""Safe subprocess code execution for code-generation evaluation.

This module provides a self-contained code execution sandbox inspired by the
HuggingFace ``evaluate`` library's ``code_eval`` metric.  Generated code is
run inside a separate process with a timeout, disabled dangerous builtins, and
IO redirection so that a single hanging or misbehaving sample cannot corrupt
the evaluation run.

.. warning::
    This is a **safety guard**, not a security sandbox.  It prevents accidental
    interference (infinite loops, filesystem noise) but does *not* protect
    against intentionally malicious code.  Only run on trusted model outputs.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import traceback
from contextlib import contextmanager, suppress
from typing import Any

# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TimeoutException(Exception):
    """Raised when code execution exceeds the allotted time."""


@contextmanager
def time_limit(seconds: float) -> Any:
    """Context manager that raises :class:`TimeoutException` after *seconds*.

    Uses ``signal.setitimer(ITIMER_REAL, ...)`` so the timeout fires even
    when the main thread is blocked in a C extension call.
    """

    def _handler(_signum: int, _frame: Any) -> None:
        raise TimeoutException("Code execution timed out")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# ---------------------------------------------------------------------------
# IO swallowing
# ---------------------------------------------------------------------------


class WriteOnlyStringIO(io.StringIO):
    """A StringIO that rejects reads — prevents generated code from peeking
    at captured output."""

    def read(self, *args: Any, **kwargs: Any) -> str:
        raise OSError("reading from stdout/stderr is not allowed")

    def readline(self, *args: Any, **kwargs: Any) -> str:
        raise OSError("reading from stdout/stderr is not allowed")

    def readlines(self, *args: Any, **kwargs: Any) -> list[str]:
        raise OSError("reading from stdout/stderr is not allowed")


@contextmanager
def swallow_io() -> Any:
    """Redirect stdout, stderr, and stdin so generated code cannot interfere.

    Yields nothing — after the block, callers can inspect
    ``sys.stdout.getvalue()`` etc. directly.
    """
    _out, _err, _in = sys.stdout, sys.stderr, sys.stdin
    sys.stdout = WriteOnlyStringIO()
    sys.stderr = WriteOnlyStringIO()
    sys.stdin = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = _out
        sys.stderr = _err
        sys.stdin = _in


# ---------------------------------------------------------------------------
# Reliability guard
# ---------------------------------------------------------------------------

_ORIGINAL_BUILTINS: dict[str, Any] = {}
_ORIGINAL_OS_FUNCS: dict[str, Any] = {}
_ORIGINAL_SHUTIL_FUNCS: dict[str, Any] = {}
_ORIGINAL_MODULES: dict[str, Any] = {}


def _get_builtins_module() -> Any:
    """Return the actual ``builtins`` module regardless of context.

    In the main interpreter ``__builtins__`` is the ``builtins`` module.
    Inside a ``multiprocessing.Process`` (or ``exec``) it may be the
    module's ``__dict__`` instead.
    """
    import builtins as _bi

    return _bi


def reliability_guard() -> None:
    """Disable dangerous builtins and standard-library functions.

    Generated code that calls ``exit()``, ``os.system()``, ``subprocess.Popen``,
    ``shutil.rmtree``, etc. will get an ``AttributeError`` instead of causing
    real side-effects.

    Modules are set to ``None`` in ``sys.modules`` (rather than popped) so
    that ``import subprocess`` raises ``ImportError`` instead of silently
    reloading the real module.
    """
    _bi = _get_builtins_module()

    # --- builtins ---
    for _name in ("exit", "quit"):
        _ORIGINAL_BUILTINS[_name] = getattr(_bi, _name, None)
        setattr(_bi, _name, None)

    # --- os ---
    _forbidden_os = (
        "kill",
        "system",
        "remove",
        "rmdir",
        "fork",
        "rename",
        "chmod",
        "chown",
        "getpid",
        "getppid",
        "listdir",
        "killpg",
        "unlink",
        "symlink",
        "link",
        "forkpty",
        "fchmod",
        "fchown",
        "chflags",
        "lchflags",
        "lchmod",
        "lchown",
    )
    for _name in _forbidden_os:
        if hasattr(os, _name):
            _ORIGINAL_OS_FUNCS[_name] = getattr(os, _name)
            setattr(os, _name, None)

    # --- shutil ---
    for _name in ("rmtree", "move", "copy", "copy2"):
        if hasattr(shutil, _name):
            _ORIGINAL_SHUTIL_FUNCS[_name] = getattr(shutil, _name)
            setattr(shutil, _name, None)

    # --- dangerous modules — block re-import by setting sys.modules[key] = None ---
    for _mod in (
        "subprocess",
        "faulthandler",
        "ipdb",
        "joblib",
        "resource",
        "psutil",
        "tkinter",
        "ctypes",
        "multiprocessing",
    ):
        _ORIGINAL_MODULES[_mod] = sys.modules.get(_mod)
        sys.modules[_mod] = None


def reliability_restore() -> None:
    """Reverse the effects of :func:`reliability_guard`."""
    _bi = _get_builtins_module()
    for _name, _val in _ORIGINAL_BUILTINS.items():
        setattr(_bi, _name, _val)
    for _name, _val in _ORIGINAL_OS_FUNCS.items():
        setattr(os, _name, _val)
    for _name, _val in _ORIGINAL_SHUTIL_FUNCS.items():
        setattr(shutil, _name, _val)
    for _name, _val in _ORIGINAL_MODULES.items():
        if _val is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _val


# ---------------------------------------------------------------------------
# Temp directory
# ---------------------------------------------------------------------------


@contextmanager
def create_tempdir() -> Any:
    """Create and chdir into a temporary directory, cleaning up on exit."""
    _orig_cwd = os.getcwd()
    _td = tempfile.TemporaryDirectory()
    try:
        os.chdir(_td.name)
        yield _td.name
    finally:
        os.chdir(_orig_cwd)
        _td.cleanup()


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


def unsafe_execute(
    check_program: str,
    timeout: float,
    exec_globals: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Execute *check_program* with safety guards and a timeout.

    Parameters
    ----------
    check_program:
        Python source to execute (candidate + test harness concatenated).
    timeout:
        Maximum seconds before raising :exc:`TimeoutException`.
    exec_globals:
        Optional globals dict for ``exec``.  If ``None`` a fresh ``{}`` is used.

    Returns
    -------
    (status, stderr_str)
        ``status`` is one of ``"passed"``, ``"timed out"``, or
        ``"failed: <exception-name>"``.
    """
    if exec_globals is None:
        exec_globals = {}

    with create_tempdir():
        # Preserve cleanup helpers that reliability_guard would break.
        _saved_rmtree = shutil.rmtree
        _saved_os_rmdir = os.rmdir
        _saved_os_chdir = os.chdir

        reliability_guard()
        try:
            with swallow_io(), time_limit(timeout):
                exec(check_program, exec_globals)
            return ("passed", "")
        except TimeoutException:
            return ("timed out", "")
        except BaseException as exc:
            _stack = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            return (f"failed: {type(exc).__name__}", _stack)
        finally:
            reliability_restore()
            shutil.rmtree = _saved_rmtree
            os.rmdir = _saved_os_rmdir
            os.chdir = _saved_os_chdir


# ---------------------------------------------------------------------------
# Multiprocess wrapper
# ---------------------------------------------------------------------------


def _worker(
    check_program: str,
    timeout: float,
    task_id: str,
    result_file: str,
) -> None:
    """Entry-point for the child process — runs *check_program* and writes
    the result dict to *result_file* as JSON."""
    status, stderr = unsafe_execute(check_program, timeout)
    result: dict[str, Any] = {
        "task_id": task_id,
        "passed": status == "passed",
        "result": status,
        "stderr": stderr,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f)


def _cleanup_tmp(path: str) -> None:
    """Remove temporary file, ignoring errors if it does not exist."""
    with suppress(OSError):
        os.unlink(path)


def check_correctness(
    check_program: str,
    timeout: float,
    task_id: str = "",
) -> dict[str, Any]:
    """Execute *check_program* in a child process with a double safety net.

    The inner guard is :func:`unsafe_execute` with a signal-based timeout.
    The outer guard is ``multiprocessing.Process.join(timeout+1)`` followed
    by ``p.kill()`` — this catches cases where the signal-based timeout
    itself is blocked or delayed (e.g. inside an uninterruptible syscall).

    Uses ``spawn`` context for cross-platform safety (``fork`` is unsafe on
    macOS) and a temporary JSON file for IPC so that no complex object
    pickling (queues, pipes) is needed across the spawn boundary.

    Returns a dict with keys ``task_id``, ``passed``, ``result``, ``stderr``.
    """
    # Default to "fork" for library‑friendly usage (no ``if __name__``
    # guard required in the caller).  On macOS, set ``LLMEVAL_MP_METHOD=spawn``
    # if your calling script has proper ``if __name__ == "__main__"`` guards
    # and you want to avoid potential fork‑safety issues.
    _mp_method = os.environ.get("LLMEVAL_MP_METHOD", "fork")
    try:
        ctx = multiprocessing.get_context(_mp_method)
    except ValueError:
        ctx = multiprocessing.get_context()

    # Temporary JSON file for IPC — avoids Manager / pickling of Queue objects.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="code_eval_")
    os.close(tmp_fd)

    p = ctx.Process(
        target=_worker,
        args=(check_program, timeout, task_id, tmp_path),
    )
    try:
        p.start()
    except Exception:
        _cleanup_tmp(tmp_path)
        return {
            "task_id": task_id,
            "passed": False,
            "result": "failed: could not start worker process",
            "stderr": "",
        }

    # If the process exits before the timeout, p.join returns immediately.
    p.join(timeout + 1)

    if p.is_alive():
        p.kill()
        p.join(5)  # give SIGKILL time to deliver
        _cleanup_tmp(tmp_path)
        return {
            "task_id": task_id,
            "passed": False,
            "result": "timed out",
            "stderr": "",
        }

    if p.exitcode == -signal.SIGSEGV:
        _cleanup_tmp(tmp_path)
        return {
            "task_id": task_id,
            "passed": False,
            "result": "failed: SegmentationFault",
            "stderr": "",
        }

    try:
        with open(tmp_path, encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
    except Exception:
        result = {
            "task_id": task_id,
            "passed": False,
            "result": "failed: worker did not produce a result",
            "stderr": "",
        }
    finally:
        _cleanup_tmp(tmp_path)

    return result
