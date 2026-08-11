"""Safe subprocess code execution for code-generation evaluation.

This module provides a self-contained code execution sandbox.  Generated code
is run inside a separate process with:

* **Timeout** — a signal-based timer and a process-level ``join(timeout)``
  double safety net.
* **Dangerous-function guard** — builtins like ``exit()``, ``quit()``, and
  many ``os`` / ``shutil`` functions are disabled; modules like ``subprocess``
  and ``ctypes`` are blocked from re-import.
* **IO redirection** — ``stdout``, ``stderr``, and ``stdin`` are redirected so
  that generated code cannot read captured output or interfere with the parent
  process's streams.

.. warning::
    This is a **safety guard**, not a security sandbox.  It prevents accidental
    interference but does *not* protect against intentionally malicious code.
    Only run on trusted model outputs.
"""

from __future__ import annotations

import io
import json
import math
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import traceback
import types
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

from llmeval.utils.log import init_logger

logger = init_logger("code_execute")

__all__ = [
    "TimeoutException",
    "check_correctness",
]

# ===========================================================================
# Module-level data for reliability_guard / reliability_restore
# ===========================================================================

#: Saved values keyed by name so that ``reliability_restore`` can undo
#: all modifications made by ``reliability_guard``.
_ORIGINAL_BUILTINS: dict[str, Any] = {}
_ORIGINAL_OS_FUNCS: dict[str, Any] = {}
_ORIGINAL_SHUTIL_FUNCS: dict[str, Any] = {}
_ORIGINAL_IO_FUNCS: dict[str, Any] = {}
_ORIGINAL_MODULES: dict[str, Any] = {}

#: ``os`` functions that are disabled by ``reliability_guard``.
_FORBIDDEN_OS_FUNCTIONS: tuple[str, ...] = (
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
    "open",
    # Without this, candidate code calling ``os._exit()`` kills the worker
    # before it writes its result file, turning a wrong answer into an
    # infrastructure failure excluded from the Pass@k denominator.
    "_exit",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
)

#: ``shutil`` functions that are disabled by ``reliability_guard``.
_FORBIDDEN_SHUTIL_FUNCTIONS: tuple[str, ...] = (
    "rmtree",
    "move",
    "copy",
    "copy2",
)

#: Modules set to ``None`` in ``sys.modules`` so that ``import`` of them
#: raises ``ImportError``.
_BLOCKED_MODULES: tuple[str, ...] = (
    "subprocess",
    "faulthandler",
    "ipdb",
    "joblib",
    "resource",
    "psutil",
    "tkinter",
    "ctypes",
    "multiprocessing",
    "socket",
    "ssl",
    "urllib",
    "http",
    "httpx",
    "requests",
    "aiohttp",
    "ftplib",
)

#: Built-in names that are set to ``None`` inside the ``builtins`` module.
_DISABLED_BUILTINS: tuple[str, ...] = ("exit", "quit", "open", "input")


def _current_vsz_bytes() -> int | None:
    """Best-effort current virtual memory size in bytes (Linux ``/proc``).

    Returns ``None`` when the value cannot be determined (non-Linux
    platforms, restricted ``/proc``).
    """
    try:
        with open("/proc/self/stat", encoding="ascii") as f:
            # Field 23 (vsize); everything after the closing paren of ``comm``
            # starts at field 3, so vsize sits at index 20 of the remainder.
            return int(f.read().rsplit(")", 1)[1].split()[20])
    except (OSError, ValueError, IndexError):
        return None


def _current_fd_count() -> int | None:
    """Best-effort count of currently open file descriptors.

    Returns ``None`` when neither ``/proc/self/fd`` nor ``/dev/fd`` is
    readable.
    """
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError:
            continue
    return None


def _apply_resource_limits(timeout: float) -> None:
    """Apply best-effort Unix limits inside the disposable worker process.

    ``RLIMIT_AS`` and ``RLIMIT_NOFILE`` are *growth-relative*: with the
    default ``fork`` start method the worker inherits the parent's whole
    address space and fd table (heavy in production, where scoring workers
    are forked from a process that has the inference stack loaded), so an
    absolute budget calibrated for a fresh interpreter is already exceeded
    at birth — every normal solution then dies with ``MemoryError`` /
    ``EMFILE`` and pass@1 collapses to 0.  The configured budget is therefore
    applied as headroom on top of the inherited baseline.  Under ``spawn``
    the baseline is a fresh interpreter (~tens of MB / a handful of fds), so
    behavior there is effectively unchanged.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - unavailable on Windows
        return

    fd_count = _current_fd_count()
    nofile_limit = 64 if fd_count is None else max(64, fd_count + 16)
    limits = (
        (resource.RLIMIT_CPU, max(1, math.ceil(timeout) + 1)),
        (resource.RLIMIT_FSIZE, 1 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, nofile_limit),
        (resource.RLIMIT_CORE, 0),
    )
    if hasattr(resource, "RLIMIT_NPROC"):
        limits = (*limits, (resource.RLIMIT_NPROC, 32))
    memory_mb = int(os.environ.get("LLMEVAL_MEMORY_LIMIT_MB", "2048"))
    if memory_mb > 0 and hasattr(resource, "RLIMIT_AS"):
        headroom = memory_mb * 1024 * 1024
        vsz = _current_vsz_bytes()
        limits = (
            *limits,
            (resource.RLIMIT_AS, headroom if vsz is None else vsz + headroom),
        )
    for resource_name, soft_limit in limits:
        try:
            _, hard_limit = resource.getrlimit(resource_name)
            effective = soft_limit if hard_limit < 0 else min(soft_limit, hard_limit)
            resource.setrlimit(resource_name, (effective, effective))
        except (OSError, ValueError):
            continue


# ===========================================================================
# Timeout
# ===========================================================================


class TimeoutException(Exception):
    """Raised when code execution exceeds the allotted time."""


@dataclass
class _TimeoutState:
    """Record whether SIGALRM fired, even if candidate code caught it."""

    expired: bool = False


@contextmanager
def time_limit(seconds: float) -> Generator[_TimeoutState, None, None]:
    """Context manager that raises :class:`TimeoutException` after *seconds*.

    Uses ``signal.setitimer(ITIMER_REAL, ...)`` so the timeout fires even
    when the main thread is blocked in a C-extension call.

    The previous ``SIGALRM`` handler is saved and restored on exit.
    """

    state = _TimeoutState()

    def _handler(_signum: int, _frame: Any) -> None:
        state.expired = True
        raise TimeoutException("Code execution timed out")

    _prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield state
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, _prev_handler)


# ===========================================================================
# IO swallowing
# ===========================================================================


class WriteOnlyStringIO(io.StringIO):
    """A ``StringIO`` that rejects reads.

    Prevents generated code from peeking at captured output.
    """

    def read(self, *args: Any, **kwargs: Any) -> str:
        raise OSError("reading from stdout/stderr is not allowed")

    def readline(self, size: int = -1) -> str:  # type: ignore[override]
        raise OSError("reading from stdout/stderr is not allowed")

    def readlines(self, hint: int = -1) -> list[str]:  # type: ignore[override]
        raise OSError("reading from stdout/stderr is not allowed")


@contextmanager
def swallow_io() -> Generator[None, None, None]:
    """Redirect stdout, stderr, and stdin so generated code cannot interfere.

    After the block, callers can inspect ``sys.stdout.getvalue()`` etc.
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


# ===========================================================================
# Reliability guard
# ===========================================================================


def _get_builtins_module() -> types.ModuleType:
    """Return the ``builtins`` module regardless of execution context.

    Inside ``multiprocessing.Process`` or ``exec()``, ``__builtins__`` is
    the module's ``__dict__`` instead of the module itself.  This helper
    always returns the real module.
    """
    import builtins

    return builtins


def reliability_guard() -> None:
    """Disable dangerous builtins and stdlib functions.

    After calling this function, generated code that tries to:

    * call ``exit()`` / ``quit()``
    * call ``os.system()``, ``os.kill()``, ``os.remove()``, …
    * call ``shutil.rmtree()``, ``shutil.move()``, …
    * ``import subprocess``, ``import ctypes``, ``import multiprocessing``, …

    will receive an ``AttributeError`` or ``ImportError`` instead of
    causing side effects.

    Call :func:`reliability_restore` to undo all changes.
    """
    _bi = _get_builtins_module()

    # -- builtins ---------------------------------------------------------------
    # ``setdefault`` keeps the first saved value so a repeated guard call does
    # not snapshot the already-disabled ``None`` sentinels.
    for _name in _DISABLED_BUILTINS:
        _ORIGINAL_BUILTINS.setdefault(_name, getattr(_bi, _name, None))
        setattr(_bi, _name, None)

    # -- os ---------------------------------------------------------------------
    for _name in _FORBIDDEN_OS_FUNCTIONS:
        if hasattr(os, _name):
            _ORIGINAL_OS_FUNCS.setdefault(_name, getattr(os, _name))
            setattr(os, _name, None)

    # -- shutil -----------------------------------------------------------------
    for _name in _FORBIDDEN_SHUTIL_FUNCTIONS:
        if hasattr(shutil, _name):
            _ORIGINAL_SHUTIL_FUNCS.setdefault(_name, getattr(shutil, _name))
            setattr(shutil, _name, None)

    _ORIGINAL_IO_FUNCS.setdefault("open", io.open)
    io.open = None  # type: ignore[assignment]

    # -- dangerous modules (block re-import) ------------------------------------
    for _mod in _BLOCKED_MODULES:
        _ORIGINAL_MODULES.setdefault(_mod, sys.modules.get(_mod))
        # ``None`` is a supported runtime sentinel for blocking imports, but
        # typeshed models ``sys.modules`` as containing modules only.
        sys.modules[_mod] = None  # type: ignore[assignment]


def reliability_restore() -> None:
    """Reverse the effects of :func:`reliability_guard`.

    Restores the original values of all disabled builtins, ``os`` /
    ``shutil`` functions, and ``sys.modules`` entries.
    """
    _bi = _get_builtins_module()
    for _name, _val in _ORIGINAL_BUILTINS.items():
        setattr(_bi, _name, _val)
    for _name, _val in _ORIGINAL_OS_FUNCS.items():
        setattr(os, _name, _val)
    for _name, _val in _ORIGINAL_SHUTIL_FUNCS.items():
        setattr(shutil, _name, _val)
    for _name, _val in _ORIGINAL_IO_FUNCS.items():
        setattr(io, _name, _val)
    for _name, _val in _ORIGINAL_MODULES.items():
        if _val is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _val
    # Clear the snapshots so a later guard/restore pair starts fresh.
    _ORIGINAL_BUILTINS.clear()
    _ORIGINAL_OS_FUNCS.clear()
    _ORIGINAL_SHUTIL_FUNCS.clear()
    _ORIGINAL_IO_FUNCS.clear()
    _ORIGINAL_MODULES.clear()


# ===========================================================================
# Temp directory
# ===========================================================================


@contextmanager
def create_tempdir() -> Generator[str, None, None]:
    """Create a temporary directory, ``chdir`` into it, and clean up on exit.

    Yields the path of the temporary directory.
    """
    _orig_cwd = os.getcwd()
    _td = tempfile.TemporaryDirectory()
    try:
        os.chdir(_td.name)
        yield _td.name
    finally:
        os.chdir(_orig_cwd)
        _td.cleanup()


# ===========================================================================
# Core execution
# ===========================================================================


def unsafe_execute(
    check_program: str,
    timeout: float,
    exec_globals: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Execute *check_program* inside safety guards with a timeout.

    Parameters
    ----------
    check_program : str
        Python source to execute (candidate function + test harness).
    timeout : float
        Maximum seconds before raising :exc:`TimeoutException`.
    exec_globals : dict or None
        Optional globals dict for ``exec()``. ``None`` creates fresh globals
        with ``__name__`` set to ``"__main__"``.

    Returns
    -------
    tuple[str, str]
        ``(status, stderr_traceback)`` where *status* is one of
        ``"passed"``, ``"timed out"``, or ``"failed: <ExceptionName>"``.
    """
    if exec_globals is None:
        exec_globals = {"__name__": "__main__"}
    else:
        exec_globals.setdefault("__name__", "__main__")

    with create_tempdir():
        reliability_guard()
        try:
            with swallow_io(), time_limit(timeout) as timeout_state:
                exec(check_program, exec_globals)
            if timeout_state.expired:
                return ("timed out", "")
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


# ===========================================================================
# Multiprocess wrapper
# ===========================================================================


def _worker(
    check_program: str,
    timeout: float,
    task_id: str,
    result_file: str,
) -> None:
    """Run *check_program* in a child process and write the result to disk.

    This function is the ``target`` of ``multiprocessing.Process``.
    """
    _apply_resource_limits(timeout)
    os.environ.clear()
    os.environ["PYTHONIOENCODING"] = "utf-8"
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
    """Remove *path*, ignoring ``OSError`` if it does not exist."""
    with suppress(OSError):
        os.unlink(path)


#: Methods whose resolution has already been logged in this process.  The
#: "effective mp method" line is emitted once per method per process so that
#: per-sample ``check_correctness`` calls do not flood the log.
_LOGGED_MP_METHODS: set[str] = set()


def _resolve_mp_method() -> str:
    """Resolve the multiprocessing start method for worker processes.

    Linux defaults to ``"fork"`` for low per-sample startup overhead. macOS
    and platforms without ``fork`` default to ``"spawn"`` because forking a
    multi-threaded process can deadlock. The environment variable
    ``LLMEVAL_MP_METHOD`` explicitly overrides the default; a value not in
    :func:`multiprocessing.get_all_start_methods` raises ``ValueError``
    instead of silently falling back.
    """
    available = multiprocessing.get_all_start_methods()

    override = os.environ.get("LLMEVAL_MP_METHOD")
    if override:
        if override not in available:
            raise ValueError(
                f"invalid LLMEVAL_MP_METHOD={override!r}; "
                f"supported start methods on this platform: {available}"
            )
        method = override
    elif sys.platform == "darwin" and "spawn" in available:
        method = "spawn"
    elif "fork" in available:
        method = "fork"
    else:
        method = "spawn"
        logger.warning(
            "fork start method not supported on this platform (%s); "
            "falling back to spawn — per-sample interpreter startup will "
            "consume part of the execution timeout budget",
            available,
        )

    if method not in _LOGGED_MP_METHODS:
        _LOGGED_MP_METHODS.add(method)
        logger.info("code execution mp method: %s (supported: %s)", method, available)
    return method


def check_correctness(
    check_program: str,
    timeout: float,
    task_id: str = "",
    allow_unsafe_code: bool = False,
) -> dict[str, Any]:
    """Execute *check_program* in a child process with a double safety net.

    The **inner** guard is :func:`unsafe_execute` (signal-based timeout +
    reliability guard).  The **outer** guard is a process-level
    ``join(timeout+5)`` followed by ``kill()`` — this catches cases where
    the signal-based timeout is blocked (e.g. inside an uninterruptible
    syscall).  The 5s margin (not 1s) leaves room for interpreter startup,
    tempdir setup and result-file writes on loaded machines, so a worker
    that already finished is not misreported as "timed out".

    IPC uses a temporary JSON file so no complex object pickling (queues,
    pipes, managers) is needed across the process boundary.

    ``allow_unsafe_code`` is an explicit safety boundary.  The lower-level
    :func:`unsafe_execute` helper remains available for trusted internal tests,
    but production callers should use this function with an isolated runtime.

    Linux defaults to ``"fork"`` because scoring already runs inside Pebble
    workers and ``check_correctness`` starts one child per sample. macOS uses
    ``"spawn"`` to avoid the deadlock risk of forking a multi-threaded process.

    .. warning::
        Forking a multi-threaded process can deadlock if the child inherits a
        held lock. Linux callers with background threads should set
        ``LLMEVAL_MP_METHOD=spawn``. An invalid override raises ``ValueError``.

    Returns
    -------
    dict[str, Any]
        Keys: ``task_id``, ``passed`` (bool), ``result`` (str), ``stderr`` (str).
    """
    if not allow_unsafe_code:
        return _fail(task_id, "unsafe execution disabled")

    # -- resolve multiprocessing context ----------------------------------------
    ctx = multiprocessing.get_context(_resolve_mp_method())

    # -- create temporary result file -------------------------------------------
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="code_eval_")
    os.close(tmp_fd)

    # -- start worker process ---------------------------------------------------
    process_factory: Any = ctx.Process  # type: ignore[attr-defined]
    p = process_factory(
        target=_worker,
        args=(check_program, timeout, task_id, tmp_path),
    )
    try:
        p.start()
    except Exception:
        _cleanup_tmp(tmp_path)
        return _fail(task_id, "failed: could not start worker process")

    p.join(timeout + 5)

    # -- timeout -----------------------------------------------------------------
    # The worker did not finish even after the inner signal-based timeout
    # plus the startup margin — it is hung (deadlock, uninterruptible
    # syscall), not merely running slow candidate code.  Keep this distinct
    # from a candidate-level "timed out" (reported via the result file):
    # only the latter counts as an incorrect model observation.
    if p.is_alive():
        p.kill()
        p.join(5)
        _cleanup_tmp(tmp_path)
        return _fail(task_id, "timed out: worker killed")

    # -- segmentation fault -----------------------------------------------------
    if p.exitcode == -signal.SIGSEGV:
        _cleanup_tmp(tmp_path)
        return _fail(task_id, "failed: SegmentationFault")

    # -- killed by another signal ------------------------------------------------
    # e.g. SIGKILL/SIGXCPU from RLIMIT_CPU on an infinite loop, or the OOM
    # killer. The candidate code caused this, so report it separately from a
    # worker that simply failed to write its result file.
    if p.exitcode is not None and p.exitcode < 0:
        _cleanup_tmp(tmp_path)
        return _fail(task_id, f"failed: killed by signal {-p.exitcode}")

    # -- collect result ----------------------------------------------------------
    try:
        with open(tmp_path, encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
    except Exception:
        result = _fail(task_id, "failed: worker did not produce a result")
    finally:
        _cleanup_tmp(tmp_path)

    return result


def _fail(task_id: str, reason: str) -> dict[str, Any]:
    """Return a uniform failure record."""
    return {
        "task_id": task_id,
        "passed": False,
        "result": reason,
        "stderr": "",
    }
