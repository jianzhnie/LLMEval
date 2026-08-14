"""Tests for llmeval.tasks.code_eval.code_score — the scoring driver.

These tests only exercise the **serial** scoring path and the pure utility
functions so that heavy dependencies (``pebble``, ``multiprocessing``) are
stubbed or avoided entirely.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
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
    _code_worker_timeout,
    _failure_code_record,
    _process_code_item,
    estimate_pass_at_k,
    extract_code,
    score_code_result,
)
from llmeval.tasks.postprocess import strip_reasoning_wrappers


def _score_code_value(
    eval_dataset: list[dict],
    label_key: str,
    response_key: str,
    _result_path: Path | None = None,
    **kwargs: object,
) -> float:
    """Return the primary metric while tests exercise the structured API."""
    return score_code_result(eval_dataset, label_key, response_key, **kwargs).metrics[
        "pass@1"
    ]


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

    assert record["filtered_gen"] == ""
    assert "raw_gen" not in record
    assert record["filter_trace"]["pipeline"] == "code_generation"


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
        with pytest.raises(ValueError, match="num_samples"):
            estimate_pass_at_k(0, 0, 1)

    def test_negative_correct(self) -> None:
        with pytest.raises(ValueError, match="num_correct"):
            estimate_pass_at_k(5, -1, 1)

    @pytest.mark.parametrize(
        "args", [(2, 3, 1), (2, 1, 0), (2, 1, 3), (2.0, 1, 1), (2, 1, True)]
    )
    def test_invalid_inputs_raise(self, args: tuple[object, object, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            estimate_pass_at_k(*args)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# Structured code scoring (serial path only)
# ═══════════════════════════════════════════════════════════════════════


class TestScoreCode:
    def test_worker_exception_returns_indexed_failure(self) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        index, record = code_score._process_code_item(
            (0, {}, "answer", "gen", 1.0, True)
        )

        assert index == 0
        assert record["evaluation_status"] == "failed"

    def test_serial_worker_exception_is_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        monkeypatch.setattr(
            code_score,
            "_process_code_item",
            MagicMock(side_effect=RuntimeError("worker failed")),
        )
        result = score_code_result(
            [{"task_id": "task/0", "prompt": "def f():\n", "answer": "", "gen": [""]}],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.failed_count == 1
        assert result.records[0]["evaluation_status"] == "failed"

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
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.metrics["pass@1"] == 0.0
        assert result.failed_count == 0
        assert result.effective_sample_count == 1

    def test_inference_error_is_excluded_from_pass_at_k(self) -> None:
        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": "",
                    "error": "context_length_exceeded",
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.failed_count == 1
        assert result.effective_sample_count == 0
        assert result.observations["pass@1"] == []

    @pytest.mark.parametrize("generation", [None, [None], 123])
    def test_malformed_generation_is_infrastructure_failure(
        self, generation: object
    ) -> None:
        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": generation,
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.failed_count == 1
        assert result.records[0]["evaluation_status"] == "failed"

    def test_all_infrastructure_failures_remain_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        check = MagicMock(
            return_value={
                "task_id": "task/0",
                "passed": False,
                "result": "worker failed",
                "stderr": "",
                "evaluation_status": "failed",
            }
        )
        monkeypatch.setattr(code_score, "check_correctness", check)
        monkeypatch.setattr(
            code_score,
            "_build_check_programs",
            MagicMock(return_value=[("first", "a"), ("fallback", "b")]),
        )

        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "assert f() == 1",
                    "gen": "return 1",
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.failed_count == 1
        assert check.call_count == 2

    def test_fallback_success_overrides_infrastructure_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        check = MagicMock(
            side_effect=[
                {
                    "task_id": "task/0",
                    "passed": False,
                    "result": "worker failed",
                    "stderr": "",
                    "evaluation_status": "failed",
                },
                {
                    "task_id": "task/0",
                    "passed": True,
                    "result": "passed",
                    "stderr": "",
                    "evaluation_status": "completed",
                },
            ]
        )
        monkeypatch.setattr(code_score, "check_correctness", check)
        monkeypatch.setattr(
            code_score,
            "_build_check_programs",
            MagicMock(return_value=[("first", "a"), ("fallback", "b")]),
        )

        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "assert f() == 1",
                    "gen": "return 1",
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert check.call_count == 2
        assert result.records[0]["passed"] is True
        assert result.failed_count == 0

    def test_fallback_failure_preserves_completed_incorrect_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        check = MagicMock(
            side_effect=[
                {
                    "task_id": "task/0",
                    "passed": False,
                    "result": "failed: AssertionError",
                    "stderr": "",
                    "evaluation_status": "completed",
                },
                {
                    "task_id": "task/0",
                    "passed": False,
                    "result": "worker failed",
                    "stderr": "",
                    "evaluation_status": "failed",
                },
            ]
        )
        monkeypatch.setattr(code_score, "check_correctness", check)
        monkeypatch.setattr(
            code_score,
            "_build_check_programs",
            MagicMock(return_value=[("first", "a"), ("fallback", "b")]),
        )

        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "assert f() == 1",
                    "gen": "return 2",
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert check.call_count == 2
        assert result.records[0]["evaluation_status"] == "completed"
        assert result.records[0]["result"] == "failed: AssertionError"
        assert result.failed_count == 0
        assert result.effective_sample_count == 1

    def test_fallback_exception_preserves_completed_incorrect_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        check = MagicMock(
            side_effect=[
                {
                    "task_id": "task/0",
                    "passed": False,
                    "result": "failed: AssertionError",
                    "stderr": "",
                    "evaluation_status": "completed",
                },
                OSError("executor unavailable"),
            ]
        )
        monkeypatch.setattr(code_score, "check_correctness", check)
        monkeypatch.setattr(
            code_score,
            "_build_check_programs",
            MagicMock(return_value=[("first", "a"), ("fallback", "b")]),
        )

        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "assert f() == 1",
                    "gen": "return 2",
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert check.call_count == 2
        assert result.records[0]["evaluation_status"] == "completed"
        assert result.records[0]["result"] == "failed: AssertionError"
        assert result.failed_count == 0
        assert result.effective_sample_count == 1

    def test_invalid_fallback_status_preserves_completed_incorrect_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import llmeval.tasks.code_eval.code_score as code_score

        check = MagicMock(
            side_effect=[
                {
                    "task_id": "task/0",
                    "passed": False,
                    "result": "failed: AssertionError",
                    "stderr": "",
                    "evaluation_status": "completed",
                },
                {
                    "task_id": "task/0",
                    "passed": False,
                    "result": "unknown",
                    "stderr": "",
                },
            ]
        )
        monkeypatch.setattr(code_score, "check_correctness", check)
        monkeypatch.setattr(
            code_score,
            "_build_check_programs",
            MagicMock(return_value=[("first", "a"), ("fallback", "b")]),
        )

        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "assert f() == 1",
                    "gen": "return 2",
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.records[0]["evaluation_status"] == "completed"
        assert result.records[0]["result"] == "failed: AssertionError"
        assert result.failed_count == 0

    def test_incomplete_problem_is_excluded_from_problem_level_pass_at_k(
        self,
    ) -> None:
        result = score_code_result(
            [
                {
                    "task_id": "task/incomplete",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": "    return 1",
                    "sample_index": 0,
                },
                {
                    "task_id": "task/incomplete",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": "",
                    "error": "context_length_exceeded",
                    "sample_index": 1,
                },
                {
                    "task_id": "task/complete",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": "    return 2",
                    "sample_index": 0,
                },
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.metrics["pass@1"] == 0.0
        assert result.observations["pass@1"] == [0.0]
        assert result.effective_sample_count == 1
        assert result.failed_count == 1
        assert result.excluded_count == 1
        assert result.details == {
            "complete_problem_count": 1,
            "incomplete_problem_count": 1,
            "excluded_problem_task_ids": ["task/incomplete"],
        }

    def test_missing_sample_row_warns_and_excludes_problem_metrics(self) -> None:
        result = score_code_result(
            [
                {
                    "task_id": "task/incomplete",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": "    return 1",
                    "sample_index": 0,
                    "n_samples": 2,
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        assert result.effective_sample_count == 0
        assert result.failed_count == 0
        assert result.excluded_count == 1
        assert result.metrics["pass@1"] == 0.0
        assert result.observations["pass@1"] == []
        assert result.details["incomplete_problem_count"] == 1
        assert result.details["excluded_problem_task_ids"] == ["task/incomplete"]

    def test_os_exit_candidate_is_completed_not_infra_failure(
        self, tmp_path: Path
    ) -> None:
        """A candidate calling ``os._exit()`` stays in the Pass@k denominator."""
        result = score_code_result(
            [
                {
                    "task_id": "task/0",
                    "prompt": "def f():\n",
                    "answer": "\nassert f() == 1\n",
                    "gen": ["import os\nos._exit(0)"],
                }
            ],
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )

        record = result.records[0]
        assert record["passed"] is False
        assert record["evaluation_status"] == "completed"
        assert result.failed_count == 0
        assert result.effective_sample_count == 1
        assert result.metrics["pass@1"] == 0.0

    def test_pool_timeout_is_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing worker result is represented as a failed sample."""
        import llmeval.tasks.code_eval.code_score as code_score

        class _EmptyFuture:
            def result(self) -> object:
                return iter([])

        class _FakePool:
            def __init__(self, max_workers: int) -> None:
                pass

            def __enter__(self) -> _FakePool:
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

            def map(self, *_args: object, **_kwargs: object) -> _EmptyFuture:
                return _EmptyFuture()

        monkeypatch.setattr(code_score, "ProcessPool", _FakePool)
        items = [
            {
                "task_id": f"task/{index}",
                "prompt": "def f():\n",
                "answer": "\nassert f() == 1\n",
                "gen": ["    return 1"],
            }
            for index in range(2)
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=2,
            allow_unsafe_code=True,
        )

        assert result.sample_count == 2
        assert result.failed_count == 2
        assert result.effective_sample_count == 0
        assert all(r["evaluation_status"] == "failed" for r in result.records)

    def test_execution_requires_explicit_opt_in(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError, match="executes generated code"):
            _score_code_value(
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
                max_workers=1,
            )

    def test_all_pass(self) -> None:
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
            },
            {
                "task_id": "task/1",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a - b"],
            },
        ]
        acc = _score_code_value(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=3.0,
            allow_unsafe_code=True,
        )
        assert acc == 1.0

    def test_mixed_results(self) -> None:
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(2, 3) == 5\n",
                "gen": ["    return a + b"],
            },  # correct
            {
                "task_id": "task/1",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a * b"],
            },  # wrong
        ]
        acc = _score_code_value(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=3.0,
            allow_unsafe_code=True,
        )
        assert acc == 0.5

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
            },
            {
                "task_id": "task/1",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a - b"],
            },
        ]
        acc = _score_code_value(
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
        acc = _score_code_value([], "answer", "gen", max_workers=1)
        assert acc == 0.0

    def test_empty_generation_marked_failed(self) -> None:
        items = [
            {
                "task_id": "task/0",
                "prompt": "def foo():\n",
                "answer": "\nassert foo() == 1\n",
                "gen": [""],
            },
        ]
        acc = _score_code_value(
            items,
            "answer",
            "gen",
            max_workers=1,
            allow_unsafe_code=True,
        )
        assert acc == 0.0

    def test_multi_sample_pass_at_k_summary(self, tmp_path: Path) -> None:
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a * b"],
            },
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a + b"],
            },
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=3.0,
            k_values=(1, 2),
            allow_unsafe_code=True,
        )

        assert result.metrics["pass@1"] == pytest.approx(0.5)
        assert result.metrics["pass@2"] == 1.0
        assert result.sample_count == 2

    def test_pass_at_1_reported_when_k_values_exclude_1(self, tmp_path: Path) -> None:
        """pass@1 is computed even when the caller's k_values omit 1."""
        items = [
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a * b"],
            },
            {
                "task_id": "task/0",
                "prompt": "def add(a, b):\n",
                "answer": "\nassert add(1, 2) == 3\n",
                "gen": ["    return a + b"],
            },
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=3.0,
            k_values=(2,),
            allow_unsafe_code=True,
        )

        assert result.metrics["pass@1"] == pytest.approx(0.5)
        assert result.metrics["pass@2"] == pytest.approx(1.0)
        assert result.observations["pass@1"] == [pytest.approx(0.5)]

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
            }
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=3.0,
            allow_unsafe_code=True,
        )

        record = result.records[0]
        assert record["passed"] is False
        assert record["result"].startswith("failed: killed by signal")
        assert record["evaluation_status"] == "completed"
        assert result.failed_count == 0
        assert result.effective_sample_count == 1
        assert result.metrics["pass@1"] == 0.0

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
        assert csr.records == []

    def test_populated(self) -> None:
        csr = CodeScoreResult(pass_at_1=0.5, total=4, correct=2, records=[])
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
# Code scoring with reasoning-model wrappers
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
        acc = _score_code_value(
            items,
            "answer",
            "gen",
            tmp_path / "cache.jsonl",
            max_workers=1,
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
            }
        ]

        acc = _score_code_value(
            items,
            "answer",
            "gen",
            tmp_path / "mbpp.jsonl",
            max_workers=1,
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
            }
        ]

        acc = _score_code_value(
            items,
            "answer",
            "gen",
            tmp_path / "humaneval.jsonl",
            max_workers=1,
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
            }
        ]
        acc = _score_code_value(
            items,
            "answer",
            "gen",
            tmp_path / "cache.jsonl",
            max_workers=1,
            allow_unsafe_code=True,
        )
        assert acc == 1.0


# ═══════════════════════════════════════════════════════════════════════
# Sample-index protocol (P0-1)
# ═══════════════════════════════════════════════════════════════════════


class TestCodeRepeatedRows:
    """Repeated rows are independent code-generation samples."""

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
            **extra,
        }

    def _score(self, items: list[dict], tmp_path: Path):
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=3.0,
            k_values=(1, 2),
            allow_unsafe_code=True,
        )
        return result.metrics["pass@1"], result.records

    def test_repeated_rows_are_scored_independently(self, tmp_path: Path) -> None:
        items = [
            self._item([self._WRONG]),
            self._item([self._RIGHT]),
        ]
        acc, records = self._score(items, tmp_path)
        assert len(records) == 2
        assert acc == pytest.approx(0.5)

    def test_multi_generation_row_is_failed(self, tmp_path: Path) -> None:
        items = [self._item([self._WRONG, self._RIGHT])]
        _, records = self._score(items, tmp_path)

        assert records[0]["evaluation_status"] == "failed"

    def test_multi_sample_row_requires_task_id(self, tmp_path: Path) -> None:
        item = self._item([self._WRONG, self._RIGHT])
        item.pop("task_id")

        with pytest.raises(ValueError, match=r"missing required.*task_id"):
            self._score([item], tmp_path)

    def test_empty_generation_is_recorded(self, tmp_path: Path) -> None:
        items = [self._item([])]

        acc, records = self._score(items, tmp_path)

        assert acc == 0.0
        assert len(records) == 1
        assert records[0]["result"] == "failed: empty generation"

    def test_identical_responses_remain_independent_samples(
        self, tmp_path: Path
    ) -> None:
        items = [
            self._item([self._RIGHT]),
            self._item([self._RIGHT]),
        ]
        acc, records = self._score(items, tmp_path)
        assert acc == 1.0
        assert len(records) == 2

    def test_conflicting_duplicate_rows_raise(self, tmp_path: Path) -> None:
        """Same task_id with a different test harness signals a corrupt resume."""
        rows = [
            self._item([self._RIGHT]),
            self._item([self._RIGHT], answer="\nassert add(1, 2) == 4\n"),
        ]
        with pytest.raises(ValueError, match="Conflicting 'answer'"):
            self._score(rows, tmp_path)

    def test_different_generations_remain_distinct_samples(
        self, tmp_path: Path
    ) -> None:
        items = [
            self._item([self._WRONG]),
            self._item([self._RIGHT]),
        ]
        acc, records = self._score(items, tmp_path)
        assert acc == pytest.approx(0.5)
        assert len(records) == 2


# ═══════════════════════════════════════════════════════════════════════
# Timeout classification (candidate timeout = wrong, worker hang = excluded)
# ═══════════════════════════════════════════════════════════════════════


class TestTimeoutClassification:
    def test_pool_timeout_covers_nested_execution_budget(self) -> None:
        # 2 candidates × (exec_timeout + join margin + kill margin) + overhead
        assert _code_worker_timeout(20, 3.0) == 27  # 2*(3+5+5)+1 = 27
        assert _code_worker_timeout(20, 30.0) == 81  # 2*(30+5+5)+1 = 81
        assert _code_worker_timeout(20, 0.5) == 22  # 2*(0.5+5+5)+1 = 22

    def test_candidate_infinite_loop_counts_as_wrong(self, tmp_path: Path) -> None:
        """A candidate that dead-loops must stay in the Pass@k denominator."""
        items = [
            {
                "task_id": "task/0",
                "prompt": "def f():\n",
                "answer": "\nf()\n",
                "gen": ["    while True:\n        pass"],
            }
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=1,
            exec_timeout=1.0,
            allow_unsafe_code=True,
        )
        record = result.records[0]
        assert record["result"] == "timed out"
        assert record["evaluation_status"] == "completed"
        assert result.metrics["pass@1"] == 0.0
        assert result.failed_count == 0
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
            },
            {
                "task_id": "task/1",
                "prompt": "def square(x):\n",
                "answer": "\nassert square(3) == 9\nassert square(4) == 16\n",
                "gen": ["    return x * x"],
            },
            {
                "task_id": "task/2",
                "prompt": "def sub(a, b):\n",
                "answer": "\nassert sub(5, 2) == 3\n",
                "gen": ["    return a + b"],  # wrong on purpose
            },
        ]
        result = score_code_result(
            items,
            "answer",
            "gen",
            max_workers=2,
            timeout=60,
            exec_timeout=5.0,
            allow_unsafe_code=True,
        )
        statuses = {r["task_id"]: r["evaluation_status"] for r in result.records}
        assert all(status == "completed" for status in statuses.values())
        assert result.metrics["pass@1"] == pytest.approx(2 / 3)
        assert result.failed_count == 0
