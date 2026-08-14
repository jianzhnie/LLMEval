"""Tests for code benchmark schema conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.data_process import prepare_code_benchmarks as prepare


def test_prepare_humaneval_builds_check_harness() -> None:
    records = prepare.prepare_humaneval(
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                "entry_point": "add",
            }
        ]
    )

    assert records == [
        {
            "doc_id": "humaneval:HumanEval/0",
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "answer": (
                "\ndef check(candidate):\n    assert candidate(1, 2) == 3\ncheck(add)"
            ),
            "prompt_mode": "human_eval",
        }
    ]


def test_prepare_humaneval_plus_uses_plus_prompt_mode() -> None:
    records = prepare.prepare_humaneval_plus(
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
                "entry_point": "add",
            }
        ]
    )

    assert records[0]["answer"].endswith("\ncheck(add)")
    assert records[0]["prompt_mode"] == "human_eval_plus"


def test_prepare_mbpp_builds_prompt_from_base_tests() -> None:
    records = prepare.prepare_mbpp(
        [
            {
                "task_id": 12,
                "text": "Return the sum of two integers.",
                "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
            }
        ]
    )

    assert records[0]["task_id"] == "12"
    assert "Return the sum of two integers." in records[0]["prompt"]
    assert "assert add(-1, 1) == 0" in records[0]["prompt"]
    assert records[0]["answer"] == ("\nassert add(1, 2) == 3\nassert add(-1, 1) == 0")
    assert records[0]["prompt_mode"] == "mbpp"


def test_prepare_mbpp_plus_uses_full_enhanced_harness() -> None:
    full_harness = (
        "import numpy as np\n"
        "inputs = [(1, 2), (1000, 2000)]\n"
        "expected = [3, 3000]\n"
        "for args, result in zip(inputs, expected):\n"
        "    assert np.isclose(add(*args), result)"
    )
    records = prepare.prepare_mbpp_plus(
        [
            {
                "task_id": 12,
                "prompt": "Return the sum of two integers.",
                "test_list": ["assert add(1, 2) == 3"],
                "test": full_harness,
            }
        ]
    )

    assert records[0]["prompt"] == "Return the sum of two integers."
    assert records[0]["answer"] == "\n" + full_harness
    assert records[0]["prompt_mode"] == "mbpp_plus"


def test_prepare_mbpp_plus_rejects_base_schema_without_full_harness() -> None:
    with pytest.raises(ValueError, match="non-empty string field 'test'"):
        prepare.prepare_mbpp_plus(
            [
                {
                    "task_id": 12,
                    "prompt": "Return the sum of two integers.",
                    "test_list": ["assert add(1, 2) == 3"],
                }
            ]
        )


def test_main_continues_after_conversion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets: dict[str, list[dict[str, Any]]] = {
        "evalplus/mbppplus": [
            {
                "task_id": 1,
                "prompt": "Return one.",
                "test_list": ["assert one() == 1"],
            }
        ],
        "openai/openai_humaneval": [
            {
                "task_id": "HumanEval/0",
                "prompt": "def one():\n",
                "test": "def check(candidate):\n    assert candidate() == 1",
                "entry_point": "one",
            }
        ],
    }

    def fake_load_dataset(
        path: str, _config: str | None, *, split: str
    ) -> list[dict[str, Any]]:
        assert split == "test"
        return datasets[path]

    monkeypatch.setattr(prepare, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_code_benchmarks.py",
            "--benchmarks",
            "mbpp_plus",
            "humaneval",
            "--output_dir",
            str(tmp_path),
        ],
    )

    assert prepare.main() == 1
    assert not (tmp_path / "mbpp_plus.jsonl").exists()
    output = (tmp_path / "humaneval.jsonl").read_text(encoding="utf-8")
    assert json.loads(output)["task_id"] == "HumanEval/0"
