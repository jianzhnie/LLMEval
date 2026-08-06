"""Boundary tests for scripts/data_process/post_process.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_process"
    / "post_process.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("post_process", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "has_boxed,total_length,expected",
    [
        (True, 128, True),
        (True, 129, False),
        (False, 128, False),
        (False, 129, False),
    ],
)
def test_filter_truth_table(
    has_boxed: bool, total_length: int, expected: bool
) -> None:
    module = _load_script()

    assert (
        module.should_keep_example(
            {"has_boxed": has_boxed, "total_token_length": total_length}, 128
        )
        is expected
    )


def test_empty_dataset_does_not_divide_by_zero() -> None:
    datasets = pytest.importorskip("datasets")
    module = _load_script()
    dataset = datasets.Dataset.from_dict({"prompt": [], "gen": []})

    result = module.process_dataset(
        dataset,
        tokenizer=lambda text, **_: {"input_ids": text.split()},
        max_token_length=128,
        num_proc=1,
    )

    assert len(result) == 0
