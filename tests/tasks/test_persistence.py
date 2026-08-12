"""Tests for atomic task-result persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmeval.tasks.postprocess import atomic_write_json


def test_atomic_json_writes_exact_destination(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    atomic_write_json(output, {"accuracy": 1.0}, indent=2)

    assert json.loads(output.read_text()) == {"accuracy": 1.0}


def test_atomic_json_rejects_non_finite_numbers(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(output, {"score": float("nan")})

    assert not output.exists()
