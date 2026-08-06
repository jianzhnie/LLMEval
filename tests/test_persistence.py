"""Tests for atomic task-result persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmeval.tasks.persistence import atomic_write_jsonl


def test_atomic_jsonl_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    output.write_text('{"old": true}\n')

    atomic_write_jsonl(output, ({"index": index} for index in range(2)))

    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"index": 0},
        {"index": 1},
    ]


def test_serialization_failure_preserves_old_file_and_cleans_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.jsonl"
    original = '{"old": true}\n'
    output.write_text(original)

    with pytest.raises(TypeError):
        atomic_write_jsonl(output, [{"ok": 1}, {"bad": object()}])

    assert output.read_text() == original
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_interrupted_generator_preserves_old_file_and_cleans_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.jsonl"
    original = '{"old": true}\n'
    output.write_text(original)

    def records():
        yield {"ok": 1}
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        atomic_write_jsonl(output, records())

    assert output.read_text() == original
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
