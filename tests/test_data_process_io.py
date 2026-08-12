"""Tests for dataset preparation output safety."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.data_process.io_utils import atomic_output_path, has_valid_doc_ids
from scripts.data_process.post_process import load_and_validate_args


def test_atomic_output_replaces_destination(tmp_path: Path) -> None:
    destination = tmp_path / "data.jsonl"
    destination.write_text("old\n", encoding="utf-8")

    with atomic_output_path(destination) as temporary:
        assert temporary.suffix == destination.suffix
        temporary.write_text("new\n", encoding="utf-8")

    assert destination.read_text(encoding="utf-8") == "new\n"


def test_atomic_output_preserves_destination_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "data.jsonl"
    destination.write_text("old\n", encoding="utf-8")

    with pytest.raises(RuntimeError), atomic_output_path(destination) as temporary:
        temporary.write_text("partial\n", encoding="utf-8")
        raise RuntimeError("conversion failed")

    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


def test_has_valid_doc_ids_requires_non_empty_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"doc_id": "a"}\n{"doc_id": "b"}\n', encoding="utf-8")
    assert has_valid_doc_ids(path) is True

    path.write_text('{"doc_id": "a"}\n{"doc_id": "a"}\n', encoding="utf-8")
    assert has_valid_doc_ids(path) is False


def test_post_process_rejects_output_matched_by_input_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.jsonl"
    output.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "post_process.py",
            "--input_path",
            str(tmp_path / "*.jsonl"),
            "--output_file",
            str(output),
            "--tokenizer_name_or_path",
            "unused",
        ],
    )

    with pytest.raises(SystemExit):
        load_and_validate_args()
