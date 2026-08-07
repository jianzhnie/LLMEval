from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_jsonl_items() -> list[dict[str, Any]]:
    """Minimal math-problem items matching the project's JSONL schema."""
    return [
        {
            "prompt": "What is 2 + 3?",
            "answer": "5",
        },
        {
            "prompt": "Solve x^2 = 4",
            "answer": "2",
        },
    ]


@pytest.fixture
def sample_jsonl_with_gen() -> list[dict[str, Any]]:
    """Items that already carry a 'gen' list (partially completed)."""
    return [
        {
            "prompt": "What is 2 + 3?",
            "answer": "5",
            "gen": ["The answer is 5"],
        },
    ]


@pytest.fixture
def tmp_input_file(sample_jsonl_items: list[dict[str, Any]], tmp_path: Path) -> str:
    """Write sample items to a temporary JSONL file and return its path."""
    p = tmp_path / "input.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for item in sample_jsonl_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(p)


@pytest.fixture
def tmp_output_file(tmp_path: Path) -> str:
    """Return a path for a temporary output JSONL file."""
    return str(tmp_path / "output.jsonl")
