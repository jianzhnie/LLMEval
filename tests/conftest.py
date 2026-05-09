from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


@pytest.fixture
def sample_jsonl_items() -> List[Dict[str, Any]]:
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
def sample_jsonl_with_gen() -> List[Dict[str, Any]]:
    """Items that already carry a 'gen' list (partially completed)."""
    return [
        {
            "prompt": "What is 2 + 3?",
            "answer": "5",
            "gen": ["The answer is 5"],
        },
    ]


@pytest.fixture
def verifier_input_items() -> List[Dict[str, Any]]:
    """Items suitable for verifier inference (prompt + answer + gen)."""
    return [
        {
            "prompt": "What is 1+1?",
            "answer": "2",
            "gen": ["Let me think...\n</think />\n\nThe answer is <answer>2</answer>"],
        },
        {
            "prompt": "Solve x=3",
            "answer": "3",
            "gen": ["\\boxed{3}"],
        },
    ]


@pytest.fixture
def tmp_input_file(sample_jsonl_items: List[Dict[str, Any]],
                   tmp_path: Path) -> str:
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
