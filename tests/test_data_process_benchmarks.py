"""Tests for benchmark registry consistency in data preparation scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
MATH_SCRIPT = ROOT / "scripts" / "data_process" / "prepare_math_benchmarks.py"
MC_SCRIPT = ROOT / "scripts" / "data_process" / "prepare_mc_benchmarks.py"
CODE_SCRIPT = ROOT / "scripts" / "data_process" / "prepare_code_benchmarks.py"


def _load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_math_prepare_registry_supports_documented_aime_benchmarks() -> None:
    module = _load_script(MATH_SCRIPT)

    assert module.BENCHMARKS["aime24"] == (
        "math-ai/aime24",
        "default",
        "test",
        "problem",
        "solution",
    )
    assert module.BENCHMARKS["aime25"] == (
        "math-ai/aime25",
        "default",
        "test",
        "problem",
        "answer",
    )
    assert module.BENCHMARKS["aime26"] == (
        "math-ai/aime26",
        "default",
        "test",
        "problem",
        "answer",
    )
    assert not hasattr(module, "MC_BENCHMARKS")
    assert not hasattr(module, "prepare_mc_benchmark")


def test_mc_prepare_registry_supports_only_mc_benchmarks() -> None:
    mc_module = _load_script(MC_SCRIPT)

    assert set(mc_module.BENCHMARKS) == {"mmlu", "mmlu_pro", "ceval"}
    assert "aime24" not in mc_module.BENCHMARKS
    assert not hasattr(mc_module, "QWEN_MATH_COT_PROMPT")
    assert not hasattr(mc_module, "prepare_benchmark")


def test_math_formatter_persists_document_id() -> None:
    module = _load_script(MATH_SCRIPT)

    row = module.format_example(
        {"problem": "1+1", "answer": "2"},
        3,
        "aime24",
        "problem",
        "answer",
    )

    assert row["doc_id"] == "aime24:3"


def test_mc_formatter_persists_document_id() -> None:
    module = _load_script(MC_SCRIPT)

    row = module._format_mc_row(
        "mmlu",
        {
            "question": "1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": 1,
        },
        source_id="algebra:3",
    )

    assert row["doc_id"] == "mmlu:algebra:3"


def test_code_preparers_persist_document_id() -> None:
    module = _load_script(CODE_SCRIPT)

    humaneval = module.prepare_humaneval(
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def f():\n",
                "test": "assert f() == 1",
                "entry_point": "f",
            }
        ],
        "humaneval",
    )
    mbpp = module.prepare_mbpp(
        [{"task_id": 7, "text": "return one", "test_list": ["assert f()"]}],
        "mbpp",
    )

    assert humaneval[0]["doc_id"] == "humaneval:HumanEval/0"
    assert mbpp[0]["doc_id"] == "mbpp:7"
