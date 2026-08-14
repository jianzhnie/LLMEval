#!/usr/bin/env python3
"""Prepare code benchmark data files in LLMEval JSONL format.

Downloads HumanEval / MBPP / HumanEval+ / MBPP+ from HuggingFace and
converts them to the unified schema: ``{"doc_id": ..., "task_id": ..., "prompt": ...,
"answer": ...}``.

Usage::

    python scripts/data_process/prepare_code_benchmarks.py \\
        --benchmarks humaneval mbpp \\
        --output_dir ./data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from .io_utils import atomic_output_path
except ImportError:  # Direct script execution
    from io_utils import atomic_output_path

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, tuple[str, str | None, str]] = {
    "humaneval": ("openai/openai_humaneval", None, "test"),
    "humaneval_plus": ("evalplus/humanevalplus", None, "test"),
    "mbpp": ("google-research-datasets/mbpp", "full", "test"),
    "mbpp_plus": ("evalplus/mbppplus", None, "test"),
}

PROMPT_TEMPLATE = (
    "You are an expert Python programmer.  Write a function that "
    "satisfies the following description.\n\n{text}\n\n"
    "Your code should pass these tests:\n\n{tests}\n\n[BEGIN]\n"
)


def _required_text(
    item: dict[str, Any], field: str, benchmark_name: str, index: int
) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{benchmark_name} record {index} requires a non-empty string "
            f"field {field!r}"
        )
    return value


def _task_id(item: dict[str, Any], benchmark_name: str, index: int) -> str:
    if "task_id" not in item or not str(item["task_id"]).strip():
        raise ValueError(
            f"{benchmark_name} record {index} requires a non-empty 'task_id'"
        )
    return str(item["task_id"])


def _prepare_humaneval_records(
    data: list[dict[str, Any]], benchmark_name: str, prompt_mode: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        task_id = _task_id(item, benchmark_name, index)
        prompt = _required_text(item, "prompt", benchmark_name, index)
        test = _required_text(item, "test", benchmark_name, index)
        entry_point = _required_text(item, "entry_point", benchmark_name, index)
        test_harness = f"{test.rstrip()}\ncheck({entry_point})"
        records.append(
            {
                "doc_id": f"{benchmark_name}:{task_id}",
                "task_id": task_id,
                "prompt": prompt,
                "answer": "\n" + test_harness,
                "prompt_mode": prompt_mode,
            }
        )
    return records


def prepare_humaneval(
    data: list[dict[str, Any]], benchmark_name: str = "humaneval"
) -> list[dict[str, Any]]:
    """Convert HumanEval-format data to LLMEval JSONL schema.

    Input fields: ``task_id``, ``prompt``, ``test``, ``entry_point``.
    Output fields: ``doc_id``, ``task_id``, ``prompt``, ``answer`` (test + check()).
    """
    return _prepare_humaneval_records(data, benchmark_name, "human_eval")


def prepare_humaneval_plus(
    data: list[dict[str, Any]], benchmark_name: str = "humaneval_plus"
) -> list[dict[str, Any]]:
    """Convert HumanEval+ data using its full ``check(candidate)`` harness."""
    return _prepare_humaneval_records(data, benchmark_name, "human_eval_plus")


def prepare_mbpp(
    data: list[dict[str, Any]], benchmark_name: str = "mbpp"
) -> list[dict[str, Any]]:
    """Convert MBPP-format data to LLMEval JSONL schema.

    MBPP ``test_list`` is a list of assert strings; we join them and
    prepend a code-generation instruction prompt.

    Input fields: ``task_id``, ``text``, ``test_list``, ``code``.
    Output fields: ``doc_id``, ``task_id``, ``prompt``, ``answer``.
    """
    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        task_id = _task_id(item, benchmark_name, index)
        description = item.get("text", item.get("prompt"))
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"{benchmark_name} record {index} requires a non-empty string "
                "field 'text' or 'prompt'"
            )
        test_list = item.get("test_list")
        if (
            not isinstance(test_list, list)
            or not test_list
            or any(not isinstance(test, str) or not test.strip() for test in test_list)
        ):
            raise ValueError(
                f"{benchmark_name} record {index} requires a non-empty "
                "'test_list' of non-empty strings"
            )
        tests = "\n".join(test_list)
        prompt = PROMPT_TEMPLATE.format(text=description, tests=tests)
        records.append(
            {
                "doc_id": f"{benchmark_name}:{task_id}",
                "task_id": task_id,
                "prompt": prompt,
                "answer": "\n" + tests,
                "prompt_mode": "mbpp",
            }
        )
    return records


def prepare_mbpp_plus(
    data: list[dict[str, Any]], benchmark_name: str = "mbpp_plus"
) -> list[dict[str, Any]]:
    """Convert MBPP+ using the complete enhanced harness in ``test``.

    EvalPlus's ``test_list`` contains only the original MBPP assertions. The
    ``test`` field carries the augmented inputs, expected results, imports,
    and comparison logic required for MBPP+ evaluation.
    """
    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        task_id = _task_id(item, benchmark_name, index)
        prompt = _required_text(item, "prompt", benchmark_name, index)
        test_harness = _required_text(item, "test", benchmark_name, index)
        records.append(
            {
                "doc_id": f"{benchmark_name}:{task_id}",
                "task_id": task_id,
                "prompt": prompt,
                "answer": "\n" + test_harness.rstrip(),
                "prompt_mode": "mbpp_plus",
            }
        )
    return records


CONVERTERS: dict[str, Callable[[list[dict[str, Any]], str], list[dict[str, Any]]]] = {
    "humaneval": prepare_humaneval,
    "humaneval_plus": prepare_humaneval_plus,
    "mbpp": prepare_mbpp,
    "mbpp_plus": prepare_mbpp_plus,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and format code benchmarks for LLMEval."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mbpp"],
        help="Benchmarks to prepare (default: humaneval mbpp).",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Output directory for JSONL files (default: ./data).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in args.benchmarks:
        if name not in BENCHMARKS:
            print(f"[ERROR] Unknown benchmark: {name}. Available: {list(BENCHMARKS)}")
            failures += 1
            continue

        hf_path, hf_config, hf_split = BENCHMARKS[name]
        print(
            f"[INFO] Downloading {name} from {hf_path} (config={hf_config}, split={hf_split}) ..."
        )

        out_path = output_dir / f"{name}.jsonl"
        try:
            ds = load_dataset(hf_path, hf_config, split=hf_split)
            data: list[dict[str, Any]] = [dict(item) for item in ds]
            records = CONVERTERS[name](data, name)
            with (
                atomic_output_path(out_path) as temporary,
                temporary.open("w", encoding="utf-8") as handle,
            ):
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
        except Exception as exc:
            print(f"[ERROR] Failed to prepare {name}: {exc}")
            failures += 1
            continue

        print(f"[OK]   {name}: {len(records)} items -> {out_path}")

    if failures:
        print(f"[FAIL] {failures} benchmark(s) failed to prepare.")
        return 1
    print("[DONE] All code benchmarks prepared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
