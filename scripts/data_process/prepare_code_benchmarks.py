#!/usr/bin/env python3
"""Prepare code benchmark data files in LLMEval JSONL format.

Downloads HumanEval / MBPP / HumanEval+ / MBPP+ from HuggingFace and
converts them to the unified schema: ``{"task_id": ..., "prompt": ...,
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
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, tuple[str, str | None, str]] = {
    "humaneval": ("openai/openai_humaneval", None, "test"),
    "humaneval_plus": ("evalplus/humanevalplus", None, "test"),
    "mbpp": ("google-research-datasets/mbpp", "full", "test"),
    "mbpp_plus": ("evalplus/mbppplus", None, "test"),
}


def prepare_humaneval(data: list[dict]) -> list[dict]:
    """Convert HumanEval-format data to LLMEval JSONL schema.

    Input fields: ``task_id``, ``prompt``, ``test``, ``entry_point``.
    Output fields: ``task_id``, ``prompt``, ``answer`` (test + check()).
    """
    records = []
    for item in data:
        test_with_check = item["test"].rstrip() + f"\ncheck({item['entry_point']})"
        records.append(
            {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "answer": "\n" + test_with_check,
            }
        )
    return records


def prepare_mbpp(data: list[dict]) -> list[dict]:
    """Convert MBPP-format data to LLMEval JSONL schema.

    MBPP ``test_list`` is a list of assert strings; we join them and
    prepend a code-generation instruction prompt.

    Input fields: ``task_id``, ``text``, ``test_list``, ``code``.
    Output fields: ``task_id``, ``prompt``, ``answer``.
    """
    PROMPT_TEMPLATE = (
        "You are an expert Python programmer.  Write a function that "
        "satisfies the following description.\n\n{text}\n\n"
        "Your code should pass these tests:\n\n{tests}\n\n[BEGIN]\n"
    )
    records = []
    for item in data:
        tests = "\n".join(item["test_list"])
        prompt = PROMPT_TEMPLATE.format(text=item["text"], tests=tests)
        records.append(
            {
                "task_id": str(item.get("task_id", "")),
                "prompt": prompt,
                "answer": "\n" + tests,
            }
        )
    return records


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

    for name in args.benchmarks:
        if name not in BENCHMARKS:
            print(f"[ERROR] Unknown benchmark: {name}. Available: {list(BENCHMARKS)}")
            continue

        hf_path, hf_config, hf_split = BENCHMARKS[name]
        print(
            f"[INFO] Downloading {name} from {hf_path} (config={hf_config}, split={hf_split}) ..."
        )

        try:
            ds = load_dataset(hf_path, hf_config, split=hf_split)
        except Exception as exc:
            print(f"[ERROR] Failed to load {name}: {exc}")
            continue

        data = [dict(item) for item in ds]

        if name.startswith("humaneval"):
            records = prepare_humaneval(data)
        elif name.startswith("mbpp"):
            records = prepare_mbpp(data)
        else:
            print(f"[WARN]  No converter for {name}, skipping.")
            continue

        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK]   {name}: {len(records)} items -> {out_path}")

    print("[DONE] All code benchmarks prepared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
