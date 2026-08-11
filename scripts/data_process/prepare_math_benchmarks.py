#!/usr/bin/env python3
"""Prepare math benchmark data files in LLMEval JSONL format.

Downloads from HuggingFace and converts to the unified schema.
Supported math benchmarks: gsm8k, math500, hmmt25, gpqa_diamond, aime24, aime25, aime26.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from .io_utils import atomic_output_path
except ImportError:  # Direct script execution
    from io_utils import atomic_output_path

QWEN_MATH_COT_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# ---------------------------------------------------------------------------
# Benchmark definitions: {name: (hf_path, hf_config, hf_split, question_col, answer_col)}
# For datasets where config == split, set hf_config == hf_split.
# ---------------------------------------------------------------------------
BENCHMARKS: dict[str, tuple[str, str | None, str, str, str]] = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
    "hmmt25": ("MathArena/hmmt_feb_2025", None, "train", "problem", "answer"),
    "gpqa_diamond": (
        "lightonai/gpqa_diamond_multilingual",
        None,
        "en",
        "problem",
        "solution",
    ),
    "aime24": ("math-ai/aime24", "default", "test", "problem", "solution"),
    "aime25": ("math-ai/aime25", "default", "test", "problem", "answer"),
    "aime26": ("math-ai/aime26", "default", "test", "problem", "answer"),
    # HLE-Full: gated dataset, requires HF_TOKEN. Access: cais/hle
    # "hle_full": ("cais/hle", None, "test", "question", "answer"),
    # AA-LCR: not available on HuggingFace Hub.
}


def _make_doc_id(name: str, example: dict[str, Any], index: int) -> str:
    """Build a persistent benchmark-scoped question ID during preparation."""
    source_id = example.get("id", example.get("task_id", index))
    return f"{name}:{source_id}"


def _has_valid_doc_ids(path: Path) -> bool:
    """Return whether an existing JSONL file has a unique ID on every row."""
    ids: set[str] = set()
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                document_id = item.get("doc_id") if isinstance(item, dict) else None
                if not document_id or str(document_id) in ids:
                    return False
                ids.add(str(document_id))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(ids)


def format_example(
    example: dict[str, Any],
    index: int,
    name: str,
    question_col: str,
    answer_col: str,
) -> dict[str, Any]:
    """Convert one source row to the math inference schema."""
    question = str(example[question_col]).strip()
    answer = str(example[answer_col]).strip()
    prompt = f"{question}\n{QWEN_MATH_COT_PROMPT}"
    return {
        "doc_id": _make_doc_id(name, example, index),
        "prompt": prompt,
        "answer": answer,
    }


def prepare_benchmark(name: str, output_dir: Path) -> str:
    """Download and format a single benchmark.  Returns the output file path."""
    if name not in BENCHMARKS:
        print(f"[ERROR] Unknown benchmark: {name}. Available: {list(BENCHMARKS)}")
        sys.exit(1)

    hf_path, hf_config, hf_split, q_col, a_col = BENCHMARKS[name]
    output_file = output_dir / f"{name}.jsonl"

    if output_file.exists() and _has_valid_doc_ids(output_file):
        print(
            f"[SKIP] {name}: {output_file} already exists ({output_file.stat().st_size} bytes)"
        )
        return str(output_file)
    if output_file.exists():
        print(f"[REBUILD] {name}: existing file has no valid unique doc_id values")

    print(f"[LOAD] {name}: {hf_path} (config={hf_config}, split={hf_split})")
    try:
        kwargs = {"path": hf_path, "split": hf_split}
        if hf_config:
            kwargs["name"] = hf_config
        dataset = load_dataset(**kwargs)
    except Exception as e:
        print(f"[ERROR] {name}: failed to load — {e}")
        sys.exit(1)

    formatted = dataset.map(
        lambda example, index: format_example(example, index, name, q_col, a_col),
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    with atomic_output_path(output_file) as temporary:
        formatted.to_json(str(temporary), lines=True, force_ascii=False)
    print(f"[DONE] {name}: {len(formatted)} examples → {output_file}")
    return str(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare math benchmark data from HuggingFace"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=[],
        help=f"Math benchmarks to prepare. Available: {list(BENCHMARKS)}",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Output directory for JSONL files (default: ./data)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bm in args.benchmarks:
        if bm:  # skip empty strings
            prepare_benchmark(bm, output_dir)

    if not any(args.benchmarks):
        print("[INFO] No math benchmarks specified. Use --benchmarks.")

    print("\n[DONE] All benchmarks prepared.")

    # Print summary
    print("\n--- Data files ---")
    for f in sorted(output_dir.glob("*.jsonl")):
        lines = 0
        with open(f) as fh:
            for _ in fh:
                lines += 1
        print(f"  {f.name}: {lines} examples ({os.path.getsize(f)} bytes)")


if __name__ == "__main__":
    main()
