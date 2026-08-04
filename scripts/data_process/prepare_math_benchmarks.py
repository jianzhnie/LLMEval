#!/usr/bin/env python3
"""Prepare math benchmark data files in LLMEval JSONL format.

Downloads from HuggingFace and converts to the unified schema.
Supported math benchmarks: gsm8k, math500, hmmt25, gpqa_diamond, aime24, aime25, aime26.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset

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

def format_example(example: dict, question_col: str, answer_col: str) -> dict:
    question = str(example[question_col]).strip()
    answer = str(example[answer_col]).strip()
    prompt = f"{question}\n{QWEN_MATH_COT_PROMPT}"
    return {"prompt": prompt, "answer": answer}


def prepare_benchmark(name: str, output_dir: Path) -> str:
    """Download and format a single benchmark.  Returns the output file path."""
    if name not in BENCHMARKS:
        print(f"[ERROR] Unknown benchmark: {name}. Available: {list(BENCHMARKS)}")
        sys.exit(1)

    hf_path, hf_config, hf_split, q_col, a_col = BENCHMARKS[name]
    output_file = output_dir / f"{name}.jsonl"

    if output_file.exists():
        print(
            f"[SKIP] {name}: {output_file} already exists ({output_file.stat().st_size} bytes)"
        )
        return str(output_file)

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
        lambda x: format_example(x, q_col, a_col),
        remove_columns=dataset.column_names,
    )
    formatted.to_json(str(output_file), lines=True, force_ascii=False)
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
