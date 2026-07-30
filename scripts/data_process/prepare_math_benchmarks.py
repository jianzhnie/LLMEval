#!/usr/bin/env python3
"""Prepare benchmark data files in LLMEval JSONL format (prompt + answer).

Downloads from HuggingFace and converts to the unified schema.
Supported benchmarks: gsm8k, math500, hmmt25, gpqa_diamond, hle_full, aime24, aime25.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

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
    "gpqa_diamond": ("lightonai/gpqa_diamond_multilingual", None, "en", "problem", "solution"),
    "aime26": ("math-ai/aime26", "default", "test", "problem", "answer"),
    # HLE-Full: gated dataset, requires HF_TOKEN. Access: cais/hle
    # "hle_full": ("cais/hle", None, "test", "question", "answer"),
    # AA-LCR: not available on HuggingFace Hub.
}

# Multiple-choice benchmarks: need special formatting (choices → prompt)
MC_BENCHMARKS: dict[str, dict] = {
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_split": "test",
        "configs": "all",  # special: iterate all subject configs
    },
    "mmlu_pro": {
        "hf_path": "TIGER-Lab/MMLU-Pro",
        "hf_split": "test",
        "configs": None,  # single config
    },
    "ceval": {
        "hf_path": "ceval/ceval-exam",
        "hf_split": "test",
        "configs": "all",
    },
}

# Aligned with lm-evaluation-harness prompt format
MC_PROMPT_TEMPLATE = {
    "mmlu": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
    "mmlu_pro": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nF. {F}\nG. {G}\nH. {H}\nI. {I}\nJ. {J}\nAnswer:",
    "ceval": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：",
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
        print(f"[SKIP] {name}: {output_file} already exists ({output_file.stat().st_size} bytes)")
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


def prepare_mc_benchmark(name: str, output_dir: Path) -> str:
    """Download and format a multiple-choice benchmark."""
    info = MC_BENCHMARKS[name]
    hf_path = info["hf_path"]
    hf_split = info["hf_split"]
    configs = info["configs"]
    output_file = output_dir / f"{name}.jsonl"

    if output_file.exists():
        print(f"[SKIP] {name}: already exists")
        return str(output_file)

    all_rows = []
    if configs == "all":
        # Load all subject configs
        all_configs = get_dataset_config_names(hf_path)
        subject_configs = [c for c in all_configs if c not in ("all", "auxiliary_train")]
        print(f"[LOAD] {name}: {hf_path} ({len(subject_configs)} subjects)")
        for cfg in subject_configs:
            try:
                ds = load_dataset(hf_path, cfg, split=hf_split)
                for ex in ds:
                    all_rows.append(_format_mc_row(name, ex))
            except Exception as e:
                print(f"  [WARN] {cfg}: {e}")
    else:
        print(f"[LOAD] {name}: {hf_path}")
        ds = load_dataset(hf_path, split=hf_split)
        for ex in ds:
            all_rows.append(_format_mc_row(name, ex))

    import json
    with open(output_file, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] {name}: {len(all_rows)} examples → {output_file}")
    return str(output_file)


def _format_mc_row(name: str, example: dict) -> dict:
    """Format MC example into {prompt, answer, choices, gold}, aligned with lm-eval.

    Returns fields:
        prompt:  question text only (for loglikelihood mode)
        answer:  correct answer letter (A/B/C/...)
        choices: list of choice texts (for loglikelihood mode)
        gold:    integer index of correct answer (for loglikelihood mode)
    """
    template = MC_PROMPT_TEMPLATE[name]
    letters = [chr(ord("A") + i) for i in range(10)]  # A-J

    if name == "mmlu":
        choices = example["choices"]
        answer_idx = example["answer"] if isinstance(example["answer"], int) else int(example["answer"])
        answer = letters[answer_idx]
        fmt = {"question": example["question"]}
        for i, c in enumerate(choices):
            fmt[letters[i]] = c

    elif name == "mmlu_pro":
        choices = [c.strip() for c in example["options"].split(",")]
        answer_idx = example.get("answer_index")
        if answer_idx is not None and isinstance(answer_idx, int):
            answer = letters[answer_idx]
        else:
            raw = example.get("answer", "")
            answer = raw if raw in letters else letters[int(raw)] if raw.isdigit() else raw
            answer_idx = letters.index(answer) if answer in letters else -1
        fmt = {"question": example["question"]}
        for i, c in enumerate(choices):
            fmt[letters[i]] = c

    elif name == "ceval":
        choices = [example["A"], example["B"], example["C"], example["D"]]
        answer = example["answer"].strip().upper()
        answer_idx = letters.index(answer) if answer in letters else -1
        fmt = {
            "question": example["question"],
            "A": example["A"], "B": example["B"],
            "C": example["C"], "D": example["D"],
        }

    else:
        return {"prompt": str(example), "answer": "", "choices": [], "gold": -1}

    prompt = template.format(**fmt)
    return {
        "prompt": prompt,
        "answer": answer,
        "choices": choices,
        "gold": answer_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare benchmark data from HuggingFace or local files"
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=[],
        help=f"Math benchmarks to prepare. Available: {list(BENCHMARKS)}"
    )
    parser.add_argument(
        "--mc_benchmarks", nargs="+", default=[],
        help=f"MC benchmarks to prepare. Available: {list(MC_BENCHMARKS)}"
    )
    parser.add_argument(
        "--output_dir", default="./data",
        help="Output directory for JSONL files (default: ./data)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bm in args.benchmarks:
        if bm:  # skip empty strings
            prepare_benchmark(bm, output_dir)

    for bm in args.mc_benchmarks:
        if bm:
            prepare_mc_benchmark(bm, output_dir)

    if not any(args.benchmarks) and not any(args.mc_benchmarks):
        print("[INFO] No benchmarks specified. Use --benchmarks for math, --mc_benchmarks for MC.")

    print("\n[DONE] All benchmarks prepared.")

    # Print summary
    print("\n--- Data files ---")
    import os
    for f in sorted(output_dir.glob("*.jsonl")):
        lines = 0
        with open(f) as fh:
            for _ in fh:
                lines += 1
        print(f"  {f.name}: {lines} examples ({os.path.getsize(f)} bytes)")


if __name__ == "__main__":
    main()
