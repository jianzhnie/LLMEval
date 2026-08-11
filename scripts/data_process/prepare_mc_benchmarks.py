#!/usr/bin/env python3
"""Prepare multiple-choice benchmark data files in LLMEval JSONL format.

Downloads from HuggingFace and converts to the unified schema.
Supported MC benchmarks: mmlu, mmlu_pro, ceval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset

try:
    from .io_utils import atomic_output_path
except ImportError:  # Direct script execution
    from io_utils import atomic_output_path

# Multiple-choice benchmarks: need special formatting (choices → prompt)
BENCHMARKS: dict[str, dict] = {
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
    "ceval": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：",
}


def prepare_mc_benchmark(name: str, output_dir: Path) -> str:
    """Download and format a multiple-choice benchmark."""
    if name not in BENCHMARKS:
        print(f"[ERROR] Unknown benchmark: {name}. Available: {list(BENCHMARKS)}")
        sys.exit(1)

    info = BENCHMARKS[name]
    hf_path = info["hf_path"]
    hf_split = info["hf_split"]
    configs = info["configs"]
    output_file = output_dir / f"{name}.jsonl"

    if output_file.exists() and _has_valid_doc_ids(output_file):
        print(f"[SKIP] {name}: already exists")
        return str(output_file)
    if output_file.exists():
        print(f"[REBUILD] {name}: existing file has no valid unique doc_id values")

    all_rows = []
    failed_configs: list[str] = []
    if configs == "all":
        # Load all subject configs
        all_configs = get_dataset_config_names(hf_path)
        subject_configs = [
            c for c in all_configs if c not in ("all", "auxiliary_train")
        ]
        print(f"[LOAD] {name}: {hf_path} ({len(subject_configs)} subjects)")
        for cfg in subject_configs:
            try:
                ds = load_dataset(hf_path, cfg, split=hf_split)
                for index, ex in enumerate(ds):
                    all_rows.append(
                        _format_mc_row(name, ex, source_id=f"{cfg}:{index}")
                    )
            except Exception as e:
                print(f"  [WARN] {cfg}: {e}")
                failed_configs.append(cfg)
    else:
        print(f"[LOAD] {name}: {hf_path}")
        ds = load_dataset(hf_path, split=hf_split)
        for index, ex in enumerate(ds):
            all_rows.append(_format_mc_row(name, ex, source_id=f"{hf_split}:{index}"))

    if failed_configs:
        raise RuntimeError(
            f"Failed to load {len(failed_configs)} config(s) for {name}: "
            + ", ".join(failed_configs)
        )
    if not all_rows:
        raise RuntimeError(f"No rows loaded for {name}; refusing to write cache")

    with (
        atomic_output_path(output_file) as temporary,
        temporary.open("w", encoding="utf-8") as handle,
    ):
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] {name}: {len(all_rows)} examples → {output_file}")
    return str(output_file)


def _format_mc_row(
    name: str, example: dict[str, Any], source_id: str
) -> dict[str, Any]:
    """Format MC example into {prompt, answer, choices, gold}, aligned with lm-eval.

    Returns fields:
        doc_id:  persistent benchmark-scoped question identifier
        prompt:  question text only (for loglikelihood mode)
        answer:  correct answer letter (A/B/C/...)
        choices: list of choice texts (for loglikelihood mode)
        gold:    integer index of correct answer (for loglikelihood mode)
    """

    def choice_labels(count: int) -> list[str]:
        labels: list[str] = []
        for index in range(count):
            value = index
            label = ""
            while True:
                label = chr(ord("A") + value % 26) + label
                value = value // 26 - 1
                if value < 0:
                    break
            labels.append(label)
        return labels

    if name == "mmlu":
        choices = example["choices"]
        if not isinstance(choices, list):
            raise TypeError("mmlu 'choices' must be a list")
        answer_idx = (
            example["answer"]
            if isinstance(example["answer"], int)
            else int(example["answer"])
        )
        letters = choice_labels(len(choices))
        if answer_idx < 0 or answer_idx >= len(choices):
            raise ValueError(f"mmlu answer index out of range: {answer_idx}")
        answer = letters[answer_idx]
        fmt = {"question": example["question"]}
        for i, c in enumerate(choices):
            fmt[letters[i]] = c

    elif name == "mmlu_pro":
        raw_options = example["options"]
        if isinstance(raw_options, list):
            choices = [str(choice).strip() for choice in raw_options]
        elif isinstance(raw_options, str):
            # Retain compatibility with older comma-separated local datasets.
            choices = [choice.strip() for choice in raw_options.split(",")]
        else:
            raise TypeError("mmlu_pro 'options' must be a list or string")
        if not 2 <= len(choices) <= 10:
            raise ValueError(
                f"mmlu_pro must contain between 2 and 10 choices, got {len(choices)}"
            )
        letters = choice_labels(len(choices))
        answer_idx = example.get("answer_index")
        if answer_idx is not None and isinstance(answer_idx, int):
            if answer_idx < 0 or answer_idx >= len(choices):
                raise ValueError(f"mmlu_pro answer index out of range: {answer_idx}")
            answer = letters[answer_idx]
        else:
            raw = example.get("answer", "")
            if not isinstance(raw, str):
                raise ValueError(
                    f"mmlu_pro answer must be a string, got "
                    f"{type(raw).__name__}: {raw!r}"
                )
            if raw in letters:
                answer = raw
            elif raw.isdigit() and 0 <= int(raw) < len(letters):
                answer = letters[int(raw)]
            else:
                raise ValueError(f"mmlu_pro answer label is invalid: {raw!r}")
            answer_idx = letters.index(answer)
        fmt = {"question": example["question"]}
        for i, c in enumerate(choices):
            fmt[letters[i]] = c

    elif name == "ceval":
        choices = [example["A"], example["B"], example["C"], example["D"]]
        letters = choice_labels(len(choices))
        answer = example["answer"].strip().upper()
        if answer not in letters:
            raise ValueError(f"ceval answer label is invalid: {answer!r}")
        answer_idx = letters.index(answer)
        fmt = {
            "question": example["question"],
            "A": example["A"],
            "B": example["B"],
            "C": example["C"],
            "D": example["D"],
        }

    else:
        raise ValueError(f"Unsupported MC benchmark: {name!r}")

    if not choices:
        raise ValueError(f"{name} must contain at least one choice")
    if not isinstance(answer_idx, int) or not 0 <= answer_idx < len(choices):
        raise ValueError(
            f"{name} gold index must be in [0, {len(choices)}), got {answer_idx!r}"
        )

    if name == "mmlu_pro":
        # MMLU-Pro records can contain fewer than ten options; build only the
        # labels present instead of formatting the fixed A-J template.
        option_lines = "\n".join(
            f"{letters[i]}. {choice}" for i, choice in enumerate(choices)
        )
        prompt = f"{example['question']}\n{option_lines}\nAnswer:"
    else:
        template = MC_PROMPT_TEMPLATE[name]
        prompt = template.format(**fmt)
    return {
        "doc_id": f"{name}:{source_id}",
        "prompt": prompt,
        "answer": answer,
        "choices": choices,
        "choice_tokens": letters[: len(choices)],
        "gold": answer_idx,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare multiple-choice benchmark data from HuggingFace"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=[],
        help=f"MC benchmarks to prepare. Available: {list(BENCHMARKS)}",
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
            prepare_mc_benchmark(bm, output_dir)

    if not any(args.benchmarks):
        print("[INFO] No MC benchmarks specified. Use --benchmarks.")

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
