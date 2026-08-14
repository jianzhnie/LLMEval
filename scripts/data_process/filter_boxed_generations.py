"""Filter boxed generations by their actual chat-template token length."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from llmeval.inference.common import build_chat_messages
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY

try:
    from .io_utils import atomic_output_path
except ImportError:  # Direct script execution
    from io_utils import atomic_output_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_single_text(field: Any, field_name: str) -> str:
    """Return one text value while rejecting ambiguous or malformed fields."""
    if isinstance(field, str):
        return field
    if field is None:
        return ""
    if isinstance(field, list):
        if not field:
            return ""
        if len(field) > 1:
            raise ValueError(
                f"{field_name!r} contains {len(field)} generations; expected exactly one"
            )
        if isinstance(field[0], str):
            return field[0]
    raise ValueError(f"{field_name!r} must be a string or a list containing one string")


def compute_token_lengths(
    example: dict[str, Any], tokenizer: AutoTokenizer, system_prompt: str | None
) -> dict[str, Any]:
    """Compute lengths matching the tokens used for chat inference."""
    prompt_text = extract_single_text(example.get("prompt"), "prompt")
    gen_text = extract_single_text(example.get("gen"), "gen")
    messages = build_chat_messages(prompt_text, system_prompt)
    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    gen_tokens = tokenizer(
        gen_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )

    prompt_length = len(prompt_token_ids)
    gen_length = len(gen_tokens["input_ids"])

    tail_text = gen_text[-1000:]
    has_boxed = "boxed{" in tail_text

    return {
        "prompt_length": prompt_length,
        "gen_length": gen_length,
        "total_token_length": prompt_length + gen_length,
        "has_boxed": has_boxed,
    }


def should_keep_example(example: dict[str, Any], max_token_length: int) -> bool:
    """Determine if example should be kept based on filtering criteria."""
    exceeded = example["total_token_length"] > max_token_length
    return example["has_boxed"] and not exceeded


def load_and_validate_args() -> argparse.Namespace:
    """Load and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter dataset by combined prompt and generation token length, "
            "then remove the generation field."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to a local .jsonl or glob pattern (e.g., '/data/*.jsonl')",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL file path (e.g., './filtered.jsonl')",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        required=True,
        help="Tokenizer name or path (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=128,
        help="Maximum allowed combined token length of prompt and gen fields",
    )
    parser.add_argument(
        "--num_proc", type=int, default=16, help="Number of processes for filtering"
    )
    parser.add_argument(
        "--system_prompt_type",
        choices=sorted(SYSTEM_PROMPT_FACTORY),
        default="amthinking",
        help="System prompt used by inference when applying the chat template",
    )

    args = parser.parse_args()
    if args.max_token_length <= 0:
        parser.error("--max_token_length must be positive")
    if args.num_proc <= 0:
        parser.error("--num_proc must be positive")
    output_path = Path(args.output_file).resolve()
    matched_inputs = {Path(path).resolve() for path in glob.glob(args.input_path)}
    if output_path in matched_inputs or (
        not any(marker in args.input_path for marker in "*?[]")
        and Path(args.input_path).resolve() == output_path
    ):
        parser.error("--output_file must not be included in --input_path")
    return args


def process_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_token_length: int,
    num_proc: int,
    system_prompt: str | None,
) -> Dataset:
    """Process dataset with token length computation and filtering."""
    logger.info("Computing token lengths...")

    dataset = dataset.map(
        lambda x: compute_token_lengths(x, tokenizer, system_prompt),
        num_proc=num_proc,
        desc="Computing token lengths",
    )

    initial_count = len(dataset)
    logger.info(f"Initial dataset size: {initial_count}")

    dataset = dataset.filter(
        lambda x: should_keep_example(x, max_token_length),
        num_proc=num_proc,
        desc="Filtering by criteria",
    )

    filtered_count = len(dataset)
    logger.info(f"Filtered dataset size: {filtered_count}")
    removal_rate = (
        (initial_count - filtered_count) / initial_count * 100 if initial_count else 0.0
    )
    logger.info(
        f"Removed {initial_count - filtered_count} examples ({removal_rate:.1f}%)"
    )

    if "gen" in dataset.column_names:
        dataset = dataset.remove_columns(["gen"])
        logger.info("Removed 'gen' field")

    return dataset


def main() -> None:
    """Main processing function."""
    try:
        args = load_and_validate_args()

        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading tokenizer from: %s", args.tokenizer_name_or_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        except Exception as exc:
            logger.error("Failed to load tokenizer: %s", exc)
            raise

        logger.info("Loading dataset from: %s", args.input_path)
        try:
            dataset = load_dataset(
                "json", data_files=str(args.input_path), split="train"
            )
        except Exception as exc:
            logger.error("Failed to load dataset: %s", exc)
            raise

        logger.info("Loaded %d examples", len(dataset))

        processed_dataset = process_dataset(
            dataset,
            tokenizer,
            args.max_token_length,
            args.num_proc,
            SYSTEM_PROMPT_FACTORY[args.system_prompt_type],
        )

        logger.info("Saving filtered dataset to: %s", output_file)
        with atomic_output_path(output_file) as temporary:
            processed_dataset.to_json(str(temporary), lines=True, force_ascii=False)
        logger.info("Processing completed successfully")

    except Exception as exc:
        logger.error("Processing failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
