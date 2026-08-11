import argparse
from pathlib import Path

from datasets import load_dataset

try:
    from .io_utils import atomic_output_path
except ImportError:  # Direct script execution
    from io_utils import atomic_output_path


def main():
    parser = argparse.ArgumentParser(description="Split a dataset into multiple shards")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a local .jsonl or a glob like '/path/*.jsonl'",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split expression for HF datasets (usually leave None for local files)",
    )
    parser.add_argument(
        "--prefix", default="data_shard", help="Prefix for output shard files"
    )
    parser.add_argument(
        "--num_shards", type=int, default=128, help="Number of shards to create"
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save shards")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num_shards must be at least 1")

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load local JSONL(s)
    if "*" in str(input_path):
        # Handle glob pattern
        dataset = load_dataset(
            "json", data_files={"train": str(input_path)}, split="train"
        )
    else:
        # Handle single file
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        dataset = load_dataset(
            "json", data_files=str(input_path), split=args.split or "train"
        )

    print(f"Loaded dataset with {len(dataset)} examples")

    n = len(dataset)
    num_shards = args.num_shards
    if n == 0:
        parser.error("input dataset must not be empty")
    if num_shards > n:
        parser.error(f"--num_shards ({num_shards}) cannot exceed dataset size ({n})")
    base_size = n // num_shards
    remainder = n % num_shards

    # Create and save shards
    offset = 0
    for i in range(num_shards):
        # Calculate shard size (first 'remainder' shards get one extra example)
        shard_size = base_size + (1 if i < remainder else 0)

        # Extract shard and save
        shard = dataset.select(range(offset, min(offset + shard_size, n)))
        shard_filename = out_dir / f"{args.prefix}_{i:03d}.jsonl"
        with atomic_output_path(shard_filename) as temporary:
            shard.to_json(temporary, lines=True, force_ascii=False)

        offset += shard_size
        print(f"Saved shard {i:03d} with {shard_size} examples to {shard_filename}")

    print(f"Successfully saved {num_shards} shards to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
