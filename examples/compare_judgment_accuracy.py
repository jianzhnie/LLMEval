#!/usr/bin/env python3
"""
Compare model-reported accuracy against a human-provided judgment for a JSONL dataset.

The script loads a JSONL dataset using the Hugging Face `datasets` library and
compares a model's reported accuracy (`accuracy` field) against a human judgment
(`compassverifier_judgment` field) for each record. It then generates a diff
file containing only the records where the two assessments disagree and prints a
summary of the evaluation statistics.

Rules for comparison:
- Human Judgment (`compassverifier_judgment`):
  - 'A' is treated as correct (1).
  - 'B', 'C', or any other value is treated as incorrect (0).
- Model Accuracy (`accuracy`):
  - The value is converted to a float (non-numeric values default to 0.0).
  - The result is binarized to 1 if `accuracy` > `threshold`, and 0 otherwise.

Outputs:
- A `_diff.jsonl` file containing all records where the model's accuracy
  assessment differs from the human judgment.
- A summary of match/mismatch counts and a detailed breakdown printed to standard
  output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, IterableDataset, load_dataset

# -----------------------------
# Constants
# -----------------------------

DEFAULT_THRESHOLD: float = 0.5
DEFAULT_SPLIT: str = 'train'
CORRECT_JUDGMENT: str = 'A'
TEMP_FIELD_PREFIX: str = '__'
ACCURACY_FIELD: str = 'accuracy'
JUDGMENT_FIELD: str = 'compassverifier_judgment'

# -----------------------------
# Data structures and utilities
# -----------------------------


@dataclass
class EvalStats:
    """
    Data structure to hold and aggregate evaluation statistics.

    Attributes:
        total: The total number of records processed.
        same: The number of records where model accuracy matches human judgment.
        different: The number of records where model accuracy mismatches human judgment.
        breakdown: A detailed breakdown of mismatches by judgment type.
    """
    total: int = 0
    same: int = 0
    different: int = 0
    breakdown: Counter[str] = field(default_factory=Counter)

    def match_rate(self) -> float:
        """Calculates the percentage of matching assessments."""
        return (self.same / self.total) if self.total else 0.0

    def mismatch_rate(self) -> float:
        """Calculates the percentage of mismatching assessments."""
        return (self.different / self.total) if self.total else 0.0

    def add_record(self, model_acc: int, judgment_acc: int,
                   judgment_str: str) -> None:
        """
        Add a single record to the statistics.

        Args:
            model_acc: The binarized model accuracy (0 or 1).
            judgment_acc: The binarized human judgment (0 or 1).
            judgment_str: The string representation of the judgment for breakdown.
        """
        self.total += 1
        if model_acc == judgment_acc:
            self.same += 1
        else:
            self.different += 1

        breakdown_key = f'acc{model_acc}_judg{judgment_str}'
        self.breakdown[breakdown_key] += 1


def normalize_accuracy(value: Any, threshold: float) -> int:
    """
    Binarizes a model's accuracy value based on a given threshold.

    Args:
        value: The accuracy value, which may be a string, float, or other type.
        threshold: The float threshold for binarization.

    Returns:
        1 if the numeric value of `accuracy` is greater than the threshold,
        otherwise 0. Non-numeric values are treated as 0.0.
    """
    try:
        acc_val = float(value)
    except (ValueError, TypeError):
        acc_val = 0.0
    return 1 if acc_val > threshold else 0


def judgment_to_accuracy(judgment: Any) -> int:
    """
    Maps a human judgment string to a binary accuracy value.

    Args:
        judgment: The human judgment value, e.g., 'A', 'B', 'C'.

    Returns:
        1 if the judgment is 'A' (case-insensitive), otherwise 0.
    """
    if not judgment:
        return 0
    return 1 if str(judgment).strip().upper() == CORRECT_JUDGMENT else 0


def get_judgment_string(judgment: Any) -> str:
    """
    Converts a judgment value to a standardized string for breakdown keys.

    Args:
        judgment: The judgment value to convert.

    Returns:
        A standardized string representation of the judgment.
    """
    if not judgment:
        return 'OTHER'

    judgment_str = str(judgment).strip().upper()
    if judgment_str in ['A', 'B', 'C']:
        return judgment_str
    return 'OTHER'


def pretty_print_stats(stats: EvalStats) -> None:
    """
    Prints a formatted summary of the evaluation statistics.

    Args:
        stats: An `EvalStats` object containing the evaluation results.
    """
    print('--- Summary ---')
    print(f'Total records evaluated: {stats.total}')
    print(f'Matching assessments: {stats.same} ({stats.match_rate():.2%})')
    print(
        f'Mismatching assessments: {stats.different} ({stats.mismatch_rate():.2%})'
    )

    print('\nDetailed Breakdown:')
    # Define a consistent order for printing the breakdown keys
    ordered_keys = [
        'acc1_judgA',
        'acc1_judgB',
        'acc1_judgC',
        'acc1_judgOTHER',
        'acc0_judgA',
        'acc0_judgB',
        'acc0_judgC',
        'acc0_judgOTHER',
    ]

    found_keys = False
    for key in ordered_keys:
        if stats.breakdown.get(key, 0):
            print(f'  - {key:<15}: {stats.breakdown[key]}')
            found_keys = True

    if not found_keys:
        print('  No detailed breakdown data available.')


# -----------------------------
# Argument parsing and path resolution
# -----------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses command-line arguments for the evaluation script.

    Args:
        argv: Optional list of command line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=
        'Compare model accuracy to human judgment in a JSONL dataset.')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_path',
        type=str,
        help='Path to a single JSONL file (e.g., ./data/test.jsonl).')
    input_group.add_argument(
        '--data_files',
        type=str,
        help='Local file pattern for the HF loader (e.g., "data/*.jsonl"). '
        'Overrides `--input_path` if both are provided.')

    parser.add_argument(
        '--split',
        type=str,
        default=DEFAULT_SPLIT,
        help=f'Dataset split to load (default: {DEFAULT_SPLIT}).')
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=
        f'Threshold for binarizing model accuracy (default: {DEFAULT_THRESHOLD}).'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=None,
        help=
        'Number of processes for HF map/filter. Defaults to os.cpu_count().')
    parser.add_argument('--output_path',
                        type=str,
                        help='Output path for the mismatch JSONL file. '
                        'Defaults to <input_path>_diff.jsonl.')

    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Resolves effective input and output paths based on parsed arguments.

    Args:
        args: The argparse namespace object.

    Returns:
        A tuple containing the effective `data_files` path and the `output_path`.

    Raises:
        ValueError: If neither --data_files nor --input_path is provided.
    """
    effective_data_files = args.data_files or args.input_path
    if not effective_data_files:
        raise ValueError(
            'Either --data_files or --input_path must be provided.')

    if args.output_path:
        effective_output_path = args.output_path
    else:
        # Extract the first file from the pattern for default output naming
        first_file = effective_data_files.split('.')[0].strip()
        effective_output_path = f'{first_file}_diff.jsonl'

    return effective_data_files, effective_output_path


# -----------------------------
# Core evaluation pipeline
# -----------------------------


def process_record(record: Dict[str, Any],
                   threshold: float) -> Tuple[Dict[str, Any], int, int, str]:
    """
    Process a single record and compute accuracy metrics.

    Args:
        record: The input record dictionary.
        threshold: The threshold for binarizing model accuracy.

    Returns:
        A tuple containing:
        - The processed record with computed fields
        - The binarized model accuracy
        - The binarized human judgment
        - The judgment string for breakdown
    """
    model_acc = normalize_accuracy(record.get(ACCURACY_FIELD), threshold)
    judgment_acc = judgment_to_accuracy(record.get(JUDGMENT_FIELD))
    judgment_str = get_judgment_string(record.get(JUDGMENT_FIELD))

    # Create a copy of the record with computed fields
    processed_record = record.copy()
    processed_record[f'{TEMP_FIELD_PREFIX}model_acc'] = model_acc
    processed_record[f'{TEMP_FIELD_PREFIX}judgment_acc'] = judgment_acc
    processed_record[f'{TEMP_FIELD_PREFIX}mismatch'] = (model_acc !=
                                                        judgment_acc)

    return processed_record, model_acc, judgment_acc, judgment_str


def write_mismatches_to_file(dataset: Union[Dataset,
                                            IterableDataset], output_path: str,
                             num_proc: Optional[int]) -> None:
    """
    Write mismatching records to the output file.

    Args:
        dataset: The dataset to filter for mismatches.
        output_path: The output file path.
        num_proc: Number of processes for parallel operations.
    """

    def is_mismatch(record: Dict[str, Any]) -> bool:
        """Filter predicate for mismatches."""
        return record.get(f'{TEMP_FIELD_PREFIX}mismatch', False)

    def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Remove temporary fields from the record."""
        return {
            k: v
            for k, v in record.items() if not k.startswith(TEMP_FIELD_PREFIX)
        }

    def clean_data(record: Dict[str, Any]) -> Dict[str, Any]:
        """Remove temporary fields from the record."""
        return {
            k: v
            for k, v in record.items()
            if k not in ['prompt', 'gen', 'task', 'timeout_cnt']
        }

    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            mismatch_dataset = dataset.filter(is_mismatch, num_proc=num_proc)

            for record in mismatch_dataset:
                clean_record_dict = clean_record(record)
                clean_record_dict = clean_data(record)
                output_file.write(
                    json.dumps(clean_record_dict, ensure_ascii=False) + '\n')
    except IOError as e:
        raise IOError(
            f'Failed to write mismatches to {output_path}: {e}') from e


def process_and_evaluate(
    dataset: Union[Dataset, IterableDataset],
    threshold: float,
    output_path: str,
    num_proc: Optional[int],
) -> EvalStats:
    """
    Processes the dataset, identifies mismatches, writes them to a file,
    and returns aggregate statistics.

    This function leverages the Hugging Face `datasets` library for efficient
    parallel processing of the data.

    Args:
        dataset: The Hugging Face `Dataset` or `IterableDataset` to process.
        threshold: The threshold for binarizing model accuracy.
        output_path: The file path to write mismatching records to.
        num_proc: The number of processes for parallel operations.

    Returns:
        An `EvalStats` object with the aggregated results.
    """
    stats = EvalStats()
    effective_proc = num_proc if num_proc is not None else os.cpu_count()

    # Process records and collect statistics
    if effective_proc == 1 or isinstance(dataset, IterableDataset):
        # Single-threaded processing
        processed_records = []
        for record in dataset:
            processed_record, model_acc, judgment_acc, judgment_str = process_record(
                record, threshold)
            stats.add_record(model_acc, judgment_acc, judgment_str)
            processed_records.append(processed_record)

        # Create a new dataset with processed records for filtering
        if isinstance(dataset, Dataset):
            processed_dataset = Dataset.from_list(processed_records)
        else:
            processed_dataset = dataset  # For IterableDataset, we'll handle differently
    else:
        # Parallel processing
        def map_function(record: Dict[str, Any]) -> Dict[str, Any]:
            """Map function for parallel processing."""
            processed_record, _, _, _ = process_record(record, threshold)
            return processed_record

        processed_dataset = dataset.map(
            map_function,
            num_proc=effective_proc,
            desc='Computing accuracy and judgment',
        )

        # Collect statistics from the processed dataset
        for record in processed_dataset:
            model_acc = record.get(f'{TEMP_FIELD_PREFIX}model_acc', 0)
            judgment_acc = record.get(f'{TEMP_FIELD_PREFIX}judgment_acc', 0)
            judgment_str = get_judgment_string(record.get(JUDGMENT_FIELD))
            stats.add_record(model_acc, judgment_acc, judgment_str)

    # Write mismatches to file
    write_mismatches_to_file(processed_dataset, output_path, effective_proc)

    return stats


# -----------------------------
# Main entry point
# -----------------------------


def main() -> None:
    """Main function to orchestrate the entire process."""
    try:
        args = parse_args(sys.argv[1:])
        data_files, output_path = resolve_paths(args)

        print(f'Loading data from: {data_files}')
        dataset = load_dataset('json', data_files=data_files, split=args.split)

        print(f'Starting evaluation with threshold: {args.threshold}')
        stats = process_and_evaluate(
            dataset=dataset,
            threshold=args.threshold,
            output_path=output_path,
            num_proc=args.num_proc,
        )

        pretty_print_stats(stats)
        print(f'\nMismatching records saved to: {output_path}')

    except KeyboardInterrupt:
        print('\nOperation cancelled by user.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'An error occurred: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
