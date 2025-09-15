#!/usr/bin/env python3
"""
Compare model-reported accuracy against a rule derived from `compassverifier_judgment`
for each line in a JSONL file.

Rule:
- If judgment == "A" → treated as correct (1)
- If judgment in {"B", "C"} or anything else → treated as incorrect (0)

For the model-reported accuracy, values are normalized to float and then binarized:
- acc_bin = 1 if acc_val > threshold else 0

A diff file is written containing only those JSON lines where `acc_bin != rule_acc`.
By default, the diff file path is `<input>_diff.jsonl`, but it can be overridden.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass
class EvalStats:
    """Aggregation of evaluation statistics."""
    total: int = 0
    same: int = 0
    different: int = 0
    breakdown: Counter = None  # Counter[str, int]

    def __post_init__(self) -> None:
        if self.breakdown is None:
            self.breakdown = Counter()

    def match_rate(self) -> float:
        return (self.same / self.total) if self.total else 0.0

    def mismatch_rate(self) -> float:
        return (self.different / self.total) if self.total else 0.0


def normalize_accuracy(value: Any, threshold: float) -> int:
    """
    Convert accuracy-like value to a binary 0/1 using the provided threshold.

    - Non-numeric values are treated as 0.0.
    - Binarization uses: 1 if acc_val > threshold else 0.

    Returns:
        0 or 1
    """
    try:
        acc_val = float(value)
    except Exception:
        acc_val = 0.0
    return 1 if acc_val > threshold else 0


def rule_accuracy_from_judgment(judgment: Any) -> int:
    """
    Map `compassverifier_judgment` to a binary accuracy according to the rule.

    - "A" -> 1
    - "B", "C", anything else -> 0
    """
    label = str(judgment).strip().upper()
    return 1 if label == 'A' else 0


def evaluate_stream(
    lines: Iterable[str],
    threshold: float,
) -> Tuple[EvalStats, list[str]]:
    """
    Evaluate an iterable of JSONL lines.

    Args:
        lines: Iterable of raw JSON text lines.
        threshold: Threshold used to binarize accuracy.

    Returns:
        (stats, mismatches)
        - stats: aggregated `EvalStats`
        - mismatches: list of raw JSON lines where acc_bin != rule_acc
    """
    stats = EvalStats()
    mismatches: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        try:
            obj: Dict[str, Any] = json.loads(line)
        except Exception:
            stats.breakdown['invalid_lines'] += 1
            continue

        if 'accuracy' not in obj or 'compassverifier_judgment' not in obj:
            stats.breakdown['invalid_lines'] += 1
            continue

        acc_bin = normalize_accuracy(obj.get('accuracy'), threshold)
        rule_acc = rule_accuracy_from_judgment(
            obj.get('compassverifier_judgment'))

        stats.total += 1
        if acc_bin == rule_acc:
            stats.same += 1
        else:
            mismatches.append(line)
            stats.different += 1

        # Detailed breakdown
        judgment_label = str(
            obj.get('compassverifier_judgment')).strip().upper()
        if acc_bin == 1:
            if judgment_label == 'A':
                stats.breakdown['acc1_judgA'] += 1
            elif judgment_label == 'B':
                stats.breakdown['acc1_judgB'] += 1
            elif judgment_label == 'C':
                stats.breakdown['acc1_judgC'] += 1
            else:
                stats.breakdown['acc1_judgOther'] += 1
        else:
            if judgment_label == 'A':
                stats.breakdown['acc0_judgA'] += 1
            elif judgment_label == 'B':
                stats.breakdown['acc0_judgB'] += 1
            elif judgment_label == 'C':
                stats.breakdown['acc0_judgC'] += 1
            else:
                stats.breakdown['acc0_judgOther'] += 1

    return stats, mismatches


def pretty_print_stats(stats: EvalStats) -> None:
    """Print summary statistics and breakdown to stdout."""
    print('Total evaluated lines:', stats.total)
    print('Same (accuracy matches rule):', stats.same)
    print('Different (accuracy mismatches rule):', stats.different)
    if stats.total > 0:
        print('Match rate: {:.2%}'.format(stats.match_rate()))
        print('Mismatch rate: {:.2%}'.format(stats.mismatch_rate()))

    print('\nSummary:\n')
    keys = [
        'acc1_judgA',
        'acc1_judgB',
        'acc1_judgC',
        'acc1_judgOther',
        'acc0_judgA',
        'acc0_judgB',
        'acc0_judgC',
        'acc0_judgOther',
        'invalid_lines',
    ]
    for k in keys:
        if stats.breakdown.get(k, 0):
            print(f'  {k}: {stats.breakdown[k]}')


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=
        'Compare compassverifier_judgment with rule-based accuracy in a JSONL file.'
    )
    parser.add_argument(
        '--input_path',
        help='Path to JSONL file (e.g., ./output/test.jsonl)',
    )
    parser.add_argument(
        '--output_path',
        default=None,
        help='Output JSONL path for mismatches. Default: <input>_diff.jsonl',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help=
        'Threshold to treat accuracy as correct (acc_bin = 1 if accuracy > threshold; default: 0.5)',
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point: parse args, process file, write mismatches, print stats."""
    args = parse_args()

    # Determine output path (fixes previous None handling)
    output_path = args.output_path or f'{args.input_path}_diff.jsonl'

    try:
        with open(args.input_path, 'r', encoding='utf-8') as fin:
            stats, mismatches = evaluate_stream(fin, threshold=args.threshold)
    except FileNotFoundError:
        raise SystemExit(f'Input file not found: {args.input_path}')
    except OSError as e:
        raise SystemExit(f'Failed to read input file: {args.input_path} ({e})')

    # Write mismatches to the diff file
    try:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for m in mismatches:
                fout.write(m + '\n')
    except OSError as e:
        raise SystemExit(f'Failed to write output file: {output_path} ({e})')

    # Print stats
    pretty_print_stats(stats)


if __name__ == '__main__':
    main()
