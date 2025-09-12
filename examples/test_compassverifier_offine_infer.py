#!/usr/bin/env python3
"""
Test script for CompassVerifier offline inference.

This script demonstrates how to use the optimized CompassVerifierOfflineInferenceRunner
for an end-to-end offline evaluation workflow:
- parse CLI arguments into `OfflineInferArguments`
- create sample input data
- run inference with `CompassVerifierOfflineInferenceRunner`
- read and display results
- clean up temporary files
"""

import dataclasses
import json
import logging
import os
import sys
from typing import Dict, List

from transformers import HfArgumentParser

from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.logger import init_logger
from llmeval.vllm.compassverifier_offline_infer import \
    CompassVerifierOfflineInferenceRunner

# Initialize logger early so we can report parsing/IO issues consistently.
logger = init_logger('vllm_infer', logging.INFO)


def create_test_data() -> List[Dict[str, str]]:
    """
    Create deterministic test samples for CompassVerifier inference.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - 'question': the problem statement
        - 'gold_answer': the expected correct answer
        - 'llm_response': a model-generated rationale/answer to be judged
        - 'gold_judgment': the expected judgment label (e.g., 'A', 'B')
    """
    test_samples = [{
        'question':
        'Harold tosses a nickel four times. What is the probability that he gets at least as many heads as tails?',
        'gold_answer': '\\frac{11}{16}',
        'llm_response':
        "Alright, I have this probability problem to solve:\n\n**Problem:** Harold tosses a nickel four times. What is the probability that he gets at least as many heads as tails?\n\nFirst, I need to understand what's being asked. Harold is tossing a nickel (which I assume is a fair coin) four times. I need to find the probability that the number of heads is at least as many as the number of tails.\n\n### Understanding the Problem\n\n- **Total number of tosses (n):** 4\n- **Coin:** Fair, so probability of heads (H) = probability of tails (T) = 0.5\n- **Desired outcome:** Number of heads ≥ number of tails\n\nSince there are four tosses, the number of heads can range from 0 to 4. The number of tails will then be 4 minus the number of heads.\n\nSo, the possible number of heads (k) and corresponding number of tails (4 - k) are:\n\n- k = 0: Tails = 4\n- k = 1: Tails = 3\n- k = 2: Tails = 2\n- k = 3: Tails = 1\n- k = 4: Tails = 0\n\nWe need the number of heads to be at least as many as the number of tails, i.e., k ≥ 4 - k.\n\nLet's solve the inequality:\n\nk ≥ 4 - k  \nk + k ≥ 4  \n2k ≥ 4  \nk ≥ 2\n\nSo, the number of heads must be 2, 3, or 4.\n\n### Calculating the Probabilities\n\nTo find the probability of getting at least as many heads as tails, I need to calculate the probabilities for k = 2, 3, and 4 and then sum them up.\n\nThe probability of getting exactly k heads in n tosses of a fair coin is given by the binomial probability formula:\n\nP(k) = C(n, k) * (0.5)^n\n\nWhere:\n- C(n, k) is the number of combinations of n items taken k at a time.\n- (0.5)^n is the probability of any specific sequence of n tosses.\n\nSince the coin is fair, (0.5)^n will be the same for all k, so we can factor that out when summing the probabilities.\n\nBut to simplify, I'll calculate each probability separately.\n\n#### 1. Probability of getting exactly 2 heads (k = 2)\n\nC(4, 2) = 6  \nP(2) = 6 * (0.5)^4 = 6 * 0.0625 = 0.375\n\n#### 2. Probability of getting exactly 3 heads (k = 3)\n\nC(4, 3) = 4  \nP(3) = 4 * (0.5)^4 = 4 * 0.0625 = 0.25\n\n#### 3. Probability of getting exactly 4 heads (k = 4)\n\nC(4, 4) = 1  \nP(4) = 1 * (0.5)^4 = 1 * 0.0625 = 0.0625\n\n### Summing the Probabilities\n\nNow, I'll sum the probabilities for k = 2, 3, and 4:\n\nP(at least as many heads as tails) = P(2) + P(3) + P(4)  \n= 0.375 + 0.25 + 0.0625  \n= 0.6875\n\n### Verifying the Calculation\n\nTo ensure I didn't make a mistake, I'll double-check the calculations:\n\n- **C(4, 2) = 6** ✔️\n- **C(4, 3) = 4** ✔️\n- **C(4, 4) = 1** ✔️\n- **(0.5)^4 = 0.0625** ✔️\n- **0.375 + 0.25 + 0.0625 = 0.6875** ✔️\n\nEverything seems to add up correctly.\n\n### Final Answer\n\nThe probability that Harold gets at least as many heads as tails when tossing a nickel four times is:\n\n\\boxed{0.6875}",
        'gold_judgment': 'A'
    }, {
        'question':
        'Factor completely over the set of polynomials with integer coefficients:\n\\[x^4 - 4x^3 + 14x^2 - 4x + 13.\\]',
        'gold_answer': '(x^2 + 1)(x^2 - 4x + 13)',
        'llm_response':
        "To factor the polynomial $x^4 - 4x^3 + 14x^2 - 4x + 13$ completely over the set of polynomials with integer coefficients, we will follow these steps:\n\n1. **Check for possible rational roots**: Use the Rational Root Theorem, which states that any rational root, expressed as a fraction $\\frac{p}{q}$, must have $p$ as a factor of the constant term (13) and $q$ as a factor of the leading coefficient (1). The possible rational roots are $\\pm1, \\pm13$.\n\n2. **Test the possible rational roots**: Substitute each possible rational root into the polynomial to check if it equals zero.\n\nLet's start by testing $x = 1$:\n$$\n1^4 - 4 \\cdot 1^3 + 14 \\cdot 1^2 - 4 \\cdot 1 + 13 = 1 - 4 + 14 - 4 + 13 = 20 \\neq 0\n$$\nSo, $x = 1$ is not a root.\n\nNext, test $x = -1$:\n$$\n(-1)^4 - 4 \\cdot (-1)^3 + 14 \\cdot (-1)^2 - 4 \\cdot (-1) + 13 = 1 + 4 + 14 + 4 + 13 = 36 \\neq 0\n$$\nSo, $x = -1$ is not a root.\n\nNext, test $x = 13$:\n$$\n13^4 - 4 \\cdot 13^3 + 14 \\cdot 13^2 - 4 \\cdot 13 + 13\n$$\nThis value is extremely large and clearly not zero.\n\nNext, test $x = -13$:\n$$\n(-13)^4 - 4 \\cdot (-13)^3 + 14 \\cdot (-13)^2 - 4 \\cdot (-13) + 13\n$$\nThis value is also extremely large and clearly not zero.\n\nSince none of the possible rational roots are actual roots, we proceed to other methods.\n\n3. **Grouping and factoring by grouping**: This method is not directly applicable here since the polynomial is a quartic and does not factor neatly into quadratic pairs.\n\n4. **Using substitution or known identities**: Sometimes, quartic polynomials can be transformed into quadratic polynomials using substitutions or identities. However, this polynomial does not appear to fit neatly into such categories.\n\n5. **Numerical or graphical methods**: For polynomials that do not factor easily, numerical methods or graphing can be used to approximate roots. However, since we are looking for integer coefficients, we need exact factors.\n\nGiven the complexity and the results from our initial checks, we conclude that the polynomial $x^4 - 4x^3 + 14x^2 - 4x + 13$ does not factor further over the set of polynomials with integer coefficients.\n\nThus, the final answer is:\n$$\n\\boxed{x^4 - 4x^3 + 14x^2 - 4x + 13}\n$$",
        'gold_judgment': 'B'
    }]

    return test_samples


def main() -> None:
    """
    CLI entry point. Parses arguments, creates input data, runs offline inference,
    and prints results.

    Raises:
        SystemExit: on unrecoverable errors (e.g., missing dependencies, parse errors).
    """
    # Parse CLI args into dataclass instance.
    try:
        parser = HfArgumentParser(OfflineInferArguments)
        args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing OfflineInferArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        logger.info(json.dumps(dataclasses.asdict(args), indent=2))
    except ImportError as e:
        logger.error(f'A required library is missing: {e}. Please install it.')
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f'An unrecoverable error occurred during execution: {e}')
        sys.exit(1)

    # Configure IO paths (in working directory) for the demo.
    args.input_path = 'input.json'
    args.output_path = 'output.jsonl'

    try:
        # Create and write test input.
        test_data = create_test_data()
        with open(args.input_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f'Created test input file: {args.input_path}')
        print(f'Output will be saved to: {args.output_path}')

        # Run CompassVerifier inference.
        runner = CompassVerifierOfflineInferenceRunner(args)
        runner.run()

        # Read and display results.
        print('\n' + '=' * 50)
        print('RESULTS:')
        print('=' * 50)
        if not os.path.exists(args.output_path):
            logger.error(f'Output file not found: {args.output_path}')
            return

        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                result = json.loads(line)
                print(f'\nSample {line_num}:')
                print(f"Question: {result.get('question', '')[:100]}...")
                print(f"Gold Answer: {result.get('gold_answer', '')}")
                print(f"Gold Judgment: {result.get('gold_judgment', '')}")
                print(f"Model Judgment: {result.get('judgment', '')}")
                print('-' * 30)
    finally:
        # Clean up temporary files created by this demo script.
        try:
            if os.path.exists(args.input_path):
                os.unlink(args.input_path)
            if os.path.exists(args.output_path):
                os.unlink(args.output_path)
        except Exception as e:
            # Cleanup errors should not crash the script; just log them.
            logger.warning(f'Cleanup failed: {e}')


if __name__ == '__main__':
    main()
