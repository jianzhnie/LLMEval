"""
Main evaluation orchestrator — dispatches to math / code / mc scorers

This script provides functionality to evaluate language model outputs for various tasks.
It processes input data in JSONL format and computes performance metrics based on the
specified task type. Supported task families:

    - ``math_opensource``  — math answer verification via math-verify
    - ``mc_opensource``    — multiple-choice, loglikelihood or generation based
    - ``code_opensource``  — code generation, sandboxed pass@1 execution

Features:
    - JSONL input file processing
    - Flexible task-specific evaluation
    - Caching support for efficiency
    - Parallel processing capabilities
    - Robust error handling

Example:
    $ python llmeval/evaluator.py --input_path data.jsonl --task_name math_opensource/aime24 --cache_path cache/

Author: jianzhnie
Date: 2025
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from transformers import HfArgumentParser

from llmeval.tasks.code_eval.code_score import score_code
from llmeval.tasks.math_eval.math_score import compute_scores
from llmeval.tasks.mc_eval.mc_score import (
    score_generate,
    score_loglikelihood,
)
from llmeval.tasks.postprocess import (
    apply_text_pipeline,
    build_text_pipeline,
    strip_reasoning_wrappers,
)
from llmeval.tasks.provenance import (
    annotate_dataset_contamination,
    load_contamination_sources,
)
from llmeval.utils.config import EvalTaskArguments
from llmeval.utils.log import init_logger

# Initialize logger for the evaluation orchestrator
logger = init_logger("evaluator")

MATH_RESPONSE_PIPELINE = build_text_pipeline(strip_reasoning_wrappers)


def _get_after_think(text: str) -> str:
    """Compatibility wrapper for the shared reasoning-text filter."""
    return apply_text_pipeline(text, MATH_RESPONSE_PIPELINE)


def preprocess_answers(data: list[dict[str, Any]], response_key: str) -> None:
    """Strip think tags from all generated responses before scoring.

    Args:
        data: List of data items with generated responses.
        response_key: The dictionary key for model responses.

    Returns:
        The modified list (in-place) with cleaned responses.
    """
    for item in data:
        gen = item.get(response_key, [])
        if isinstance(gen, list):
            item[response_key] = [
                apply_text_pipeline(g, MATH_RESPONSE_PIPELINE) for g in gen
            ]
        elif isinstance(gen, str):
            item[response_key] = apply_text_pipeline(gen, MATH_RESPONSE_PIPELINE)
    return None  # mutates in-place; callers should not rely on return value


def _infer_mc_mode(eval_dataset: list[dict[str, Any]]) -> str:
    """Infer the MC scoring mode from the dataset shape.

    Returns:
        ``"loglikelihood"`` when every item carries ``logprobs``.
        ``"generate"`` when no item carries ``logprobs``.

    Raises:
        ValueError: If the dataset mixes both shapes.  Mixed inputs are a
            schema error, not a valid evaluation batch.
    """
    has_logprobs = ["logprobs" in item for item in eval_dataset]
    if all(has_logprobs):
        return "loglikelihood"
    if not any(has_logprobs):
        return "generate"
    raise ValueError(
        "Mixed MC dataset detected: some items have 'logprobs' and others do not. "
        "Please evaluate a single inference schema per batch."
    )


def _process_item(
    item: dict[str, Any],
    task_name: str,
    label_key: str = "answer",
    response_key: str = "gen",
) -> dict[str, Any]:
    """
    Process and validate a single data item from the input dataset.

    This function performs validation checks on the input dictionary to ensure it contains
    the required keys for evaluation. It creates a copy of the input item to avoid
    modifying the original data and adds the task name for reference.

    Args:
        item: A dictionary containing the evaluation data with the following expected keys:
            - label_key: Contains the ground truth answer
            - response_key: Contains the model's generated response
        task_name: The identifier for the evaluation task (e.g., 'math_opensource')
        label_key: The dictionary key used to access the ground truth answer (default: 'answer')
        response_key: The dictionary key used to access the model's response (default: 'gen')

    Returns:
        Dict[str, Any]: A new dictionary containing the validated data with added task field

    Raises:
        ValueError: If either the label_key or response_key is missing from the input item
        TypeError: If the item argument is not a dictionary
    """
    if not isinstance(item, dict):
        raise TypeError(f"Expected dictionary input, got {type(item).__name__}")

    # Validate required keys with detailed error messages
    if label_key not in item:
        raise ValueError(
            f"Missing ground truth label key '{label_key}' in item. Available keys: {', '.join(item.keys())}"
        )
    if response_key not in item:
        raise ValueError(
            f"Missing model response key '{response_key}' in item. Available keys: {', '.join(item.keys())}"
        )

    # Create a new copy to avoid modifying the original dictionary
    processed_item = item.copy()
    processed_item["task"] = task_name
    return processed_item


def evaluate_task(
    eval_dataset: list[dict[str, Any]],
    task_name: str,
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int,
    timeout: int = 20,
    exec_timeout: float = 3.0,
    seed: int | None = None,
) -> float | None:
    """
    Evaluate model outputs against ground truth data for a specific task.

    Dispatches on the task family (the part before ``/`` in *task_name*):

    - ``math_opensource`` → :func:`compute_scores` (math-verify)
    - ``mc_opensource``   → :func:`score_loglikelihood` when items carry
      ``logprobs``, otherwise :func:`score_generate`
    - ``code_opensource`` → :func:`score_code` (sandboxed pass@1)

    Args:
        eval_dataset: List of dictionaries containing the evaluation data
        task_name: Identifier for the evaluation task (format: 'source/specific_task')
        label_key: Dictionary key for accessing ground truth answers
        response_key: Dictionary key for accessing model responses
        cache_path: Path where evaluation results will be cached
        max_workers: Maximum number of parallel workers for processing
        timeout: Maximum time in seconds to wait for each evaluation (default: 20)
        exec_timeout: Per-item code execution timeout in seconds (code tasks only)
        seed: Random seed recorded in scorer provenance summaries

    Returns:
        Optional[float]: Evaluation accuracy score if successful, None if evaluation fails

    Example:
        >>> data = [{"input": "2+2", "answer": "4", "gen": "4"}]
        >>> accuracy = evaluate_task(
        ...     data, "math_opensource", "answer", "gen",
        ...     "cache/results", max_workers=4
        ... )
        >>> print(f"Accuracy: {accuracy:.2f}")
    """
    if not eval_dataset:
        logger.warning("Empty dataset provided for evaluation")
        return None

    # Parse task name to determine evaluation type
    task_parts = task_name.split("/")
    dataset_source = task_parts[0] if task_parts else task_name

    # Convert cache_path to Path object for consistent handling
    cache_path = Path(cache_path)

    if dataset_source == "math_opensource":
        try:
            accuracy = compute_scores(
                eval_dataset=eval_dataset,
                label_key=label_key,
                response_key=response_key,
                cache_path=str(cache_path),  # compute_scores expects string path
                max_workers=max_workers,
                timeout=timeout,
                task_name=task_name,
                seed=seed,
            )
            logger.info(f"✅ Task: {task_name}, Accuracy: {accuracy:.2%}")
            return accuracy
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e!s}", exc_info=True)
            return None
    elif dataset_source == "mc_opensource":
        try:
            mc_mode = _infer_mc_mode(eval_dataset)
            if mc_mode == "loglikelihood":
                accuracy = score_loglikelihood(
                    eval_dataset=eval_dataset,
                    cache_path=cache_path,
                    max_workers=max_workers,
                    timeout=timeout,
                    task_name=task_name,
                    seed=seed,
                )
                logger.info(
                    f"✅ Task: {task_name} (loglikelihood), Accuracy: {accuracy:.2%}"
                )
            else:
                accuracy = score_generate(
                    eval_dataset=eval_dataset,
                    label_key=label_key,
                    response_key=response_key,
                    cache_path=cache_path,
                    max_workers=max_workers,
                    timeout=timeout,
                    task_name=task_name,
                    seed=seed,
                )
                logger.info(
                    f"✅ Task: {task_name} (generate), Accuracy: {accuracy:.2%}"
                )
            return accuracy
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e!s}", exc_info=True)
            return None
    elif dataset_source == "code_opensource":
        try:
            accuracy = score_code(
                eval_dataset=eval_dataset,
                label_key=label_key,
                response_key=response_key,
                cache_path=cache_path,
                max_workers=max_workers,
                timeout=timeout,
                exec_timeout=exec_timeout,
                task_name=task_name,
                seed=seed,
            )
            logger.info(f"✅ Task: {task_name}, Pass@1: {accuracy:.2%}")
            return accuracy
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e!s}", exc_info=True)
            return None
    else:
        logger.error(f"🤷‍♂️ Unsupported task type: '{task_name}'")
        return None


def main() -> int:
    """
    Main entry point for the evaluation script.

    This function orchestrates the entire evaluation process:
    1. Parses command line arguments
    2. Sets up logging and cache directory
    3. Loads and validates input data
    4. Processes the data
    5. Runs the evaluation
    6. Reports results

    Returns:
        int: Exit code (0 for success, 1 for errors)
    """
    try:
        # Parse command line arguments using HuggingFace's argument parser
        parser = HfArgumentParser(EvalTaskArguments)
        (args,) = parser.parse_args_into_dataclasses()

        # Log initialization with formatted argument display
        logger.info("Initializing evaluation with the following configuration:")
        logger.info("\n--- Parsed Arguments ---")
        logger.info(json.dumps(dataclasses.asdict(args), indent=2))

        # Ensure cache directory exists
        cache_dir = Path(args.cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate input data
        try:
            with open(args.input_path, encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"❌ Input file not found: '{args.input_path}'")
            return 1
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format in '{args.input_path}': {e!s}")
            return 1

        if not data:
            logger.error("❌ Input file is empty")
            return 1

        # Process data items and handle potential errors.
        # MC inference must be a single schema per batch: all items with
        # ``logprobs`` (loglikelihood) or none (generate).
        try:
            if args.task_name.startswith("mc_opensource"):
                mc_mode = _infer_mc_mode(data)
                is_loglikelihood = mc_mode == "loglikelihood"
            else:
                is_loglikelihood = False

            if is_loglikelihood:
                processed_data = [{**item, "task": args.task_name} for item in data]
            else:
                processed_data = [
                    _process_item(
                        item, args.task_name, args.label_key, args.response_key
                    )
                    for item in data
                ]
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error processing data: {e!s}")
            return 1

        if args.contamination_path:
            try:
                contamination_sources = load_contamination_sources(
                    args.contamination_path
                )
                annotate_dataset_contamination(
                    processed_data,
                    contamination_sources,
                    input_key=args.input_key,
                    min_length=args.contamination_min_length,
                )
                logger.info(
                    "Loaded %d contamination reference string(s)",
                    len(contamination_sources),
                )
            except Exception as e:
                logger.error(f"❌ Error checking contamination: {e!s}", exc_info=True)
                return 1

        # Strip <think> tags from model responses before scoring.
        # Models using deepseek_r1/openr1 system prompts output
        # <think>...</think><answer>...</answer> format, and math_verify
        # may fail to extract answers from raw think-tagged text.
        preprocess_answers(processed_data, args.response_key)

        # Run evaluation and get results
        accuracy = evaluate_task(
            processed_data,
            args.task_name,
            args.label_key,
            args.response_key,
            args.cache_path,
            args.max_workers,
            args.timeout,
            args.exec_timeout,
            args.seed,
        )

        if accuracy is not None:
            logger.info("🎉 Evaluation completed successfully!")
            return 0
        else:
            logger.error("❌ Evaluation failed to produce results")
            return 1

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e!s}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
