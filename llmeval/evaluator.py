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

from llmeval.inference.common import load_jsonl
from llmeval.tasks.code_eval.code_score import score_code_result
from llmeval.tasks.math_eval.math_score import compute_score_result
from llmeval.tasks.mc_eval.mc_score import (
    score_generate_result,
    score_loglikelihood_result,
)
from llmeval.tasks.provenance import (
    annotate_dataset_contamination,
    build_run_provenance,
    load_contamination_sources,
)
from llmeval.tasks.registry import (
    EvaluationContext,
    EvaluationResult,
    PreparationContext,
    TaskRegistry,
    build_default_registry,
    evaluate_registered_task,
    write_structured_summary,
)
from llmeval.tasks.results import MetricValue
from llmeval.utils.config import EvalTaskArguments
from llmeval.utils.log import init_logger
from llmeval.utils.reproducibility import seed_everything, seed_provenance

# Initialize logger for the evaluation orchestrator
logger = init_logger("evaluator")


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
    mc_aggregation: str = "first",
    allow_unsafe_code: bool = False,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    content_cache_dir: str | Path | None = None,
    force_recompute: bool = False,
    read_only_cache: bool = False,
    model_name: str | None = None,
    model_revision: str | None = None,
    input_key: str = "prompt",
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
        mc_aggregation: Generate-mode MC aggregation strategy.
        allow_unsafe_code: Explicit opt-in required for code execution.

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
    result = evaluate_task_result(
        eval_dataset=eval_dataset,
        task_name=task_name,
        label_key=label_key,
        response_key=response_key,
        cache_path=cache_path,
        max_workers=max_workers,
        timeout=timeout,
        exec_timeout=exec_timeout,
        seed=seed,
        mc_aggregation=mc_aggregation,
        allow_unsafe_code=allow_unsafe_code,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        content_cache_dir=content_cache_dir,
        force_recompute=force_recompute,
        read_only_cache=read_only_cache,
        model_name=model_name,
        model_revision=model_revision,
        input_key=input_key,
    )
    return result.primary_value if result is not None else None


def _default_registry() -> TaskRegistry:
    """Build a registry from module symbols so scorer tests remain injectable."""
    return build_default_registry(
        compute_score_result,
        score_generate_result,
        score_loglikelihood_result,
        score_code_result,
    )


def prepare_evaluation_data(
    data: list[dict[str, Any]],
    task_name: str,
    label_key: str,
    response_key: str,
) -> list[dict[str, Any]]:
    """Validate and annotate input through the registered task adapter."""
    context = PreparationContext(
        task_name=task_name,
        label_key=label_key,
        response_key=response_key,
    )
    return _default_registry().resolve(task_name).prepare_dataset(data, context)


def evaluate_task_result(
    eval_dataset: list[dict[str, Any]],
    task_name: str,
    label_key: str,
    response_key: str,
    cache_path: str | Path,
    max_workers: int,
    timeout: int = 20,
    exec_timeout: float = 3.0,
    seed: int | None = None,
    mc_aggregation: str = "first",
    allow_unsafe_code: bool = False,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    content_cache_dir: str | Path | None = None,
    force_recompute: bool = False,
    read_only_cache: bool = False,
    model_name: str | None = None,
    model_revision: str | None = None,
    input_key: str = "prompt",
) -> EvaluationResult | None:
    """Evaluate a task through the registry and return all declared metrics."""
    actual_seed = 0 if seed is None else seed
    seed_state = seed_everything(actual_seed)
    seed_prov = seed_provenance(seed_state)
    context = EvaluationContext(
        eval_dataset=eval_dataset,
        task_name=task_name,
        label_key=label_key,
        response_key=response_key,
        cache_path=Path(cache_path),
        max_workers=max_workers,
        timeout=timeout,
        exec_timeout=exec_timeout,
        seed=actual_seed,
        mc_aggregation=mc_aggregation,
        allow_unsafe_code=allow_unsafe_code,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        model_name=model_name,
        model_revision=model_revision,
        content_cache_dir=Path(content_cache_dir) if content_cache_dir else None,
        force_recompute=force_recompute,
        read_only_cache=read_only_cache,
        input_key=input_key,
        extra_provenance=seed_prov,
    )
    registry = _default_registry()
    try:
        task = registry.resolve(task_name)
        if not eval_dataset:
            logger.warning("Empty dataset provided for evaluation")
            result = EvaluationResult(
                task_name=task_name,
                task_version=task.version,
                metrics={
                    spec.name: MetricValue(
                        value=0.0,
                        count=0,
                        higher_is_better=spec.higher_is_better,
                    )
                    for spec in task.metric_specs
                },
                provenance={
                    **build_run_provenance(
                        [],
                        task_name=task_name,
                        input_key=input_key,
                        label_key=label_key,
                        response_key=response_key,
                        seed=actual_seed,
                    ),
                    **seed_prov,
                },
            )
            write_structured_summary(result, context.cache_path)
        else:
            result = evaluate_registered_task(context, registry)
        for name, metric in result.metrics.items():
            logger.info(
                "Task %s metric %s=%.4f (n=%d, stderr=%s)",
                task_name,
                name,
                metric.value,
                metric.count,
                f"{metric.stderr:.6f}" if metric.stderr is not None else "N/A",
            )
        logger.info(
            "Task %s counts: samples=%d effective=%d failed=%d skipped=%d timeout=%d",
            task_name,
            result.sample_count,
            result.effective_sample_count,
            result.failed_count,
            result.skipped_count,
            result.timeout_count,
        )
        return result
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
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
        parser = HfArgumentParser(EvalTaskArguments)  # type: ignore[arg-type]
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
            data = load_jsonl(args.input_path)
        except FileNotFoundError:
            logger.error(f"❌ Input file not found: '{args.input_path}'")
            return 1
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format in '{args.input_path}': {e!s}")
            return 1
        except ValueError as e:
            logger.error(f"❌ Invalid evaluation record: {e!s}")
            return 1

        if not data:
            logger.error("❌ Input file is empty")
            return 1

        # Task-specific validation and annotation live in the registry. The
        # evaluator does not need to know how a task identifies its schema.
        try:
            processed_data = prepare_evaluation_data(
                data, args.task_name, args.label_key, args.response_key
            )
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

        # Run evaluation and retain the complete metric result.
        result = evaluate_task_result(
            processed_data,
            args.task_name,
            args.label_key,
            args.response_key,
            args.cache_path,
            args.max_workers,
            args.timeout,
            args.exec_timeout,
            args.seed,
            args.mc_aggregation,
            args.allow_unsafe_code,
            args.bootstrap_samples,
            args.confidence_level,
            args.content_cache_dir,
            args.force_recompute,
            args.read_only_cache,
            args.model_name,
            args.model_revision,
            input_key=args.input_key,
        )

        if result is not None:
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
