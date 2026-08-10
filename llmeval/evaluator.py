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
    - Structured JSONL and summary output
    - Parallel processing capabilities
    - Robust error handling

Example:
    $ python llmeval/evaluator.py --input_path data.jsonl --task_name math_opensource/aime24 --cache_path cache/

Author: jianzhnie
Date: 2025
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from transformers import HfArgumentParser

from llmeval.inference.common import load_jsonl, redact_config_for_logging
from llmeval.tasks.code_eval.code_score import score_code_result
from llmeval.tasks.math_eval.math_score import score_math_result
from llmeval.tasks.mc_eval.mc_score import (
    score_generate_result,
    score_loglikelihood_result,
)
from llmeval.tasks.registry import (
    EvaluationContext,
    EvaluationResult,
    MetricValue,
    PreparationContext,
    TaskRegistry,
    build_default_registry,
    evaluate_registered_task,
    persist_evaluation_result,
)
from llmeval.utils.config import (
    CodeEvalArguments,
    MathEvalArguments,
    MCEvalArguments,
)
from llmeval.utils.log import init_logger

# Initialize logger for the evaluation orchestrator
logger = init_logger("evaluator")


def _resolve_cache_path(cache_path: str | Path, task_name: str) -> Path:
    """Resolve legacy directory-style cache paths to a JSONL output file."""
    raw_path = str(cache_path)
    path = Path(cache_path)
    if (path.exists() and path.is_dir()) or raw_path.endswith(("/", "\\")):
        filename = task_name.replace("/", "_") or "evaluation"
        return path / f"{filename}.jsonl"
    return path


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
    code_k_values: tuple[int, ...] = (1, 10, 64),
) -> float:
    """
    Evaluate model outputs and return the primary metric value.

    Thin wrapper over :func:`evaluate_task_result`; the task registry dispatches
    on the family prefix in *task_name* (``math_opensource`` / ``mc_opensource`` /
    ``code_opensource``). See :func:`evaluate_task_result` for parameter docs.

    Returns:
        The task's primary metric value.

    Raises:
        Exception: Propagates task resolution, scoring, and persistence failures.

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
        code_k_values=code_k_values,
    )
    return result.primary_value


def _default_registry() -> TaskRegistry:
    """Build a registry from module symbols so scorer tests remain injectable."""
    return build_default_registry(
        score_math_result,
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
    code_k_values: tuple[int, ...] = (1, 10, 64),
) -> EvaluationResult:
    """Evaluate a task through the registry and return all declared metrics."""
    actual_seed = 0 if seed is None else seed
    resolved_cache_path = _resolve_cache_path(cache_path, task_name)
    context = EvaluationContext(
        eval_dataset=eval_dataset,
        task_name=task_name,
        label_key=label_key,
        response_key=response_key,
        cache_path=resolved_cache_path,
        max_workers=max_workers,
        timeout=timeout,
        exec_timeout=exec_timeout,
        seed=actual_seed,
        mc_aggregation=mc_aggregation,
        allow_unsafe_code=allow_unsafe_code,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        code_k_values=code_k_values,
    )
    registry = _default_registry()
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
            primary_metric=task.metric_specs[0].name if task.metric_specs else None,
        )
        persist_evaluation_result(result, context.cache_path)
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
    if result.failure_counts:
        logger.info("Task %s failure breakdown: %s", task_name, result.failure_counts)
    return result


def select_eval_arguments(task_name: str) -> type[Any]:
    """Select the task-specific CLI schema from a task family name."""
    family = task_name.split("/", 1)[0]
    argument_types = {
        "math_opensource": MathEvalArguments,
        "mc_opensource": MCEvalArguments,
        "code_opensource": CodeEvalArguments,
    }
    try:
        return argument_types[family]
    except KeyError as exc:
        available = ", ".join(argument_types)
        raise ValueError(
            f"Unsupported task family {family!r}; expected one of: {available}"
        ) from exc


def _parse_eval_arguments(argv: list[str] | None = None) -> Any:
    """Parse CLI arguments with only the selected task's options exposed."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selected = selector.parse_known_args(argv)[0]
    argument_type = select_eval_arguments(selected.task_name)
    parser = HfArgumentParser(argument_type)  # type: ignore[arg-type]
    (args,) = parser.parse_args_into_dataclasses(args=argv)
    return args


def main() -> int:
    """
    Main entry point for the evaluation script.

    This function orchestrates the entire evaluation process:
    1. Parses command line arguments
    2. Sets up logging and the result output directory
    3. Loads and validates input data
    4. Processes the data
    5. Runs the evaluation
    6. Reports results

    Returns:
        int: Exit code (0 for success, 1 for errors)
    """
    try:
        args = _parse_eval_arguments()

        # Log initialization with formatted argument display
        logger.info("Initializing evaluation with the following configuration:")
        logger.info("\n--- Parsed Arguments ---")
        logger.info(
            json.dumps(
                redact_config_for_logging(dataclasses.asdict(args)),
                indent=2,
                default=str,
            )
        )

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

        # Run evaluation and retain the complete metric result.
        evaluate_task_result(
            eval_dataset=processed_data,
            task_name=args.task_name,
            label_key=args.label_key,
            response_key=args.response_key,
            cache_path=args.cache_path,
            max_workers=args.max_workers,
            timeout=args.timeout,
            exec_timeout=getattr(args, "exec_timeout", 3.0),
            seed=args.seed,
            mc_aggregation=getattr(args, "mc_aggregation", "first"),
            allow_unsafe_code=getattr(args, "allow_unsafe_code", False),
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            code_k_values=getattr(args, "code_k_values_tuple", (1, 10, 64)),
        )

        logger.info("🎉 Evaluation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e!s}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
