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
    - Structured JSON summary output
    - Parallel processing capabilities
    - Robust error handling

Example:
    $ python -m llmeval.evaluator --input_path data.jsonl --task_name math_opensource/aime24 --result_path results.json

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
    CodeTask,
    EvaluationContext,
    EvaluationResult,
    EvaluationTask,
    MathTask,
    MCTask,
    MetricValue,
    PreparationContext,
    TaskRegistry,
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


def evaluate_task(
    eval_dataset: list[dict[str, Any]],
    task_name: str,
    label_key: str,
    response_key: str,
    result_path: str | Path,
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
        ...     "results/evaluation.json", max_workers=4
        ... )
        >>> print(f"Accuracy: {accuracy:.2f}")
    """
    result = evaluate_task_result(
        eval_dataset=eval_dataset,
        task_name=task_name,
        label_key=label_key,
        response_key=response_key,
        result_path=result_path,
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


def _task_families() -> dict[str, tuple[type[Any], EvaluationTask]]:
    """Build the task-family definitions from injectable module scorers."""
    return {
        "math_opensource": (MathEvalArguments, MathTask(score_math_result)),
        "mc_opensource": (
            MCEvalArguments,
            MCTask(score_generate_result, score_loglikelihood_result),
        ),
        "code_opensource": (CodeEvalArguments, CodeTask(score_code_result)),
    }


def _default_registry() -> TaskRegistry:
    return TaskRegistry(
        {family: task for family, (_, task) in _task_families().items()}
    )


def evaluate_task_result(
    eval_dataset: list[dict[str, Any]],
    task_name: str,
    label_key: str,
    response_key: str,
    result_path: str | Path,
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
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers!r}")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    if (
        not isinstance(exec_timeout, int | float)
        or isinstance(exec_timeout, bool)
        or exec_timeout <= 0
    ):
        raise ValueError(f"exec_timeout must be positive, got {exec_timeout!r}")
    if type(actual_seed) is not int or actual_seed < 0:
        raise ValueError(f"seed must be non-negative, got {actual_seed!r}")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError(
            f"bootstrap_samples must be non-negative, got {bootstrap_samples!r}"
        )
    if not isinstance(confidence_level, int | float) or not (
        0.0 < confidence_level < 1.0
    ):
        raise ValueError(
            f"confidence_level must be between 0 and 1, got {confidence_level!r}"
        )
    if (
        not isinstance(code_k_values, tuple)
        or not code_k_values
        or any(type(value) is not int or value <= 0 for value in code_k_values)
    ):
        raise ValueError(
            f"code_k_values must contain positive integers, got {code_k_values!r}"
        )
    if not str(result_path).strip():
        raise ValueError("result_path must be a non-empty file path")
    resolved_result_path = Path(result_path)
    if resolved_result_path.exists() and resolved_result_path.is_dir():
        raise ValueError("result_path must be a file path, not a directory")
    registry = _default_registry()
    task = registry.resolve(task_name)
    prepared_dataset = task.prepare_dataset(
        eval_dataset,
        PreparationContext(
            task_name=task_name,
            label_key=label_key,
            response_key=response_key,
        ),
    )
    context = EvaluationContext(
        eval_dataset=prepared_dataset,
        task_name=task_name,
        label_key=label_key,
        response_key=response_key,
        result_path=resolved_result_path,
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
    if not eval_dataset:
        logger.warning("Empty dataset provided for evaluation")
        result = EvaluationResult(
            task_name=task_name,
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
        persist_evaluation_result(result, context.result_path)
    else:
        result = registry.evaluate(context)
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
        "Task %s counts: samples=%d effective=%d failed=%d",
        task_name,
        result.sample_count,
        result.effective_sample_count,
        result.failed_count,
    )
    return result


def select_eval_arguments(task_name: str) -> type[Any]:
    """Select the task-specific CLI schema from a task family name."""
    family = task_name.split("/", 1)[0]
    argument_types = {
        family: definition[0] for family, definition in _task_families().items()
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
    selector.add_argument("--task_name", default="math_opensource/aime24")
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

        # Run evaluation and retain the complete metric result.
        evaluate_task_result(
            eval_dataset=data,
            task_name=args.task_name,
            label_key=args.label_key,
            response_key=args.response_key,
            result_path=args.result_path,
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
