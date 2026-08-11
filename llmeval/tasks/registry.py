"""Evaluation result models and the structured task registry.

The result types (:class:`EvaluationResult`, :class:`ScorerResult`,
:class:`MetricValue`) plus their bootstrap/aggregation helpers are the
contract shared by every scorer; the registry adapters translate scorer
output into persisted result files without filesystem round-trips.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Protocol

from llmeval.tasks.postprocess import atomic_write_json
from llmeval.utils.log import init_logger

logger = init_logger("task_registry")

__all__ = [
    "CodeTask",
    "EvaluationContext",
    "EvaluationResult",
    "EvaluationTask",
    "MCTask",
    "MathTask",
    "MetricValue",
    "PreparationContext",
    "ScorerResult",
    "TaskRegistry",
    "metric_from_samples",
    "persist_evaluation_result",
]


@dataclass(frozen=True)
class MetricValue:
    """A metric value with sample count and optional uncertainty metadata."""

    value: float
    count: int
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    higher_is_better: bool = True


@dataclass
class EvaluationResult:
    """Structured result shared by all registered evaluation tasks."""

    task_name: str
    task_version: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    sample_count: int = 0
    effective_sample_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    timeout_count: int = 0
    failure_counts: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    primary_metric: str | None = None

    @property
    def primary_value(self) -> float:
        """Return the conventional primary metric value."""
        if not self.metrics:
            return 0.0
        if self.primary_metric and self.primary_metric in self.metrics:
            return self.metrics[self.primary_metric].value
        return next(iter(self.metrics.values())).value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the aggregate result."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task_name": self.task_name,
            "task_version": self.task_version,
            "metrics": {
                name: {
                    "value": metric.value,
                    "count": metric.count,
                    "stderr": metric.stderr,
                    "ci_low": metric.ci_low,
                    "ci_high": metric.ci_high,
                    "higher_is_better": metric.higher_is_better,
                }
                for name, metric in self.metrics.items()
            },
            "sample_count": self.sample_count,
            "effective_sample_count": self.effective_sample_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "timeout_count": self.timeout_count,
            "failure_counts": dict(self.failure_counts),
            "details": self.details,
            "primary_metric": self.primary_metric,
        }
        return payload


@dataclass
class ScorerResult:
    """Task-scorer output consumed directly by registry adapters.

    ``metrics`` stores aggregate values for persistence and compatibility.
    ``observations`` stores the denominator-level values used to recompute
    uncertainty in :class:`EvaluationResult`. Scorers also return their
    per-item records so adapters never need to read JSONL or summary files.
    """

    metrics: dict[str, float]
    observations: dict[str, list[float]]
    records: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    effective_sample_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    timeout_count: int = 0
    failure_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("scorer metrics must be finite")
        counts = (
            self.sample_count,
            self.effective_sample_count,
            self.failed_count,
            self.skipped_count,
            self.timeout_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("scorer result counts must be non-negative")
        if any(
            not isinstance(name, str) or not isinstance(value, int) or value < 0
            for name, value in self.failure_counts.items()
        ):
            raise ValueError("failure counts must be non-negative integers")
        excluded_count = self.failed_count + self.skipped_count + self.timeout_count
        if excluded_count > self.sample_count:
            raise ValueError("excluded sample counts cannot exceed sample count")
        if self.effective_sample_count > self.sample_count:
            raise ValueError("effective sample count cannot exceed sample count")
        if self.effective_sample_count != self.sample_count - excluded_count:
            raise ValueError(
                "effective sample count must equal sample_count minus excluded counts"
            )
        unknown = set(self.observations) - set(self.metrics)
        if unknown:
            raise ValueError(
                f"observations reference undeclared metrics: {sorted(unknown)}"
            )
        for name, values in self.observations.items():
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"observations for metric {name!r} must be finite")


def metric_from_samples(
    samples: Iterable[float],
    seed: int,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    higher_is_better: bool = True,
) -> MetricValue:
    """Build a metric with deterministic bootstrap uncertainty."""
    values = [float(value) for value in samples]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return MetricValue(0.0, len(values), higher_is_better=higher_is_better)

    value = fmean(finite_values)
    if n_resamples <= 1:
        stderr = ci_low = ci_high = None
    elif len(finite_values) == 1:
        stderr, ci_low, ci_high = 0.0, value, value
    else:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        rng = random.Random(seed)
        bootstrap_means = sorted(
            fmean(rng.choice(finite_values) for _ in finite_values)
            for _ in range(n_resamples)
        )
        stderr = pstdev(bootstrap_means)

        def percentile(fraction: float) -> float:
            position = fraction * (len(bootstrap_means) - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return bootstrap_means[lower]
            weight = position - lower
            return (
                bootstrap_means[lower] * (1.0 - weight)
                + bootstrap_means[upper] * weight
            )

        alpha = (1.0 - confidence_level) / 2.0
        ci_low, ci_high = percentile(alpha), percentile(1.0 - alpha)
    return MetricValue(
        value=value,
        count=len(values),
        stderr=stderr,
        ci_low=ci_low,
        ci_high=ci_high,
        higher_is_better=higher_is_better,
    )


StructuredScorer = Callable[..., ScorerResult]


@dataclass(frozen=True)
class MetricSpec:
    """Static metadata for a task metric."""

    name: str
    higher_is_better: bool = True


@dataclass(frozen=True)
class PreparationContext:
    """Task metadata required while validating and annotating input rows."""

    task_name: str
    label_key: str
    response_key: str


@dataclass(frozen=True)
class EvaluationContext(PreparationContext):
    """All inputs that affect one evaluation invocation."""

    eval_dataset: list[dict[str, Any]]
    result_path: Path
    max_workers: int
    timeout: int
    exec_timeout: float
    seed: int
    mc_aggregation: str = "first"
    allow_unsafe_code: bool = False
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    code_k_values: tuple[int, ...] = (1, 10, 64)


class EvaluationTask(Protocol):
    """Protocol implemented by every registered task adapter."""

    family: str
    version: str
    metric_specs: tuple[MetricSpec, ...]

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]: ...

    def score(self, context: EvaluationContext) -> EvaluationResult: ...


def _build_evaluation_result(
    scored: ScorerResult,
    context: EvaluationContext,
    task: EvaluationTask,
) -> EvaluationResult:
    """Translate the shared scorer contract without filesystem round-trips."""
    specifications = {spec.name: spec for spec in task.metric_specs}
    metrics: dict[str, MetricValue] = {}
    bootstrap_cache: dict[tuple[float, ...], MetricValue] = {}
    for name, value in scored.metrics.items():
        spec = specifications.get(name, MetricSpec(name))
        observations = scored.observations.get(name, [])
        if observations:
            cache_key = tuple(float(observation) for observation in observations)
            cached = bootstrap_cache.get(cache_key)
            if cached is None:
                cached = metric_from_samples(
                    cache_key,
                    context.seed,
                    n_resamples=context.bootstrap_samples,
                    confidence_level=context.confidence_level,
                )
                bootstrap_cache[cache_key] = cached
            metrics[name] = MetricValue(
                value=cached.value,
                count=cached.count,
                stderr=cached.stderr,
                ci_low=cached.ci_low,
                ci_high=cached.ci_high,
                higher_is_better=spec.higher_is_better,
            )
        else:
            metrics[name] = MetricValue(
                value=float(value),
                count=0,
                higher_is_better=spec.higher_is_better,
            )
    missing = set(specifications) - set(metrics)
    if missing:
        raise ValueError(f"Scorer omitted declared metrics: {sorted(missing)}")
    return EvaluationResult(
        task_name=context.task_name,
        task_version=task.version,
        metrics=metrics,
        sample_count=scored.sample_count,
        effective_sample_count=scored.effective_sample_count,
        failed_count=scored.failed_count,
        skipped_count=scored.skipped_count,
        timeout_count=scored.timeout_count,
        failure_counts=dict(scored.failure_counts),
        records=[dict(item) for item in scored.records],
        details=dict(scored.details),
        primary_metric=task.metric_specs[0].name if task.metric_specs else None,
    )


def persist_evaluation_result(result: EvaluationResult, result_path: Path) -> None:
    """Persist only the aggregate result summary."""
    summary = result.to_dict()
    summary["summary_version"] = 1
    metric_values = {name: metric.value for name, metric in result.metrics.items()}
    summary["metric_values"] = metric_values
    for name, value in metric_values.items():
        summary.setdefault(name, value)
    atomic_write_json(result_path, summary, indent=2)


class GeneratedTask:
    """Shared validation for task families that score generated responses."""

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]:
        required = (context.label_key, context.response_key)
        prepared: list[dict[str, Any]] = []
        for item in data:
            missing = [key for key in required if key not in item]
            if missing:
                raise ValueError(
                    f"Missing required evaluation key(s) {missing!r}; "
                    f"available keys: {list(item)}"
                )
            prepared.append({**item, "task": context.task_name})
        return prepared


@dataclass
class MathTask(GeneratedTask):
    """Adapter for math-verify scoring."""

    scorer: StructuredScorer
    family: str = "math_opensource"
    version: str = "math_v1"
    metric_specs: tuple[MetricSpec, ...] = (MetricSpec("accuracy"),)

    def score(self, context: EvaluationContext) -> EvaluationResult:
        scored = self.scorer(
            eval_dataset=context.eval_dataset,
            label_key=context.label_key,
            response_key=context.response_key,
            max_workers=context.max_workers,
            timeout=context.timeout,
        )
        return _build_evaluation_result(scored, context, self)


@dataclass
class MCTask(GeneratedTask):
    """Adapter for generation and loglikelihood MC scoring."""

    generate_scorer: StructuredScorer
    loglikelihood_scorer: StructuredScorer
    family: str = "mc_opensource"
    version: str = "mc_v1"
    metric_specs: tuple[MetricSpec, ...] = (
        MetricSpec("acc"),
        MetricSpec("acc_norm"),
        MetricSpec("acc_bytes"),
        MetricSpec("exact_match"),
    )

    @staticmethod
    def _mc_schema(data: Sequence[dict[str, Any]]) -> bool:
        """Return whether *data* is uniformly loglikelihood schema.

        Raises ValueError when the dataset mixes loglikelihood and generate items.
        """
        has_logprobs: list[bool] = []
        for index, item in enumerate(data):
            if "logprobs" not in item:
                has_logprobs.append(False)
                continue
            if not isinstance(item["logprobs"], list):
                raise ValueError(
                    f"MC item {index} has invalid logprobs; expected a list"
                )
            has_logprobs.append(True)
        if any(has_logprobs) and not all(has_logprobs):
            raise ValueError(
                "Mixed MC dataset detected: all items must use loglikelihood or "
                "generate schema"
            )
        return all(has_logprobs)

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]:
        if self._mc_schema(data):
            return [{**item, "task": context.task_name} for item in data]
        return super().prepare_dataset(data, context)

    def score(self, context: EvaluationContext) -> EvaluationResult:
        is_loglikelihood = self._mc_schema(context.eval_dataset)
        scorer = self.loglikelihood_scorer if is_loglikelihood else self.generate_scorer
        kwargs: dict[str, Any] = {
            "eval_dataset": context.eval_dataset,
            "max_workers": context.max_workers,
            "timeout": context.timeout,
        }
        if not is_loglikelihood:
            kwargs.update(
                label_key=context.label_key,
                response_key=context.response_key,
                aggregation=context.mc_aggregation,
            )
        scored = scorer(**kwargs)
        return _build_evaluation_result(scored, context, self)


@dataclass
class CodeTask(GeneratedTask):
    """Adapter for sandboxed code scoring."""

    scorer: StructuredScorer
    family: str = "code_opensource"
    version: str = "code_v1"
    metric_specs: tuple[MetricSpec, ...] = (MetricSpec("pass@1"),)

    def score(self, context: EvaluationContext) -> EvaluationResult:
        scored = self.scorer(
            eval_dataset=context.eval_dataset,
            label_key=context.label_key,
            response_key=context.response_key,
            max_workers=context.max_workers,
            timeout=context.timeout,
            exec_timeout=context.exec_timeout,
            k_values=context.code_k_values,
            allow_unsafe_code=context.allow_unsafe_code,
        )
        return _build_evaluation_result(scored, context, self)


class TaskRegistry:
    """Resolve task names to registered adapters without central branching."""

    def __init__(self, tasks: dict[str, EvaluationTask] | None = None) -> None:
        self._tasks = dict(tasks or {})

    def register(self, task: EvaluationTask) -> None:
        existing = self._tasks.get(task.family)
        if existing is not None and existing is not task:
            logger.warning(
                "Task family %r re-registered: replacing %r with %r",
                task.family,
                type(existing).__name__,
                type(task).__name__,
            )
        self._tasks[task.family] = task

    def resolve(self, task_name: str) -> EvaluationTask:
        family = task_name.split("/", 1)[0]
        try:
            return self._tasks[family]
        except KeyError as exc:
            available = ", ".join(sorted(self._tasks)) or "<none>"
            raise ValueError(
                f"Unsupported task family {family!r}; registered tasks: {available}"
            ) from exc

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Resolve, score, and persist one evaluation invocation."""
        result = self.resolve(context.task_name).score(context)
        persist_evaluation_result(result, context.result_path)
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))
