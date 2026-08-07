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

from llmeval.tasks.persistence import (
    atomic_write_json,
    atomic_write_jsonl,
    persist_results,
)
from llmeval.utils.log import init_logger

logger = init_logger("task_registry")

__all__ = [
    "CodeTask",
    "EvaluationContext",
    "EvaluationResult",
    "MCTask",
    "MathTask",
    "MetricValue",
    "PreparationContext",
    "ScorerResult",
    "TaskRegistry",
    "aggregate_metric_values",
    "build_default_registry",
    "evaluate_registered_task",
    "metric_from_samples",
    "persist_evaluation_result",
    "write_per_item_results",
    "write_structured_summary",
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the metric using JSON-compatible primitive values."""
        return {
            "value": self.value,
            "count": self.count,
            "stderr": self.stderr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "higher_is_better": self.higher_is_better,
        }


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
    per_item: list[dict[str, Any]] = field(default_factory=list)
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

    def to_dict(self, *, include_per_item: bool = False) -> dict[str, Any]:
        """Serialize a compact result, optionally including detailed records."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task_name": self.task_name,
            "task_version": self.task_version,
            "metrics": {
                name: metric.to_dict() for name, metric in self.metrics.items()
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
        if include_per_item:
            payload["per_item"] = self.per_item
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
    per_item: list[dict[str, Any]] = field(default_factory=list)
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
        if any(count < 0 for count in self._counts):
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

    @property
    def _counts(self) -> tuple[int, ...]:
        return (
            self.sample_count,
            self.effective_sample_count,
            self.failed_count,
            self.skipped_count,
            self.timeout_count,
        )


def bootstrap_metric(
    samples: Iterable[float],
    seed: int,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float | None, float | None, float | None]:
    """Calculate mean, bootstrap stderr, and percentile confidence bounds.

    The sampler uses a local ``random.Random`` instance, so evaluating one task
    cannot perturb another task's random stream. Empty samples return a zero
    value with no uncertainty; one sample returns a degenerate interval.
    """
    values = [float(value) for value in samples if math.isfinite(float(value))]
    if not values:
        return 0.0, None, None, None
    mean_value = fmean(values)
    if len(values) == 1 or n_resamples <= 1:
        return mean_value, 0.0, mean_value, mean_value
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = random.Random(seed)
    bootstrap_means = [
        fmean(rng.choice(values) for _ in values) for _ in range(n_resamples)
    ]
    stderr = pstdev(bootstrap_means)
    sorted_means = sorted(bootstrap_means)
    alpha = (1.0 - confidence_level) / 2.0

    def percentile(fraction: float) -> float:
        position = fraction * (len(sorted_means) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_means[lower]
        weight = position - lower
        return sorted_means[lower] * (1.0 - weight) + sorted_means[upper] * weight

    return mean_value, stderr, percentile(alpha), percentile(1.0 - alpha)


def metric_from_samples(
    samples: Iterable[float],
    seed: int,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    higher_is_better: bool = True,
) -> MetricValue:
    """Build a :class:`MetricValue` from per-sample observations."""
    values = [float(value) for value in samples]
    value, stderr, ci_low, ci_high = bootstrap_metric(
        values,
        seed,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )
    return MetricValue(
        value=value,
        count=len(values),
        stderr=stderr,
        ci_low=ci_low,
        ci_high=ci_high,
        higher_is_better=higher_is_better,
    )


def aggregate_metric_values(
    values: Iterable[MetricValue],
    *,
    mode: str = "micro",
    seed: int = 0,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> MetricValue:
    """Aggregate metric values using a documented macro or micro denominator.

    ``micro`` treats each source observation as one observation and therefore
    weights inputs by ``count``. ``macro`` gives every input metric equal
    weight. The input values are already summaries, so uncertainty is
    recomputed from the available values rather than combining incompatible
    standard errors.
    """
    metrics = list(values)
    if mode not in {"macro", "micro"}:
        raise ValueError("mode must be 'macro' or 'micro'")
    if not metrics:
        return MetricValue(value=0.0, count=0)

    if mode == "micro":
        total_count = sum(max(metric.count, 0) for metric in metrics)
        if total_count == 0:
            return MetricValue(value=0.0, count=0)
        weighted = [metric.value for metric in metrics for _ in range(metric.count)]
        return metric_from_samples(
            weighted,
            seed,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            higher_is_better=all(metric.higher_is_better for metric in metrics),
        )

    return metric_from_samples(
        (metric.value for metric in metrics),
        seed,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        higher_is_better=all(metric.higher_is_better for metric in metrics),
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
    cache_path: Path
    max_workers: int
    timeout: int
    exec_timeout: float
    seed: int
    mc_aggregation: str = "first"
    allow_unsafe_code: bool = False
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    output_schema: str = "compact"
    expected_samples: int = 0
    code_k_values: tuple[int, ...] = (1, 10, 64)


class EvaluationTask(Protocol):
    """Protocol implemented by every registered task adapter."""

    family: str
    version: str
    pipeline_version: str
    metric_specs: tuple[MetricSpec, ...]

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]: ...

    def score(self, context: EvaluationContext) -> EvaluationResult: ...


def _metric(
    values: list[float], context: EvaluationContext, *, higher_is_better: bool = True
) -> MetricValue:
    return metric_from_samples(
        values,
        context.seed,
        n_resamples=context.bootstrap_samples,
        confidence_level=context.confidence_level,
        higher_is_better=higher_is_better,
    )


def _build_evaluation_result(
    scored: ScorerResult,
    context: EvaluationContext,
    task: EvaluationTask,
) -> EvaluationResult:
    """Translate the shared scorer contract without filesystem round-trips."""
    specifications = {spec.name: spec for spec in task.metric_specs}
    metrics: dict[str, MetricValue] = {}
    for name, value in scored.metrics.items():
        spec = specifications.get(name, MetricSpec(name))
        observations = scored.observations.get(name, [])
        metrics[name] = (
            _metric(observations, context, higher_is_better=spec.higher_is_better)
            if observations
            else MetricValue(
                value=float(value),
                count=0,
                higher_is_better=spec.higher_is_better,
            )
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
        per_item=[dict(item) for item in scored.per_item],
        details=dict(scored.details),
        primary_metric=task.metric_specs[0].name if task.metric_specs else None,
    )


def _structured_summary(result: EvaluationResult) -> dict[str, Any]:
    """Build the summary schema shared by all registered task families."""
    payload = result.to_dict(include_per_item=True)
    payload["summary_version"] = 1
    metric_values = {name: metric.value for name, metric in result.metrics.items()}
    payload["metric_values"] = metric_values
    for name, value in metric_values.items():
        payload.setdefault(name, value)
    return payload


def write_structured_summary(result: EvaluationResult, cache_path: Path) -> None:
    """Write the registry result in one schema shared by all task families."""
    atomic_write_json(
        cache_path.with_suffix(".summary.json"),
        _structured_summary(result),
        indent=2,
    )


def _compact_per_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep stable scoring fields while dropping prompt/generation payloads."""
    aliases = {
        "doc_id": ("doc_id",),
        "gold": ("gold", "extracted_gold", "answer"),
        "pred": ("pred", "extracted_answer", "prediction"),
        "correct": ("correct", "passed", "accuracy"),
        "score": ("score", "accuracy", "pass@1", "acc"),
        "scoring_mode": ("scoring_mode", "aggregation"),
        "evaluation_status": ("evaluation_status",),
    }
    compact: dict[str, Any] = {}
    for output_key, candidates in aliases.items():
        for key in candidates:
            value = item.get(key)
            if value is None or value == "" or value == []:
                continue
            if output_key == "correct" and key == "accuracy":
                value = float(value) == 1.0
            compact[output_key] = value
            break
    if "score" not in compact and "correct" in compact:
        compact["score"] = 1.0 if compact["correct"] else 0.0
    for key in ("id", "task", "task_id", "correct_norm", "correct_bytes"):
        value = item.get(key)
        if value is not None and value != "" and value != []:
            compact[key] = value
    return compact


def _per_item_records(
    result: EvaluationResult, output_schema: str
) -> list[dict[str, Any]]:
    """Build per-item records using the requested compact/debug schema."""
    if output_schema not in {"compact", "debug"}:
        raise ValueError("output_schema must be 'compact' or 'debug'")
    return [
        item if output_schema == "debug" else _compact_per_item(item)
        for item in result.per_item
    ]


def write_per_item_results(
    result: EvaluationResult, cache_path: Path, *, output_schema: str = "compact"
) -> None:
    """Persist per-item records using the requested compact/debug schema."""
    atomic_write_jsonl(cache_path, _per_item_records(result, output_schema))


def persist_evaluation_result(
    result: EvaluationResult, cache_path: Path, *, output_schema: str = "compact"
) -> None:
    """Persist both registry artifacts through the shared atomic entry point."""
    persist_results(
        cache_path,
        _per_item_records(result, output_schema),
        _structured_summary(result),
    )


@dataclass
class MathTask:
    """Adapter for math-verify scoring."""

    scorer: StructuredScorer
    family: str = "math_opensource"
    version: str = "math_v1"
    pipeline_version: str = "math_response_v1"
    metric_specs: tuple[MetricSpec, ...] = (MetricSpec("accuracy"),)

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]:
        return [
            _annotate_item(
                item, context.task_name, context.label_key, context.response_key
            )
            for item in data
        ]

    def score(self, context: EvaluationContext) -> EvaluationResult:
        scored = self.scorer(
            eval_dataset=context.eval_dataset,
            label_key=context.label_key,
            response_key=context.response_key,
            cache_path=str(context.cache_path),
            max_workers=context.max_workers,
            timeout=context.timeout,
            expected_samples=context.expected_samples or None,
            persist_legacy=False,
        )
        return _build_evaluation_result(scored, context, self)


@dataclass
class MCTask:
    """Adapter for generation and loglikelihood MC scoring."""

    generate_scorer: StructuredScorer
    loglikelihood_scorer: StructuredScorer
    family: str = "mc_opensource"
    version: str = "mc_v1"
    pipeline_version: str = "mc_generation_v1"
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
        has_logprobs = ["logprobs" in item for item in data]
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
        return [
            _annotate_item(
                item, context.task_name, context.label_key, context.response_key
            )
            for item in data
        ]

    def score(self, context: EvaluationContext) -> EvaluationResult:
        is_loglikelihood = self._mc_schema(context.eval_dataset)
        scorer = self.loglikelihood_scorer if is_loglikelihood else self.generate_scorer
        kwargs: dict[str, Any] = {
            "eval_dataset": context.eval_dataset,
            "cache_path": context.cache_path,
            "max_workers": context.max_workers,
            "timeout": context.timeout,
        }
        if not is_loglikelihood:
            kwargs.update(
                label_key=context.label_key,
                response_key=context.response_key,
                aggregation=context.mc_aggregation,
            )
        kwargs["persist_legacy"] = False
        scored = scorer(**kwargs)
        return _build_evaluation_result(scored, context, self)


@dataclass
class CodeTask:
    """Adapter for sandboxed code scoring."""

    scorer: StructuredScorer
    family: str = "code_opensource"
    version: str = "code_v1"
    pipeline_version: str = "code_generation_v1"
    metric_specs: tuple[MetricSpec, ...] = (MetricSpec("pass@1"),)

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]:
        return [
            _annotate_item(
                item, context.task_name, context.label_key, context.response_key
            )
            for item in data
        ]

    def score(self, context: EvaluationContext) -> EvaluationResult:
        scored = self.scorer(
            eval_dataset=context.eval_dataset,
            label_key=context.label_key,
            response_key=context.response_key,
            cache_path=context.cache_path,
            max_workers=context.max_workers,
            timeout=context.timeout,
            exec_timeout=context.exec_timeout,
            k_values=context.code_k_values,
            allow_unsafe_code=context.allow_unsafe_code,
            persist_legacy=False,
        )
        return _build_evaluation_result(scored, context, self)


def _annotate_item(
    item: dict[str, Any], task_name: str, label_key: str, response_key: str
) -> dict[str, Any]:
    """Validate common generated-output fields and attach the task name."""
    missing = [key for key in (label_key, response_key) if key not in item]
    if missing:
        raise ValueError(
            f"Missing required evaluation key(s) {missing!r}; "
            f"available keys: {list(item)}"
        )
    return {**item, "task": task_name}


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

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))


def build_default_registry(
    math_scorer: StructuredScorer,
    mc_generate_scorer: StructuredScorer,
    mc_loglikelihood_scorer: StructuredScorer,
    code_scorer: StructuredScorer,
) -> TaskRegistry:
    """Build the standard registry; scorer arguments keep tests injectable."""
    return TaskRegistry(
        {
            "math_opensource": MathTask(math_scorer),
            "mc_opensource": MCTask(mc_generate_scorer, mc_loglikelihood_scorer),
            "code_opensource": CodeTask(code_scorer),
        }
    )


def evaluate_registered_task(
    context: EvaluationContext, registry: TaskRegistry
) -> EvaluationResult:
    """Evaluate through a registry and persist structured result files."""
    task = registry.resolve(context.task_name)
    result = task.score(context)
    persist_evaluation_result(
        result, context.cache_path, output_schema=context.output_schema
    )
    return result
