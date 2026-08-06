"""Unified evaluation metrics and uncertainty helpers.

The individual task scorers keep their historical float-returning APIs, while
the evaluator and task registry use :class:`EvaluationResult` to preserve all
metrics and denominator information.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any

__all__ = [
    "EvaluationResult",
    "MetricValue",
    "ScorerResult",
    "aggregate_metric_values",
    "bootstrap_metric",
    "metric_from_samples",
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
