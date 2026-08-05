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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricValue:
        """Deserialize a metric and tolerate older cache fields."""
        return cls(
            value=float(data.get("value", 0.0)),
            count=int(data.get("count", 0)),
            stderr=(float(data["stderr"]) if data.get("stderr") is not None else None),
            ci_low=(float(data["ci_low"]) if data.get("ci_low") is not None else None),
            ci_high=(
                float(data["ci_high"]) if data.get("ci_high") is not None else None
            ),
            higher_is_better=bool(data.get("higher_is_better", True)),
        )


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
    provenance: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None

    @property
    def primary_value(self) -> float:
        """Return the conventional primary metric value."""
        if not self.metrics:
            return 0.0
        return next(iter(self.metrics.values())).value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete result for the content-addressed cache."""
        return {
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
            "provenance": self.provenance,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Deserialize a cached result and reject malformed metric payloads."""
        raw_metrics = data.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raise ValueError("Evaluation result metrics must be an object")
        metrics = {
            str(name): MetricValue.from_dict(value)
            for name, value in raw_metrics.items()
            if isinstance(value, dict)
        }
        if not metrics and raw_metrics:
            raise ValueError("Evaluation result contains no valid metrics")
        return cls(
            task_name=str(data.get("task_name", "")),
            task_version=str(data.get("task_version", "")),
            metrics=metrics,
            sample_count=int(data.get("sample_count", 0)),
            effective_sample_count=int(data.get("effective_sample_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            skipped_count=int(data.get("skipped_count", 0)),
            timeout_count=int(data.get("timeout_count", 0)),
            provenance=(
                dict(data["provenance"])
                if isinstance(data.get("provenance"), dict)
                else {}
            ),
            cache_key=(str(data["cache_key"]) if data.get("cache_key") else None),
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
