"""Task registry and structured adapters for evaluation task families."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llmeval.cache import ContentAddressedCache
from llmeval.tasks.results import (
    EvaluationResult,
    MetricValue,
    ScorerResult,
    metric_from_samples,
)

StructuredScorer = Callable[..., ScorerResult]

_EVALUATION_RUNTIME_FIELDS = {
    "accuracy",
    "evaluation_status",
    "extracted_answer",
    "extracted_gold",
    "fallback_matched",
    "filter_trace",
    "filtered_gen",
    "raw_gen",
}


def _evaluation_cache_inputs(
    dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy stable evaluator inputs without fields added by scoring."""
    return [
        {
            key: value
            for key, value in item.items()
            if key not in _EVALUATION_RUNTIME_FIELDS
        }
        for item in dataset
    ]


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
    model_name: str | None = None
    model_revision: str | None = None
    content_cache_dir: Path | None = None
    force_recompute: bool = False
    read_only_cache: bool = False
    input_key: str = "prompt"


class EvaluationTask(Protocol):
    """Protocol implemented by every registered task adapter."""

    family: str
    version: str
    pipeline_version: str
    metric_specs: tuple[MetricSpec, ...]
    required_fields: tuple[str, ...]

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
        per_item=[dict(item) for item in scored.per_item],
    )


def write_structured_summary(result: EvaluationResult, cache_path: Path) -> None:
    """Write the registry result in one schema shared by all task families."""
    summary_path = cache_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict(include_per_item=True)
    payload["summary_version"] = 1
    payload["metric_values"] = {
        name: metric.value for name, metric in result.metrics.items()
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_per_item_results(result: EvaluationResult, cache_path: Path) -> None:
    """Persist detailed records so content-cache hits reproduce normal outputs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for item in result.per_item:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


@dataclass
class MathTask:
    """Adapter for math-verify scoring."""

    scorer: StructuredScorer
    family: str = "math_opensource"
    version: str = "math_v1"
    pipeline_version: str = "math_response_v1"
    metric_specs: tuple[MetricSpec, ...] = (MetricSpec("accuracy"),)
    required_fields: tuple[str, ...] = ("label", "response")

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
    required_fields: tuple[str, ...] = ("response",)

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]:
        has_logprobs = ["logprobs" in item for item in data]
        if all(has_logprobs):
            return [{**item, "task": context.task_name} for item in data]
        if any(has_logprobs):
            raise ValueError(
                "Mixed MC dataset detected: all items must use loglikelihood or generate schema"
            )
        return [
            _annotate_item(
                item, context.task_name, context.label_key, context.response_key
            )
            for item in data
        ]

    def score(self, context: EvaluationContext) -> EvaluationResult:
        has_logprobs = ["logprobs" in item for item in context.eval_dataset]
        if any(has_logprobs) and not all(has_logprobs):
            raise ValueError(
                "Mixed MC dataset detected: all items must use loglikelihood or generate schema"
            )
        is_loglikelihood = all(has_logprobs)
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
    required_fields: tuple[str, ...] = ("label", "response")

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
            allow_unsafe_code=context.allow_unsafe_code,
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
    """Evaluate through a registry and optionally use a content-addressed cache."""
    task = registry.resolve(context.task_name)
    cache: ContentAddressedCache | None = None
    cache_key: str | None = None
    if context.content_cache_dir is not None:
        cache = ContentAddressedCache(
            context.content_cache_dir,
            "evaluation",
            read_only=context.read_only_cache,
            force_recompute=context.force_recompute,
        )
        payload = {
            "model_name": context.model_name,
            "model_revision": context.model_revision,
            "task_name": context.task_name,
            "task_version": task.version,
            "eval_dataset": _evaluation_cache_inputs(context.eval_dataset),
            "postprocess_version": task.pipeline_version,
            "generation_params": {
                "label_key": context.label_key,
                "response_key": context.response_key,
                "mc_aggregation": context.mc_aggregation,
                "seed": context.seed,
            },
            "evaluation": {
                "max_workers": context.max_workers,
                "timeout": context.timeout,
                "exec_timeout": context.exec_timeout,
                "allow_unsafe_code": context.allow_unsafe_code,
                "pipeline_version": task.pipeline_version,
                "bootstrap_samples": context.bootstrap_samples,
                "confidence_level": context.confidence_level,
            },
        }
        cache_key = cache.key(payload)
        cached = cache.get(cache_key)
        if cached is not None:
            result = EvaluationResult.from_dict(cached)
            if (
                result.task_name == context.task_name
                and result.task_version == task.version
                and (result.per_item or result.sample_count == 0)
            ):
                write_per_item_results(result, context.cache_path)
                write_structured_summary(result, context.cache_path)
                return result

    result = task.score(context)
    result.cache_key = cache_key
    write_per_item_results(result, context.cache_path)
    write_structured_summary(result, context.cache_path)
    if cache is not None and cache_key is not None:
        cache.set(cache_key, result.to_dict(include_per_item=True))
    return result
