"""Task registry and structured adapters for evaluation task families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llmeval.tasks.persistence import atomic_write_json, atomic_write_jsonl
from llmeval.tasks.results import (
    EvaluationResult,
    MetricValue,
    ScorerResult,
    metric_from_samples,
)
from llmeval.utils.log import init_logger

logger = init_logger("task_registry")

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
    input_key: str = "prompt"
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


def write_structured_summary(result: EvaluationResult, cache_path: Path) -> None:
    """Write the registry result in one schema shared by all task families."""
    summary_path = cache_path.with_suffix(".summary.json")
    payload = result.to_dict(include_per_item=True)
    payload["summary_version"] = 1
    metric_values = {name: metric.value for name, metric in result.metrics.items()}
    payload["metric_values"] = metric_values
    for name, value in metric_values.items():
        payload.setdefault(name, value)
    atomic_write_json(summary_path, payload, indent=2)


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


def write_per_item_results(
    result: EvaluationResult, cache_path: Path, *, output_schema: str = "compact"
) -> None:
    """Persist per-item records using the requested compact/debug schema."""
    if output_schema not in {"compact", "debug"}:
        raise ValueError("output_schema must be 'compact' or 'debug'")
    records = (
        item if output_schema == "debug" else _compact_per_item(item)
        for item in result.per_item
    )
    atomic_write_jsonl(cache_path, records)


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
    write_per_item_results(
        result, context.cache_path, output_schema=context.output_schema
    )
    write_structured_summary(result, context.cache_path)
    return result
