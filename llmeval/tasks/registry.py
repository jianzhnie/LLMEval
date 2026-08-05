"""Task registry and structured adapters for the three evaluation families.

The registry owns task-family discovery and task metadata. Scorers retain their
existing public APIs; adapters translate their legacy JSONL/summary output into
the common :class:`EvaluationResult` contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llmeval.cache import ContentAddressedCache
from llmeval.tasks.provenance import (
    build_run_provenance,
    get_git_hash,
    hash_evaluation_inputs,
)
from llmeval.tasks.results import EvaluationResult, MetricValue, metric_from_samples

MathScorer = Callable[..., float]
MCScorer = Callable[..., float]
CodeScorer = Callable[..., float]


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

    def prepare_dataset(
        self, data: list[dict[str, Any]], context: PreparationContext
    ) -> list[dict[str, Any]]: ...

    def score(self, context: EvaluationContext) -> EvaluationResult: ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _read_summary(cache_path: Path) -> dict[str, Any]:
    summary_path = cache_path.with_suffix(".summary.json")
    if not summary_path.exists():
        return {}
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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


def _base_provenance(
    context: EvaluationContext, task: EvaluationTask
) -> dict[str, Any]:
    provenance = build_run_provenance(
        context.eval_dataset,
        task_name=context.task_name,
        input_key=context.input_key,
        label_key=context.label_key,
        response_key=context.response_key,
        seed=context.seed,
    )
    provenance.update(
        {
            "task_registry_family": task.family,
            "task_registry_version": task.version,
            "pipeline_version": task.pipeline_version,
            "model_name": context.model_name,
            "model_revision": context.model_revision,
        }
    )
    return provenance


@dataclass
class MathTask:
    """Adapter for math-verify scoring."""

    scorer: MathScorer
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
        value = self.scorer(
            eval_dataset=context.eval_dataset,
            label_key=context.label_key,
            response_key=context.response_key,
            cache_path=str(context.cache_path),
            max_workers=context.max_workers,
            timeout=context.timeout,
            task_name=context.task_name,
            seed=context.seed,
        )
        records = [
            float(record.get("accuracy", value))
            for record in context.eval_dataset
            if isinstance(record.get("accuracy", value), (int, float))
        ]
        samples = records or [float(value)]
        return EvaluationResult(
            task_name=context.task_name,
            task_version=self.version,
            metrics={"accuracy": _metric(samples, context)},
            sample_count=len(samples),
            effective_sample_count=sum(1 for sample in samples if sample in {0.0, 1.0}),
            failed_count=sum(
                1
                for record in context.eval_dataset
                if record.get("extracted_answer") == "Error"
            ),
            timeout_count=sum(
                1
                for record in context.eval_dataset
                if record.get("extracted_answer") == "Timeout"
            ),
            provenance=_base_provenance(context, self),
        )


@dataclass
class MCTask:
    """Adapter for generation and loglikelihood MC scoring."""

    generate_scorer: MCScorer
    loglikelihood_scorer: MCScorer
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
            "task_name": context.task_name,
            "seed": context.seed,
        }
        if not is_loglikelihood:
            kwargs.update(
                label_key=context.label_key,
                response_key=context.response_key,
                aggregation=context.mc_aggregation,
            )
        value = scorer(**kwargs)
        records = _read_jsonl(context.cache_path)
        summary = _read_summary(context.cache_path)
        metric_samples: dict[str, list[float]] = {
            name: [] for name in ("acc", "acc_norm", "acc_bytes", "exact_match")
        }
        for record in records:
            if record.get("aggregation") == "per_sample" and isinstance(
                record.get("sample_correct"), list
            ):
                samples = [float(bool(item)) for item in record["sample_correct"]]
                metric_samples["acc"].extend(samples)
                metric_samples["acc_norm"].extend(samples)
                metric_samples["acc_bytes"].extend(samples)
                metric_samples["exact_match"].extend(samples)
            else:
                for name, key in (
                    ("acc", "correct"),
                    ("acc_norm", "correct_norm"),
                    ("acc_bytes", "correct_bytes"),
                    ("exact_match", "correct"),
                ):
                    metric_samples[name].append(float(bool(record.get(key, False))))
        metrics: dict[str, MetricValue] = {}
        for name in metric_samples:
            samples = metric_samples[name]
            if not samples:
                samples = [float(summary.get(name, value if name == "acc" else 0.0))]
            metrics[name] = _metric(samples, context)
        return EvaluationResult(
            task_name=context.task_name,
            task_version=self.version,
            metrics=metrics,
            sample_count=metrics["acc"].count,
            effective_sample_count=metrics["acc"].count,
            failed_count=sum(1 for record in records if record.get("error")),
            provenance={
                **_base_provenance(context, self),
                "mc_mode": "loglikelihood" if is_loglikelihood else "generate",
                "mc_aggregation": context.mc_aggregation,
            },
        )


@dataclass
class CodeTask:
    """Adapter for sandboxed code scoring."""

    scorer: CodeScorer
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
        value = self.scorer(
            eval_dataset=context.eval_dataset,
            label_key=context.label_key,
            response_key=context.response_key,
            cache_path=context.cache_path,
            max_workers=context.max_workers,
            timeout=context.timeout,
            exec_timeout=context.exec_timeout,
            task_name=context.task_name,
            seed=context.seed,
            allow_unsafe_code=context.allow_unsafe_code,
        )
        records = _read_jsonl(context.cache_path)
        samples = [float(bool(record.get("passed"))) for record in records]
        if not samples:
            samples = [float(value)]
        summary = _read_summary(context.cache_path)
        metrics = {"pass@1": _metric(samples, context)}
        for name, raw in summary.get("pass_at_k", {}).items():
            if name != "pass@1" and isinstance(raw, (int, float)):
                metrics[name] = MetricValue(
                    float(raw), int(summary.get("problems", len(samples)))
                )
        return EvaluationResult(
            task_name=context.task_name,
            task_version=self.version,
            metrics=metrics,
            sample_count=len(samples),
            effective_sample_count=len(samples),
            failed_count=len(samples) - sum(int(sample) for sample in samples),
            provenance=_base_provenance(context, self),
        )


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
    math_scorer: MathScorer,
    mc_generate_scorer: MCScorer,
    mc_loglikelihood_scorer: MCScorer,
    code_scorer: CodeScorer,
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
        run_provenance = build_run_provenance(
            context.eval_dataset,
            task_name=context.task_name,
            input_key=context.input_key,
            label_key=context.label_key,
            response_key=context.response_key,
        )
        payload = {
            "model_name": context.model_name,
            "model_revision": context.model_revision,
            "task_name": context.task_name,
            "task_version": task.version,
            "evaluation_input_hash": hash_evaluation_inputs(
                context.eval_dataset, context.response_key
            ),
            "git_hash": get_git_hash(),
            "generation": {
                "label_key": context.label_key,
                "response_key": context.response_key,
                "mc_aggregation": context.mc_aggregation,
                "seed": context.seed,
                "prompt_hash": run_provenance.get("prompt_hash"),
                "target_hash": run_provenance.get("target_hash"),
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
            ):
                return result

    result = task.score(context)
    result.cache_key = cache_key
    if cache is not None and cache_key is not None:
        cache.set(cache_key, result.to_dict())
    return result
