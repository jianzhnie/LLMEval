"""Tests for llmeval.tasks.postprocess."""

from __future__ import annotations

from llmeval.tasks.postprocess import (
    apply_text_pipeline,
    build_text_pipeline,
    strip_reasoning_wrappers,
)


def test_strip_reasoning_wrappers_prefers_answer_tag() -> None:
    text = "<think>reasoning</think><answer> 42 </answer> tail"
    assert strip_reasoning_wrappers(text) == "42"


def test_apply_text_pipeline_runs_in_order() -> None:
    def add_a(text: str) -> str:
        return text + "a"

    def add_b(text: str) -> str:
        return text + "b"

    pipeline = build_text_pipeline(add_a, add_b)
    assert apply_text_pipeline("x", pipeline) == "xab"


def test_apply_text_pipeline_handles_none() -> None:
    assert (
        apply_text_pipeline(None, build_text_pipeline(strip_reasoning_wrappers)) == ""
    )
