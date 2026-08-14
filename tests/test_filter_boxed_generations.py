"""Tests for boxed-generation token filtering."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.data_process.filter_boxed_generations import compute_token_lengths


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.generation_call: tuple[str, dict[str, Any]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        self.messages = messages
        return [1, 2, 3, 4, 5]

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.generation_call = (text, kwargs)
        return {"input_ids": [6, 7, 8]}


def test_compute_token_lengths_uses_inference_chat_template() -> None:
    tokenizer = FakeTokenizer()

    result = compute_token_lengths(
        {"prompt": "What is 1 + 1?", "gen": "The answer is \\boxed{2}."},
        tokenizer,  # type: ignore[arg-type]
        "System instructions",
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "What is 1 + 1?"},
    ]
    assert tokenizer.generation_call == (
        "The answer is \\boxed{2}.",
        {
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        },
    )
    assert result == {
        "prompt_length": 5,
        "gen_length": 3,
        "total_token_length": 8,
        "has_boxed": True,
    }


def test_compute_token_lengths_rejects_multiple_generations() -> None:
    with pytest.raises(ValueError, match="contains 2 generations"):
        compute_token_lengths(
            {"prompt": "Question", "gen": ["first", "second"]},
            FakeTokenizer(),  # type: ignore[arg-type]
            None,
        )


def test_compute_token_lengths_accepts_singleton_generation_list() -> None:
    result = compute_token_lengths(
        {"prompt": ["Question"], "gen": ["\\boxed{1}"]},
        FakeTokenizer(),  # type: ignore[arg-type]
        None,
    )

    assert result["total_token_length"] == 8
    assert result["has_boxed"] is True
