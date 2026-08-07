"""Tests for llmeval.utils.prompts."""

from __future__ import annotations

import pytest

from llmeval.utils.prompts import (
    SYSTEM_PROMPT_FACTORY,
    is_chat_template_applied,
)


class TestSystemPromptFactory:
    def test_all_keys_are_strings(self) -> None:
        for key in SYSTEM_PROMPT_FACTORY:
            assert isinstance(key, str)

    def test_known_keys_exist(self) -> None:
        for key in ("deepseek_r1", "amthinking", "openr1", "default", "empty"):
            assert key in SYSTEM_PROMPT_FACTORY

    def test_empty_key_returns_none(self) -> None:
        assert SYSTEM_PROMPT_FACTORY["empty"] is None

    def test_default_is_non_empty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT_FACTORY["default"], str)
        assert len(SYSTEM_PROMPT_FACTORY["default"]) > 0


class TestIsChatTemplateApplied:
    @pytest.mark.parametrize(
        "query",
        [
            "<|im_start|>user\nHello",
            "<s>[INST] hello",
            "### Human:\nWhat?",
            "Human: hi\nAssistant: hello",
        ],
    )
    def test_detects_known_markers(self, query: str) -> None:
        assert is_chat_template_applied(query) is True

    def test_plain_text_returns_false(self) -> None:
        assert is_chat_template_applied("What is 2+2?") is False

    def test_human_word_inside_plain_text_returns_false(self) -> None:
        query = "In this passage, the label Human: appears as ordinary text."
        assert is_chat_template_applied(query) is False

    def test_single_human_role_line_is_raw_dialogue(self) -> None:
        assert is_chat_template_applied("Human: hi") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_chat_template_applied("") is False

    @pytest.mark.parametrize(
        "query",
        [
            "Use <sub>2</sub> for subscripts in chemistry.",
            "E = mc<sup>2</sup> is famous.",
            "A <span> tag wraps inline text.",
            "The <strong> tag marks important text.",
        ],
    )
    def test_html_s_prefixed_tags_return_false(self, query: str) -> None:
        assert is_chat_template_applied(query) is False

    @pytest.mark.parametrize(
        "query",
        [
            "<s>",
            "</s>",
            "<s>\n[INST] hello [/INST]",
            "prefix <s> suffix",
        ],
    )
    def test_complete_s_tags_return_true(self, query: str) -> None:
        assert is_chat_template_applied(query) is True
