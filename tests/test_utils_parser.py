"""Tests for llmeval.tasks.math_eval.utils_parser."""

from __future__ import annotations

import pytest

from llmeval.tasks.math_eval.utils_parser import parse_ground_truth


class TestParseGroundTruthGeneric:
    def test_simple_string(self) -> None:
        _, answer = parse_ground_truth({"answer": "42"}, "aime24")
        assert answer == "42"

    def test_strips_whitespace(self) -> None:
        _, answer = parse_ground_truth({"answer": "  100  "}, "math500")
        assert answer == "100"

    def test_numeric_answer(self) -> None:
        _, answer = parse_ground_truth({"answer": 204}, "aime24")
        assert answer == "204"

    def test_custom_label_key(self) -> None:
        _, answer = parse_ground_truth({"output": "99"}, "aime24", label_key="output")
        assert answer == "99"

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValueError, match=r"[Ee]mpty"):
            parse_ground_truth({"answer": ""}, "aime24")

    def test_none_answer_raises(self) -> None:
        with pytest.raises(ValueError, match=r"[Ee]mpty"):
            parse_ground_truth({"answer": None}, "aime24")

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            parse_ground_truth({"prompt": "q"}, "aime24", label_key="ans")


class TestParseGroundTruthGsm8k:
    def test_standard_format(self) -> None:
        cot, answer = parse_ground_truth({"answer": "2+2=4\n#### 4"}, "gsm8k")
        assert answer == "4"
        assert "2+2=4" in cot

    def test_missing_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="####"):
            parse_ground_truth({"answer": "no separator"}, "gsm8k")


class TestParseGroundTruthOlympiadbench:
    def test_list_answer(self) -> None:
        _, answer = parse_ground_truth({"answer": ["42"]}, "olympiadbench")
        assert answer == "42"

    def test_string_answer(self) -> None:
        _, answer = parse_ground_truth({"answer": "$x^2$"}, "olympiadbench")
        assert answer == "x^2"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match=r"[Ee]mpty"):
            parse_ground_truth({"answer": []}, "olympiadbench")


class TestParseGroundTruthValidation:
    def test_non_dict_raises(self) -> None:
        with pytest.raises(TypeError, match="dictionary"):
            parse_ground_truth("not a dict", "aime24")

    def test_non_string_data_name_raises(self) -> None:
        with pytest.raises(TypeError, match="string"):
            parse_ground_truth({"answer": "1"}, 123)

    def test_non_string_label_key_raises(self) -> None:
        with pytest.raises(TypeError, match="string"):
            parse_ground_truth({"answer": "1"}, "aime24", label_key=42)
