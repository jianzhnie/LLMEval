"""Regression tests for MC benchmark preparation."""

from scripts.data_process.prepare_mc_benchmarks import _format_mc_row


def test_mmlu_pro_builds_dynamic_prompt_without_fixed_template() -> None:
    result = _format_mc_row(
        "mmlu_pro",
        {
            "question": "Which option?",
            "options": ["first", "second", "third"],
            "answer": "B",
        },
        "test:0",
    )

    assert result["prompt"] == (
        "Which option?\nA. first\nB. second\nC. third\nAnswer:"
    )
    assert result["gold"] == 1
    assert result["answer"] == "B"
