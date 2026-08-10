"""Tests for backend-independent inference helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmeval.inference.common import (
    ResumeState,
    get_request_seed,
    is_explicit_tool_choice,
    load_jsonl,
    load_resume_state,
    prepare_sample_requests,
    redact_config_for_logging,
    save_failed_items,
)


def _resume(
    *,
    completed_counts: dict[str, int] | None = None,
    prompts: dict[str, str] | None = None,
) -> ResumeState:
    return ResumeState(
        completed_counts=completed_counts or {},
        prompts=prompts or {},
    )


class TestLoadJsonl:
    def test_parses_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "in.jsonl"
        path.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
        assert load_jsonl(path) == [{"a": 1}, {"a": 2}]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "missing.jsonl")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")
        with pytest.raises(json.JSONDecodeError):
            load_jsonl(path)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "scalar.jsonl"
        path.write_text("[1, 2]\n")
        with pytest.raises(ValueError, match="must contain an object"):
            load_jsonl(path)


class TestResumeState:
    def test_missing_and_empty_files_return_zero(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.jsonl"
        assert load_resume_state(missing, "prompt", "gen").completed_count == 0
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        assert load_resume_state(empty, "prompt", "gen").completed_count == 0

    def test_counts_completed_rows_by_document(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        rows = [
            {"doc_id": "q1", "prompt": "p", "gen": ["a"]},
            {"doc_id": "q1", "prompt": "p", "gen": ["b"]},
            {"doc_id": "q2", "prompt": "p", "gen": ["c"]},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        state = load_resume_state(path, "prompt", "gen")

        assert state.completed_counts == {"q1": 2, "q2": 1}
        assert state.completed_count == 3

    def test_missing_or_null_generation_is_not_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        rows = [
            {"doc_id": "q1", "prompt": "p"},
            {"doc_id": "q2", "prompt": "p", "gen": None},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assert load_resume_state(path, "prompt", "gen").completed_count == 0

    def test_logprobs_row_is_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "logprobs": [0.0]}) + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_counts == {"q1": 1}

    def test_logprobs_with_null_missing_choice_is_completed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "logprobs": [None, -0.5]}) + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_counts == {"q1": 1}

    @pytest.mark.parametrize(
        "logprobs",
        [
            [],  # empty list: no scores recorded
            "not-a-list",  # input data carrying an unrelated logprobs field
            0.5,  # bare number, not a per-choice list
            ["A", "B"],  # non-numeric elements
            None,  # explicit null
            [None, None],  # no choice received a finite score
            [float("nan")],  # non-standard/non-finite JSON number
            [float("inf")],  # non-standard/non-finite JSON number
        ],
    )
    def test_non_score_logprobs_row_is_not_completed(
        self, tmp_path: Path, logprobs: object
    ) -> None:
        """Only a non-empty numeric logprobs list marks a row completed."""
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "logprobs": logprobs}) + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_count == 0

    def test_context_length_error_row_is_completed(self, tmp_path: Path) -> None:
        """A permanent-failure row (empty response + error marker) is completed."""
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "gen": "",
                    "error": "context_length_exceeded",
                }
            )
            + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_counts == {"q1": 1}

    def test_empty_response_error_row_is_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "gen": "",
                    "error": "empty_response",
                }
            )
            + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_counts == {"q1": 1}

    def test_malformed_line_reports_path_and_line(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text("bad json\n")
        with pytest.raises(ValueError, match=r"out\.jsonl at line 1"):
            load_resume_state(path, "prompt", "gen")

    def test_repair_ignores_only_truncated_final_line(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "gen": ["a"]}) + "\n{"
        )
        state = load_resume_state(
            path, "prompt", "gen", repair_truncated_last_line=True
        )
        assert state.completed_counts == {"q1": 1}

    @pytest.mark.parametrize(
        "content",
        [
            "bad json\n" + json.dumps({"doc_id": "q1", "gen": ["a"]}) + "\n",
            "bad json\n",
        ],
    )
    def test_repair_rejects_non_truncated_invalid_lines(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(content)

        with pytest.raises(ValueError, match=r"out\.jsonl at line 1"):
            load_resume_state(path, "prompt", "gen", repair_truncated_last_line=True)

    def test_custom_response_key_uses_gen_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        rows = [
            {"doc_id": "q1", "question": "p", "response": ["a"]},
            {"doc_id": "q2", "question": "p", "gen": ["b"]},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assert load_resume_state(path, "question", "response").completed_counts == {
            "q1": 1,
            "q2": 1,
        }

    def test_completed_row_requires_document_id(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"prompt": "p", "gen": ["a"]}) + "\n")
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            load_resume_state(path, "prompt", "gen")

    def test_grouped_generation_row_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "gen": ["a", "b"]}) + "\n"
        )
        with pytest.raises(ValueError, match="exactly one generation"):
            load_resume_state(path, "prompt", "gen")

    def test_conflicting_prompts_for_document_are_rejected(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.jsonl"
        rows = [
            {"doc_id": "q1", "prompt": "old", "gen": ["a"]},
            {"doc_id": "q1", "prompt": "new", "gen": ["b"]},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match="conflicting prompts"):
            load_resume_state(path, "prompt", "gen")

    def test_internal_resume_fields_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "gen": ["a"],
                    "_llmeval_legacy": True,
                }
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="unsupported internal fields"):
            load_resume_state(path, "prompt", "gen")


class TestExpansion:
    def test_copies_each_document_requested_number_of_times(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = prepare_sample_requests(raw, _resume(), "prompt", 3, base_seed=123)
        assert len(expanded) == 3
        assert len({get_request_seed(item) for item in expanded}) == 3

    def test_resume_uses_completed_count(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = prepare_sample_requests(
            raw,
            _resume(completed_counts={"q1": 2}),
            "prompt",
            4,
            base_seed=123,
        )
        assert len(expanded) == 2

        full = prepare_sample_requests(raw, _resume(), "prompt", 4, base_seed=123)
        assert [get_request_seed(item) for item in expanded] == [
            get_request_seed(item) for item in full[2:]
        ]

    def test_all_completed_yields_nothing(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        assert (
            prepare_sample_requests(
                raw,
                _resume(completed_counts={"q1": 2}),
                "prompt",
                2,
                base_seed=1,
            )
            == []
        )

    def test_too_many_completed_rows_is_rejected(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        with pytest.raises(ValueError, match="exceeding requested"):
            prepare_sample_requests(
                raw,
                _resume(completed_counts={"q1": 3}),
                "prompt",
                2,
                base_seed=1,
            )

    @pytest.mark.parametrize("count", [0, -1])
    def test_rejects_non_positive_generation_count(self, count: int) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": "p"}],
                _resume(),
                "prompt",
                count,
                base_seed=1,
            )

    def test_requires_document_id_and_prompt(self) -> None:
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            prepare_sample_requests(
                [{"prompt": "p"}], _resume(), "prompt", 1, base_seed=1
            )
        with pytest.raises(ValueError, match="non-empty string prompt"):
            prepare_sample_requests(
                [{"doc_id": "q1"}], _resume(), "prompt", 1, base_seed=1
            )

    @pytest.mark.parametrize("prompt", [{"text": "p"}, ["p"], 42])
    def test_rejects_non_string_prompts(self, prompt: object) -> None:
        with pytest.raises(ValueError, match="non-empty string prompt"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": prompt}],
                _resume(),
                "prompt",
                1,
                base_seed=1,
            )

    def test_duplicate_document_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate doc_id"):
            prepare_sample_requests(
                [
                    {"doc_id": "q1", "prompt": "first"},
                    {"doc_id": "q1", "prompt": "second"},
                ],
                _resume(),
                "prompt",
                1,
                base_seed=1,
            )

    def test_base_seed_must_be_a_non_negative_integer(self) -> None:
        with pytest.raises(ValueError, match="base_seed must be non-negative"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": "p"}],
                _resume(),
                "prompt",
                1,
                base_seed=-1,
            )

    def test_changed_prompt_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="changed prompt"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": "new"}],
                _resume(completed_counts={"q1": 1}, prompts={"q1": "old"}),
                "prompt",
                2,
                base_seed=1,
            )

    def test_copies_are_independent(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p", "gen": ["existing"]}]
        expanded = prepare_sample_requests(raw, _resume(), "prompt", 2, base_seed=1)
        expanded[0]["gen"].append("new")
        assert expanded[1]["gen"] == ["existing"]
        assert raw[0]["gen"] == ["existing"]


class TestUtilities:
    def test_redacts_nested_credentials_without_mutating_input(self) -> None:
        payload = {
            "api_key": "secret",
            "nested": {"Authorization": "Bearer secret", "value": 1},
            "cookie": "session=secret",
            "extra_body": '{"api_key": "nested-secret", "top_k": 40}',
        }
        redacted = redact_config_for_logging(payload)
        assert redacted == {
            "api_key": "***",
            "nested": {"Authorization": "***", "value": 1},
            "cookie": "***",
            "extra_body": {"api_key": "***", "top_k": 40},
        }
        assert payload["api_key"] == "secret"

    def test_tool_choice_detection(self) -> None:
        assert not is_explicit_tool_choice(None)
        assert is_explicit_tool_choice("auto")


class TestSaveFailedItems:
    def test_appends_without_losing_previous_failures(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        save_failed_items(output, [{"doc_id": "q1", "error": "first"}])
        save_failed_items(output, [{"doc_id": "q2", "error": "second"}])

        failed = tmp_path / "output_failed.jsonl"
        records = [json.loads(line) for line in failed.read_text().splitlines()]
        assert [record["doc_id"] for record in records] == ["q1", "q2"]
        assert all(record["run_id"] and record["failure_id"] for record in records)

    def test_repeated_failures_remain_append_only(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        entry = {"doc_id": "q1", "error": "transient"}
        save_failed_items(output, [entry], run_id="run-1")
        save_failed_items(output, [entry], run_id="run-2")

        records = [
            json.loads(line)
            for line in (tmp_path / "output_failed.jsonl").read_text().splitlines()
        ]
        assert [record["run_id"] for record in records] == ["run-1", "run-2"]
        assert records[0]["failure_id"] == records[1]["failure_id"]

    def test_write_failure_propagates(self, tmp_path: Path) -> None:
        output = tmp_path / "directory.jsonl"
        (tmp_path / "directory_failed.jsonl").mkdir()
        with pytest.raises(OSError):
            save_failed_items(output, [{"error": "boom"}])
