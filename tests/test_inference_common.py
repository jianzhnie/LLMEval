"""Tests for llmeval.inference.common shared helpers.

Covers the data-loading / resume helpers used by the online, offline,
verifier, and MC runners — including the gen:null regression (a
partially-written output line must not crash the resume count).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmeval.inference.common import (
    ResumeState,
    expand_data_with_resume,
    is_explicit_tool_choice,
    load_jsonl,
    load_resume_state,
    redact_config_for_logging,
    sample_seed_for_item,
    save_failed_items,
)


def _resume(
    *,
    completed_indices: dict[str, set[int]] | None = None,
    prompts: dict[str, str] | None = None,
) -> ResumeState:
    """Build a ResumeState from doc_id-keyed completed indices."""
    return ResumeState(
        completed_indices=completed_indices or {},
        prompts=prompts or {},
    )


class TestLoadJsonl:
    def test_parses_and_skips_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "in.jsonl"
        f.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
        assert load_jsonl(f) == [{"a": 1}, {"a": 2}]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n")
        with pytest.raises(json.JSONDecodeError):
            load_jsonl(f)

    def test_non_object_json_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "scalar.jsonl"
        f.write_text("[1, 2, 3]\n")
        with pytest.raises(ValueError, match="must contain an object"):
            load_jsonl(f)


class TestCountCompletedSamples:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert (
            load_resume_state(tmp_path / "none.jsonl", "prompt", "gen").completed_count
            == 0
        )

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text("")
        assert load_resume_state(f, "prompt", "gen").completed_count == 0

    def test_counts_one_row_per_sample(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(
                json.dumps(
                    {"doc_id": "q1", "prompt": "p", "gen": ["a"], "sample_index": 0}
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {"doc_id": "q1", "prompt": "p", "gen": ["b"], "sample_index": 1}
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {"doc_id": "q2", "prompt": "p", "gen": ["c"], "sample_index": 0}
                )
                + "\n"
            )
        state = load_resume_state(f, "prompt", "gen")
        assert state.completed_indices == {"q1": {0, 1}, "q2": {0}}
        assert state.completed_count == 3

    def test_incomplete_row_not_counted(self, tmp_path: Path) -> None:
        """A completed-looking row without sample_index is a protocol error."""
        f = tmp_path / "out.jsonl"
        f.write_text(json.dumps({"doc_id": "q1", "prompt": "p", "gen": ["a"]}) + "\n")

        with pytest.raises(ValueError, match="sample_index must be"):
            load_resume_state(f, "prompt", "gen")

    def test_row_without_gen_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "sample_index": 0}) + "\n"
        )
        assert load_resume_state(f, "prompt", "gen").completed_count == 0

    def test_null_gen_ignored(self, tmp_path: Path) -> None:
        """Regression: a partially-written line with gen:null is not completed."""
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(
                json.dumps(
                    {"doc_id": "q1", "prompt": "p", "gen": None, "sample_index": 0}
                )
                + "\n"
            )
        assert load_resume_state(f, "prompt", "gen").completed_count == 0

    def test_verifier_response_counts_completed(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps(
                {
                    "doc_id": "d1",
                    "prompt": "p",
                    "Verifier_response": "yes",
                    "sample_index": 0,
                }
            )
            + "\n"
        )
        assert load_resume_state(f, "prompt", "gen").completed_count == 1

    def test_logprobs_row_counts_completed(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps(
                {
                    "doc_id": "d1",
                    "prompt": "p",
                    "logprobs": [1.0, 0.0],
                    "sample_index": 0,
                }
            )
            + "\n"
        )
        assert load_resume_state(f, "prompt", "gen").completed_count == 1

    def test_malformed_line_fails_with_path_and_line(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write("bad json\n")
            fh.write(
                json.dumps(
                    {"doc_id": "q1", "prompt": "p", "gen": ["a"], "sample_index": 0}
                )
                + "\n"
            )
        with pytest.raises(ValueError, match=r"out\.jsonl at line 1"):
            load_resume_state(f, "prompt", "gen")

    def test_non_object_line_fails_with_path_and_line(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            "[1, 2]\n"
            + json.dumps(
                {"doc_id": "q1", "prompt": "p", "gen": ["a"], "sample_index": 0}
            )
            + "\n"
        )
        with pytest.raises(
            ValueError, match=r"out\.jsonl line 1 must contain an object"
        ):
            load_resume_state(f, "prompt", "gen")

    def test_repair_ignores_only_unterminated_final_json(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "gen": ["a"], "sample_index": 0})
            + "\n{"
        )

        state = load_resume_state(
            f,
            "prompt",
            "gen",
            repair_truncated_last_line=True,
        )

        assert state.completed_indices == {"q1": {0}}

    def test_repair_does_not_ignore_malformed_middle_line(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "gen": ["a"], "sample_index": 0})
            + "\n{\n"
            + json.dumps(
                {"doc_id": "q2", "prompt": "p", "gen": ["b"], "sample_index": 0}
            )
        )

        with pytest.raises(ValueError, match="at line 2"):
            load_resume_state(
                f,
                "prompt",
                "gen",
                repair_truncated_last_line=True,
            )

    def test_custom_response_key_with_gen_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "doc_id": "q1",
                        "question": "p",
                        "response": ["a"],
                        "sample_index": 0,
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {"doc_id": "q2", "question": "p", "gen": ["b"], "sample_index": 0}
                )
                + "\n"
            )
        state = load_resume_state(f, "question", "response")
        assert state.completed_indices == {"q1": {0}, "q2": {0}}

    def test_row_without_doc_id_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps({"prompt": "legacy", "gen": ["b"], "sample_index": 0}) + "\n"
        )

        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            load_resume_state(output, "prompt", "gen")

    def test_multigen_row_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps(
                {"doc_id": "q1", "prompt": "p", "gen": ["a", "b"], "sample_index": 0}
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="exactly one generation"):
            load_resume_state(output, "prompt", "gen")

    def test_duplicate_sample_index_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "gen": ["a"], "sample_index": 0})
            + "\n"
            + json.dumps(
                {"doc_id": "q1", "prompt": "p", "gen": ["b"], "sample_index": 0}
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="duplicates"):
            load_resume_state(output, "prompt", "gen")

    def test_conflicting_prompts_for_doc_id_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps(
                {"doc_id": "q1", "prompt": "old", "gen": ["a"], "sample_index": 0}
            )
            + "\n"
            + json.dumps(
                {"doc_id": "q1", "prompt": "new", "gen": ["b"], "sample_index": 1}
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="conflicting prompts"):
            load_resume_state(output, "prompt", "gen")


class TestExpandDataWithResume:
    def test_expands_all_samples(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = expand_data_with_resume(raw, _resume(), "prompt", 3)
        assert [item["sample_index"] for item in expanded] == [0, 1, 2]

    def test_regenerates_only_missing_indices(self) -> None:
        """A mid-run failure must be retried, not the highest contiguous count."""
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = expand_data_with_resume(
            raw, _resume(completed_indices={"q1": {0, 2}}), "prompt", 4
        )
        assert [item["sample_index"] for item in expanded] == [1, 3]

    def test_all_completed_yields_nothing(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        assert (
            expand_data_with_resume(
                raw, _resume(completed_indices={"q1": {0, 1}}), "prompt", 2
            )
            == []
        )

    def test_rejects_non_positive_sample_count(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            expand_data_with_resume(
                [{"doc_id": "q1", "prompt": "q"}], _resume(), "prompt", 0
            )

    def test_requires_document_id(self) -> None:
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            expand_data_with_resume([{"prompt": "p"}], _resume(), "prompt", 1)

    def test_missing_prompt_is_schema_error(self) -> None:
        with pytest.raises(ValueError, match="no non-empty prompt"):
            expand_data_with_resume([{"doc_id": "q1"}], _resume(), "prompt", 1)

    def test_changed_prompt_for_doc_id_is_rejected(self) -> None:
        """A resumed doc_id whose prompt changed must fail loudly."""
        raw = [{"doc_id": "q1", "prompt": "new"}]
        state = _resume(completed_indices={"q1": {0}}, prompts={"q1": "old"})
        with pytest.raises(ValueError, match="changed prompt"):
            expand_data_with_resume(raw, state, "prompt", 2)


class TestExpandedSamples:
    def test_offline_adapter_keeps_independent_sample_copies(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p", "gen": ["existing"]}]
        expanded = expand_data_with_resume(raw, _resume(), "prompt", 2)

        expanded[0]["gen"].append("new")
        assert [item["sample_index"] for item in expanded] == [0, 1]
        assert expanded[1]["gen"] == ["existing"]
        assert raw[0]["gen"] == ["existing"]


class TestLoggingRedaction:
    def test_redacts_nested_credentials_without_mutating_input(self) -> None:
        payload = {
            "api_key": "secret",
            "nested": {"Authorization": "Bearer secret", "value": 1},
            "cookie": "session=secret",
        }
        redacted = redact_config_for_logging(payload)
        assert redacted == {
            "api_key": "***",
            "nested": {"Authorization": "***", "value": 1},
            "cookie": "***",
        }
        assert payload["api_key"] == "secret"


class TestSampleSeed:
    def test_seed_is_stable_for_same_item(self) -> None:
        item = {
            "doc_id": "doc:1",
            "prompt": "What is 2+2?",
            "sample_index": 2,
        }
        assert sample_seed_for_item(123, item) == sample_seed_for_item(123, item)

    def test_seed_changes_with_sample_index_and_document_id(self) -> None:
        item = {"doc_id": "doc:1", "prompt": "q", "sample_index": 0}
        next_sample = {**item, "sample_index": 1}
        other_document = {**item, "doc_id": "doc:2"}
        assert sample_seed_for_item(123, item) != sample_seed_for_item(123, next_sample)
        assert sample_seed_for_item(123, item) != sample_seed_for_item(
            123, other_document
        )

    def test_negative_base_seed_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            sample_seed_for_item(-1, {"prompt": "q"})


class TestStableResumeCounts:
    def test_single_row_honors_scalar_sample_index(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps({"doc_id": "q1", "prompt": "q", "gen": ["a"], "sample_index": 2})
            + "\n"
        )

        state = load_resume_state(output, "prompt", "gen")
        assert state.completed_indices == {"q1": {2}}
        assert state.prompts == {"q1": "q"}

    def test_internal_resume_fields_are_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "q",
                    "gen": ["a"],
                    "sample_index": 0,
                    "_llmeval_legacy": [1, 3],
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="unsupported internal fields"):
            load_resume_state(output, "prompt", "gen")

    def test_multi_generation_row_is_rejected(self, tmp_path: Path) -> None:
        """Grouped rows must be migrated to one row per sample."""
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {"doc_id": "q1", "prompt": "q", "gen": ["a", "b"], "sample_index": 0}
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="exactly one generation"):
            load_resume_state(output, "prompt", "gen")

    def test_negative_scalar_sample_index_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {"doc_id": "q1", "prompt": "q", "gen": ["a"], "sample_index": -1}
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="non-negative"):
            load_resume_state(output, "prompt", "gen")

    def test_non_integer_scalar_sample_index_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {"doc_id": "q1", "prompt": "q", "gen": ["a"], "sample_index": "0"}
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="non-negative int"):
            load_resume_state(output, "prompt", "gen")

    def test_row_without_doc_id_rejected(self, tmp_path: Path) -> None:
        """Legacy rows without doc_id fail loudly and tell the user to migrate."""
        output = tmp_path / "output.jsonl"
        output.write_text(json.dumps({"prompt": "q", "gen": ["a"]}) + "\n")

        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            load_resume_state(output, "prompt", "gen")


class TestToolChoiceHelper:
    def test_none_is_not_explicit(self) -> None:
        assert not is_explicit_tool_choice("none")
        assert not is_explicit_tool_choice("")

    def test_auto_is_explicit(self) -> None:
        assert is_explicit_tool_choice("auto")
        assert is_explicit_tool_choice("my_tool")


class TestSaveFailedItems:
    def test_writes_failed_file_next_to_output(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        save_failed_items(out, [{"prompt": "q", "error": "boom"}])
        failed = tmp_path / "output_failed.jsonl"
        assert failed.exists()
        record = json.loads(failed.read_text().strip())
        assert record["prompt"] == "q"
        assert record["error"] == "boom"
        assert record["run_id"]
        assert record["failure_id"]

    def test_non_jsonl_output_name_not_clobbered(self, tmp_path: Path) -> None:
        """splitext-derived name must not collapse onto the output file."""
        out = tmp_path / "output"
        out.write_text('{"prompt": "q", "gen": ["a"]}\n')
        save_failed_items(out, [{"prompt": "q", "error": "boom"}])
        assert (tmp_path / "output_failed.jsonl").exists()
        # Output file content untouched (not truncated by "w" mode)
        assert out.read_text() == '{"prompt": "q", "gen": ["a"]}\n'

    def test_nested_output_path(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "output.jsonl"
        out.parent.mkdir(parents=True)
        save_failed_items(out, [{"error": "x"}])
        assert (tmp_path / "sub" / "dir" / "output_failed.jsonl").exists()

    def test_resume_appends_without_losing_previous_failures(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output.jsonl"
        save_failed_items(out, [{"sample_index": 0, "error": "first"}])
        save_failed_items(out, [{"sample_index": 2, "error": "second"}])

        failed = tmp_path / "output_failed.jsonl"
        records = [json.loads(line) for line in failed.read_text().splitlines()]
        assert [record["sample_index"] for record in records] == [0, 2]
        assert all(record["run_id"] and record["failure_id"] for record in records)

    def test_append_only_keeps_repeated_failure_audits(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        entry = {"doc_id": "d1", "sample_index": 0, "error": "transient"}
        save_failed_items(out, [entry], run_id="run-1")
        save_failed_items(out, [entry], run_id="run-2")

        records = [
            json.loads(line)
            for line in (tmp_path / "output_failed.jsonl").read_text().splitlines()
        ]
        assert [record["run_id"] for record in records] == ["run-1", "run-2"]
        assert records[0]["failure_id"] == records[1]["failure_id"]

    def test_batch_failure_identity_includes_items(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        first = {
            "batch_index": 0,
            "error_category": "batch_processing",
            "items": [{"doc_id": "d1", "sample_index": 0}],
        }
        second = {
            "batch_index": 0,
            "error_category": "batch_processing",
            "items": [{"doc_id": "d2", "sample_index": 0}],
        }
        save_failed_items(out, [first, second], run_id="run")

        records = [
            json.loads(line)
            for line in (tmp_path / "output_failed.jsonl").read_text().splitlines()
        ]
        assert records[0]["failure_id"] != records[1]["failure_id"]

    def test_write_failure_propagates(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("file")
        with pytest.raises(OSError):
            save_failed_items(blocker / "output.jsonl", [{"error": "boom"}])
