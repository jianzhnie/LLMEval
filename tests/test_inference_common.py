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
    expand_data_with_resume,
    expand_group_for_sampling,
    is_explicit_tool_choice,
    load_jsonl,
    load_resume_state,
    prepare_data_with_resume,
    redact_config_for_logging,
    sample_count_for_item,
    sample_seed_for_item,
    save_failed_items,
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

    def test_counts_gen_list_lengths(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"prompt": "q1", "gen": ["a", "b"]}) + "\n")
            fh.write(json.dumps({"prompt": "q1", "gen": ["c"]}) + "\n")
            fh.write(json.dumps({"prompt": "q2", "gen": ["d"]}) + "\n")
        counts = load_resume_state(f, "prompt", "gen").legacy_counts
        assert counts["q1"] == 3
        assert counts["q2"] == 1

    def test_null_gen_tolerated(self, tmp_path: Path) -> None:
        """Regression: a partially-written line with gen:null must not crash."""
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"prompt": "q1", "gen": None}) + "\n")
            fh.write(json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n")
        assert load_resume_state(f, "prompt", "gen").legacy_counts["q1"] == 1

    def test_non_list_gen_tolerated(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(json.dumps({"prompt": "q1", "gen": "bare string"}) + "\n")
        assert load_resume_state(f, "prompt", "gen").legacy_counts == {}

    def test_malformed_line_fails_with_path_and_line(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write("bad json\n")
            fh.write(json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n")
        with pytest.raises(ValueError, match=r"out\.jsonl at line 1"):
            load_resume_state(f, "prompt", "gen")

    def test_non_object_line_fails_with_path_and_line(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text("[1, 2]\n" + json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n")
        with pytest.raises(
            ValueError, match=r"out\.jsonl line 1 must contain an object"
        ):
            load_resume_state(f, "prompt", "gen")

    def test_repair_ignores_only_unterminated_final_json(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n{")

        state = load_resume_state(
            f,
            "prompt",
            "gen",
            repair_truncated_last_line=True,
        )

        assert state.legacy_counts == {"q1": 1}

    def test_repair_does_not_ignore_malformed_middle_line(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps({"prompt": "q1", "gen": ["a"]})
            + "\n{\n"
            + json.dumps({"prompt": "q2", "gen": ["b"]})
        )

        with pytest.raises(ValueError, match="at line 2"):
            load_resume_state(
                f,
                "prompt",
                "gen",
                repair_truncated_last_line=True,
            )

    def test_custom_keys_with_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"question": "q1", "response": ["a"]}) + "\n")
            fh.write(json.dumps({"prompt": "q2", "gen": ["b"]}) + "\n")
        counts = load_resume_state(f, "question", "response").legacy_counts
        assert counts["q1"] == 1
        assert counts["q2"] == 1  # fell back to prompt/gen

    def test_legacy_only_ignores_stable_id_records(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps({"prompt": "stable", "gen": ["a"], "doc_id": "d"})
            + "\n"
            + json.dumps({"prompt": "legacy", "gen": ["b"]})
            + "\n"
        )

        state = load_resume_state(output, "prompt", "gen")
        assert state.legacy_counts == {"legacy": 1}
        assert state.completed_indices == {("d", "stable"): {0}}


class TestExpandDataWithResume:
    def test_expands_to_n_samples(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = expand_data_with_resume(raw, {}, {}, "prompt", 3)
        assert [item["sample_index"] for item in expanded] == [0, 1, 2]

    def test_copies_are_independent(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p", "gen": ["existing"]}]
        expanded = expand_data_with_resume(raw, {}, {}, "prompt", 2)
        expanded[0]["gen"].append("new")
        assert expanded[1]["gen"] == ["existing"]
        assert raw[0]["gen"] == ["existing"]

    def test_regenerates_only_missing_indices(self) -> None:
        """A mid-run failure must be retried, not the highest contiguous count."""
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = expand_data_with_resume(raw, {("q1", "p"): {0, 2}}, {}, "prompt", 4)
        assert [item["sample_index"] for item in expanded] == [1, 3]

    def test_all_completed_yields_nothing(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = expand_data_with_resume(raw, {("q1", "p"): {0, 1}}, {}, "prompt", 2)
        assert expanded == []

    def test_falls_back_to_legacy_counts(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "legacy"}]
        expanded = expand_data_with_resume(raw, {}, {"legacy": 1}, "prompt", 2)
        assert [item["sample_index"] for item in expanded] == [1]

    def test_requires_document_id(self) -> None:
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            expand_data_with_resume([{"prompt": "p"}], {}, {}, "prompt", 1)

    def test_missing_prompt_is_schema_error(self) -> None:
        with pytest.raises(ValueError, match="no non-empty prompt"):
            expand_data_with_resume([{"doc_id": "q1"}], {}, {}, "prompt", 1)


class TestPrepareDataWithResume:
    def test_sets_remaining_sample_count(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "q", "answer": "a1"}]
        prepared = prepare_data_with_resume(raw, {}, {}, "prompt", 3)
        assert prepared[0]["n_samples"] == 3
        assert prepared[0]["_llmeval_requested_sample_indices"] == [0, 1, 2]

    def test_rejects_non_positive_sample_count(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            prepare_data_with_resume(
                [{"doc_id": "q1", "prompt": "q"}], {}, {}, "prompt", 0
            )

    def test_stable_ids_preserve_non_contiguous_missing_indices(self) -> None:
        prepared = prepare_data_with_resume(
            [{"doc_id": "q1", "prompt": "q"}],
            {("q1", "q"): {0, 2}},
            {},
            "prompt",
            4,
        )
        assert prepared[0]["n_samples"] == 2
        assert prepared[0]["_llmeval_requested_sample_indices"] == [1, 3]
        expanded = expand_group_for_sampling(prepared)
        assert [item["sample_index"] for item in expanded] == [1, 3]

    def test_missing_prompt_is_schema_error(self) -> None:
        with pytest.raises(ValueError, match="no non-empty prompt"):
            prepare_data_with_resume([{"doc_id": "q1"}], {}, {}, "prompt", 1)


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


class TestSampleCountHelpers:
    def test_sample_count_for_item_defaults_to_one(self) -> None:
        assert sample_count_for_item({"prompt": "q"}) == 1

    def test_sample_count_for_item_reads_n_samples(self) -> None:
        assert sample_count_for_item({"n_samples": 4}) == 4

    def test_expand_group_for_sampling_repeats_each_item(self) -> None:
        items = [
            {
                "prompt": "q",
                "n_samples": 2,
                "doc_id": "doc:q",
                "_llmeval_sample_start": 3,
            }
        ]
        expanded = expand_group_for_sampling(items)
        assert len(expanded) == 2
        assert expanded[0] is not items[0]
        assert [item["sample_index"] for item in expanded] == [3, 4]

    def test_expand_group_preserves_target_samples_on_resume(self) -> None:
        # A resumed batch carries _llmeval_target_samples (the full depth) and
        # _llmeval_requested_sample_indices (the exact missing indices). The
        # expanded copies must keep the target so online._build_result writes
        # expected_samples == 4, and must not leak request-scoped fields.
        items = [
            {
                "prompt": "q",
                "n_samples": 2,
                "doc_id": "doc:q",
                "_llmeval_target_samples": 4,
                "_llmeval_requested_sample_indices": [2, 3],
            }
        ]
        expanded = expand_group_for_sampling(items)
        assert [item["sample_index"] for item in expanded] == [2, 3]
        for item in expanded:
            assert item["_llmeval_target_samples"] == 4
            assert "_llmeval_requested_sample_indices" not in item
            assert "_llmeval_sample_start" not in item


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
    def test_identity_count_detects_prompt_changes(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps({"doc_id": "q1", "prompt": "old", "gen": ["a"]}) + "\n"
        )

        indices = load_resume_state(output, "prompt", "gen").completed_indices
        remaining = expand_data_with_resume(
            [{"doc_id": "q1", "prompt": "new"}],
            indices,
            {},
            "prompt",
            1,
        )

        assert indices == {("q1", "old"): {0}}
        assert len(remaining) == 1

    def test_identity_indices_deduplicate_resumed_rows(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        rows = [
            {
                "doc_id": "q1",
                "prompt": "q",
                "gen": ["a", "c"],
                "sample_indices": [0, 2],
            },
            {
                "doc_id": "q1",
                "prompt": "q",
                "gen": ["c"],
                "sample_indices": [2],
            },
        ]
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))

        assert load_resume_state(output, "prompt", "gen").completed_indices == {
            ("q1", "q"): {0, 2}
        }

    def test_scalar_sample_index_honored_for_single_generation(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps({"doc_id": "q1", "prompt": "q", "gen": ["a"], "sample_index": 2})
            + "\n"
        )

        assert load_resume_state(output, "prompt", "gen").completed_indices == {
            ("q1", "q"): {2}
        }

    def test_legacy_private_sample_indices_are_honored(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "q",
                    "gen": ["a", "b"],
                    "_llmeval_requested_sample_indices": [1, 3],
                }
            )
            + "\n"
        )

        assert load_resume_state(output, "prompt", "gen").completed_indices == {
            ("q1", "q"): {1, 3}
        }

    def test_duplicate_explicit_indices_are_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "q",
                    "gen": ["a", "b"],
                    "sample_indices": [0, 0],
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="unique non-negative ints"):
            load_resume_state(output, "prompt", "gen")

    def test_conflicting_single_sample_index_fields_are_rejected(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "q",
                    "gen": ["a"],
                    "sample_index": 0,
                    "sample_indices": [1],
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="provide exactly one"):
            load_resume_state(output, "prompt", "gen")

    def test_scalar_sample_index_ignored_for_multiple_generations(
        self, tmp_path: Path
    ) -> None:
        """A multi-generation row with only a scalar index is inconsistent.

        It must fall back to free-slot allocation instead of pinning a single
        index (which would re-generate already-completed samples on resume).
        """
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {"doc_id": "q1", "prompt": "q", "gen": ["a", "b"], "sample_index": 0}
            )
            + "\n"
        )

        assert load_resume_state(output, "prompt", "gen").completed_indices == {
            ("q1", "q"): {0, 1}
        }

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

    def test_completed_record_without_prompt_is_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(json.dumps({"doc_id": "q1", "gen": ["a"]}) + "\n")

        with pytest.raises(ValueError, match="no non-empty prompt"):
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
