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
    completed_sample_indices_by_identity,
    count_completed_samples,
    count_completed_samples_by_id,
    count_completed_samples_by_identity,
    expand_data_with_resume,
    expand_group_for_sampling,
    is_explicit_tool_choice,
    load_jsonl,
    prepare_data_with_resume,
    require_document_id,
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
        assert count_completed_samples(tmp_path / "none.jsonl", "prompt", "gen") == {}

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text("")
        assert count_completed_samples(f, "prompt", "gen") == {}

    def test_counts_gen_list_lengths(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"prompt": "q1", "gen": ["a", "b"]}) + "\n")
            fh.write(json.dumps({"prompt": "q1", "gen": ["c"]}) + "\n")
            fh.write(json.dumps({"prompt": "q2", "gen": ["d"]}) + "\n")
        counts = count_completed_samples(f, "prompt", "gen")
        assert counts["q1"] == 3
        assert counts["q2"] == 1

    def test_null_gen_tolerated(self, tmp_path: Path) -> None:
        """Regression: a partially-written line with gen:null must not crash."""
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"prompt": "q1", "gen": None}) + "\n")
            fh.write(json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n")
        assert count_completed_samples(f, "prompt", "gen")["q1"] == 1

    def test_non_list_gen_tolerated(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(json.dumps({"prompt": "q1", "gen": "bare string"}) + "\n")
        assert count_completed_samples(f, "prompt", "gen")["q1"] == 0

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write("bad json\n")
            fh.write(json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n")
        assert count_completed_samples(f, "prompt", "gen")["q1"] == 1

    def test_non_object_line_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text("[1, 2]\n" + json.dumps({"prompt": "q1", "gen": ["a"]}) + "\n")
        assert count_completed_samples(f, "prompt", "gen")["q1"] == 1

    def test_custom_keys_with_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"question": "q1", "response": ["a"]}) + "\n")
            fh.write(json.dumps({"prompt": "q2", "gen": ["b"]}) + "\n")
        counts = count_completed_samples(f, "question", "response")
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

        assert count_completed_samples(output, "prompt", "gen", legacy_only=True) == {
            "legacy": 1
        }


class TestExpandDataWithResume:
    def test_expands_to_n_samples(self) -> None:
        raw = [{"prompt": "q1", "answer": "a1"}]
        expanded = expand_data_with_resume(raw, {}, "prompt", 3)
        assert len(expanded) == 3

    def test_subtracts_completed(self) -> None:
        raw = [{"prompt": "q1", "answer": "a1"}]
        expanded = expand_data_with_resume(raw, {"q1": 2}, "prompt", 4)
        assert len(expanded) == 2

    def test_all_completed_yields_nothing(self) -> None:
        raw = [{"prompt": "q1", "answer": "a1"}]
        assert expand_data_with_resume(raw, {"q1": 2}, "prompt", 2) == []

    def test_empty_prompt_skipped(self) -> None:
        raw = [{"prompt": "  ", "answer": "a1"}, {"answer": "a2"}]
        assert expand_data_with_resume(raw, {}, "prompt", 2) == []

    def test_copies_are_independent(self) -> None:
        raw = [{"prompt": "q1", "gen": ["existing"]}]
        expanded = expand_data_with_resume(raw, {}, "prompt", 2)
        expanded[0]["gen"].append("new")
        assert expanded[1]["gen"] == ["existing"]
        assert raw[0]["gen"] == ["existing"]

    def test_stable_ids_are_attached_and_resume_by_id(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "same"}]
        expanded = expand_data_with_resume(
            raw, {("q1", "same"): 1}, "prompt", 2, stable_ids=True
        )
        assert len(expanded) == 1
        assert expanded[0]["doc_id"] == "q1"
        assert expanded[0]["_llmeval_sample_index"] == 1

    def test_stable_resume_falls_back_to_legacy_prompt_count(self) -> None:
        expanded = expand_data_with_resume(
            [{"doc_id": "prepared:0", "prompt": "legacy"}],
            {"legacy": 1},
            "prompt",
            2,
            stable_ids=True,
        )

        assert len(expanded) == 1
        assert expanded[0]["_llmeval_sample_index"] == 1

    def test_missing_document_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            require_document_id({"prompt": "q"}, 0)


class TestPrepareDataWithResume:
    def test_sets_remaining_sample_count(self) -> None:
        raw = [{"prompt": "q1", "answer": "a1"}]
        prepared = prepare_data_with_resume(raw, {}, "prompt", 3)
        assert prepared == [{"prompt": "q1", "answer": "a1", "n_samples": 3}]

    def test_subtracts_completed(self) -> None:
        raw = [{"prompt": "q1", "answer": "a1"}]
        prepared = prepare_data_with_resume(raw, {"q1": 2}, "prompt", 4)
        assert prepared[0]["n_samples"] == 2

    def test_skips_invalid_items(self) -> None:
        raw = [{"prompt": "  "}, "bad", {"answer": "x"}]
        assert prepare_data_with_resume(raw, {}, "prompt", 2) == []

    def test_rejects_non_positive_sample_count(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            prepare_data_with_resume([{"prompt": "q"}], {}, "prompt", 0)

    def test_stable_ids_are_written_to_prepared_items(self) -> None:
        prepared = prepare_data_with_resume(
            [{"doc_id": "q1", "prompt": "q"}], {}, "prompt", 2, stable_ids=True
        )
        assert prepared[0]["doc_id"] == "q1"
        assert prepared[0]["_llmeval_sample_start"] == 0


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
        assert [item["_llmeval_sample_index"] for item in expanded] == [3, 4]


class TestSampleSeed:
    def test_seed_is_stable_for_same_item(self) -> None:
        item = {
            "doc_id": "doc:1",
            "prompt": "What is 2+2?",
            "_llmeval_sample_index": 2,
        }
        assert sample_seed_for_item(123, item) == sample_seed_for_item(123, item)

    def test_seed_changes_with_sample_index_and_document_id(self) -> None:
        item = {"doc_id": "doc:1", "prompt": "q", "_llmeval_sample_index": 0}
        next_sample = {**item, "_llmeval_sample_index": 1}
        other_document = {**item, "doc_id": "doc:2"}
        assert sample_seed_for_item(123, item) != sample_seed_for_item(123, next_sample)
        assert sample_seed_for_item(123, item) != sample_seed_for_item(
            123, other_document
        )

    def test_negative_base_seed_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            sample_seed_for_item(-1, {"prompt": "q"})


class TestStableResumeCounts:
    def test_counts_generation_list_by_document_id(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_id": "doc:q1",
                    "gen": ["a", "b"],
                }
            )
            + "\n"
        )
        assert count_completed_samples_by_id(output, "gen") == {"doc:q1": 2}

    def test_loglikelihood_choices_count_as_one_sample(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps(
                {
                    "doc_id": "doc:q1",
                    "logprobs": [-3.0, -1.0, -2.0, -4.0],
                }
            )
            + "\n"
        )

        assert count_completed_samples_by_id(output, "gen") == {"doc:q1": 1}

    def test_identity_count_detects_prompt_changes(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text(
            json.dumps({"doc_id": "q1", "prompt": "old", "gen": ["a"]}) + "\n"
        )

        counts = count_completed_samples_by_identity(output, "prompt", "gen")
        remaining = expand_data_with_resume(
            [{"doc_id": "q1", "prompt": "new"}],
            counts,
            "prompt",
            1,
            stable_ids=True,
        )

        assert counts == {("q1", "old"): 1}
        assert len(remaining) == 1

    def test_identity_indices_deduplicate_resumed_rows(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        rows = [
            {
                "doc_id": "q1",
                "prompt": "q",
                "gen": ["a", "c"],
                "_llmeval_sample_indices": [0, 2],
            },
            {
                "doc_id": "q1",
                "prompt": "q",
                "gen": ["c"],
                "_llmeval_sample_indices": [2],
            },
        ]
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))

        assert completed_sample_indices_by_identity(output, "prompt", "gen") == {
            ("q1", "q"): {0, 2}
        }
        assert count_completed_samples_by_identity(output, "prompt", "gen") == {
            ("q1", "q"): 2
        }


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
        assert record == {"prompt": "q", "error": "boom"}

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
