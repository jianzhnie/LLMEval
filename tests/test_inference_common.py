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
    count_completed_samples,
    expand_data_with_resume,
    load_jsonl,
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

    def test_custom_keys_with_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        with open(f, "w") as fh:
            fh.write(json.dumps({"question": "q1", "response": ["a"]}) + "\n")
            fh.write(json.dumps({"prompt": "q2", "gen": ["b"]}) + "\n")
        counts = count_completed_samples(f, "question", "response")
        assert counts["q1"] == 1
        assert counts["q2"] == 1  # fell back to prompt/gen


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
