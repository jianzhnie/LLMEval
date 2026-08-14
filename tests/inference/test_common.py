"""Tests for backend-independent inference helpers."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import llmeval.inference.common as common
from llmeval.inference.common import (
    ResumeState,
    append_jsonl,
    derive_request_seed,
    load_jsonl,
    load_resume_state,
    prepare_sample_requests,
    redact_config_for_logging,
    run_concurrent_requests,
    warn_result_manifest,
    write_run_manifest,
)


def _resume(
    *,
    completed_indices: dict[str, set[int]] | None = None,
    prompts: dict[str, str] | None = None,
) -> ResumeState:
    return ResumeState(
        completed_indices=completed_indices or {},
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

    def test_non_standard_numeric_constant_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nan.jsonl"
        path.write_text('{"score": NaN}\n')
        with pytest.raises(ValueError, match="non-standard JSON"):
            load_jsonl(path)


class TestRunManifest:
    @pytest.mark.parametrize("doc_id", [1, True, [], {}])
    def test_manifest_requires_string_document_ids(
        self, tmp_path: Path, doc_id: object
    ) -> None:
        with pytest.raises(ValueError, match="non-empty doc_id"):
            write_run_manifest(tmp_path / "output.jsonl", [{"doc_id": doc_id}], 1)

    def test_round_trip_and_complete_result(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        source = [
            {"doc_id": "q1", "prompt": "one"},
            {"doc_id": "q2", "prompt": "two"},
        ]
        write_run_manifest(output, source, 2)

        assert json.loads(
            output.with_name("output.jsonl.manifest.json").read_text()
        ) == {
            "doc_ids": ["q1", "q2"],
            "n_samples": 2,
        }
        warn_result_manifest(
            [
                {
                    "doc_id": document_id,
                    "sample_index": sample_index,
                    "n_samples": 2,
                }
                for document_id in ("q1", "q2")
                for sample_index in range(2)
            ],
            output,
        )

    def test_missing_document_only_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "output.jsonl"
        write_run_manifest(
            output,
            [{"doc_id": "q1"}, {"doc_id": "q2"}],
            1,
        )

        warnings: list[str] = []
        monkeypatch.setattr(
            common.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )
        warn_result_manifest(
            [{"doc_id": "q1", "sample_index": 0, "n_samples": 1}],
            output,
        )

        assert warnings == [
            "Result completeness check: missing=1, unexpected=0, duplicates=0; "
            "evaluation will continue"
        ]

    def test_manifest_metadata_comparison_preserves_types(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "output.jsonl"
        write_run_manifest(output, [{"doc_id": "q1"}], 1)
        warnings: list[str] = []
        monkeypatch.setattr(
            common.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )

        warn_result_manifest(
            [{"doc_id": "q1", "sample_index": "0", "n_samples": "1"}],
            output,
        )

        assert "missing=1, unexpected=1" in warnings[0]

    @pytest.mark.parametrize(
        "invalid_metadata",
        [
            {"doc_id": ["q1"], "sample_index": 0, "n_samples": 1},
            {"doc_id": "q1", "sample_index": {"value": 0}, "n_samples": 1},
            {"doc_id": "q1", "sample_index": 0, "n_samples": [1]},
        ],
    )
    def test_malformed_metadata_only_warns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        invalid_metadata: dict[str, object],
    ) -> None:
        output = tmp_path / "output.jsonl"
        write_run_manifest(output, [{"doc_id": "q1"}], 1)
        warnings: list[str] = []
        monkeypatch.setattr(
            common.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )

        warn_result_manifest([invalid_metadata], output)

        assert warnings == [
            "Result completeness check: missing=1, unexpected=1, duplicates=0; "
            "evaluation will continue"
        ]

    def test_manifest_counts_missing_unexpected_and_duplicate_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "output.jsonl"
        write_run_manifest(output, [{"doc_id": "q1"}, {"doc_id": "q2"}], 2)
        warnings: list[str] = []
        monkeypatch.setattr(
            common.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )
        valid = {"doc_id": "q1", "sample_index": 0, "n_samples": 2}

        warn_result_manifest(
            [
                valid,
                valid.copy(),
                {"doc_id": "q1", "sample_index": 2, "n_samples": 2},
                {"doc_id": "unknown", "sample_index": 0, "n_samples": 2},
                {"doc_id": "q2", "sample_index": 1, "n_samples": 3},
            ],
            output,
        )

        assert warnings == [
            "Result completeness check: missing=3, unexpected=3, duplicates=1; "
            "evaluation will continue"
        ]

    def test_existing_manifest_must_match(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        write_run_manifest(output, [{"doc_id": "q1"}], 1)

        with pytest.raises(ValueError, match="does not match"):
            write_run_manifest(output, [{"doc_id": "q1"}], 2)

    def test_legacy_output_does_not_gain_manifest(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        output.write_text('{"doc_id": "q1"}\n')

        write_run_manifest(output, [{"doc_id": "q1"}], 1)

        assert not output.with_name("output.jsonl.manifest.json").exists()


class TestConcurrentRequests:
    def test_submission_window_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pending_sizes: list[int] = []
        real_wait = common.concurrent.futures.wait

        def tracking_wait(futures: set[object], **kwargs: object):
            pending_sizes.append(len(futures))
            return real_wait(futures, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(common.concurrent.futures, "wait", tracking_wait)

        processed, failed = run_concurrent_requests(
            list(range(20)),
            lambda value: value,
            lambda _value: None,
            max_workers=3,
            thread_name_prefix="test",
        )

        assert (processed, failed) == (20, 0)
        assert max(pending_sizes) <= 3

    def test_persistence_failure_stops_new_submissions(self) -> None:
        started: list[int] = []
        release = threading.Event()

        def worker(value: int) -> int:
            started.append(value)
            release.wait(timeout=1)
            return value

        def persist(_value: int) -> None:
            raise OSError("disk full")

        release.set()
        with pytest.raises(OSError, match="disk full"):
            run_concurrent_requests(
                list(range(20)),
                worker,
                persist,
                max_workers=2,
                thread_name_prefix="test",
            )

        assert len(started) <= 2

    def test_persistence_failure_waits_for_running_requests(self) -> None:
        second_started = threading.Event()
        release_second = threading.Event()
        second_finished = threading.Event()

        def worker(value: int) -> int:
            if value == 0:
                second_started.wait(timeout=1)
                return value
            second_started.set()
            release_second.wait(timeout=1)
            second_finished.set()
            return value

        def persist(_value: int) -> None:
            raise OSError("disk full")

        release_timer = threading.Timer(0.1, release_second.set)
        release_timer.start()
        try:
            with pytest.raises(OSError, match="disk full"):
                run_concurrent_requests(
                    [0, 1, 2],
                    worker,
                    persist,
                    max_workers=2,
                    thread_name_prefix="test",
                )

            assert second_finished.is_set()
        finally:
            release_second.set()
            release_timer.join()


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

        assert state.completed_indices == {"q1": {0, 1}, "q2": {0}}
        assert state.completed_count == 3

    def test_empty_string_generation_is_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"doc_id": "q1", "prompt": "p", "gen": ""}) + "\n")

        assert load_resume_state(path, "prompt", "gen").completed_indices == {"q1": {0}}

    def test_empty_list_generation_is_completed(self, tmp_path: Path) -> None:
        """An explicit empty list is one empty answer (scoring convention)."""
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"doc_id": "q1", "prompt": "p", "gen": []}) + "\n")

        assert load_resume_state(path, "prompt", "gen").completed_indices == {"q1": {0}}

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
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "choices": ["A"],
                    "gold": 0,
                    "logprobs": [0.0],
                }
            )
            + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_indices == {"q1": {0}}

    def test_logprobs_with_null_missing_choice_is_completed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "choices": ["A", "B"],
                    "gold": 1,
                    "logprobs": [None, -0.5],
                }
            )
            + "\n"
        )
        assert load_resume_state(path, "prompt", "gen").completed_indices == {"q1": {0}}

    def test_expected_scoring_mode_must_match(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "choices": ["A", "B"],
                    "gold": 0,
                    "logprobs": [-0.5, -1.0],
                    "scoring_mode": "first_token",
                }
            )
            + "\n"
        )

        state = load_resume_state(
            path,
            "prompt",
            "gen",
            expected_scoring_mode="first_token",
        )
        assert state.completed_indices == {"q1": {0}}

        with pytest.raises(ValueError, match="expected 'continuation'"):
            load_resume_state(
                path,
                "prompt",
                "gen",
                expected_scoring_mode="continuation",
            )

    @pytest.mark.parametrize(
        "row",
        [
            {"choices": ["A", "B"], "gold": 0, "logprobs": [-0.5]},
            {"choices": ["A"], "gold": 1, "logprobs": [-0.5]},
            {"choices": ["A"], "gold": "0", "logprobs": [-0.5]},
            {"choices": ["A"], "gold": 0.0, "logprobs": [-0.5]},
            {"choices": ["A"], "gold": True, "logprobs": [-0.5]},
            {
                "choices": ["A", "B"],
                "choice_tokens": [],
                "gold": 0,
                "logprobs": [-0.5, -1.0],
            },
        ],
    )
    def test_invalid_loglikelihood_row_remains_retryable(
        self, tmp_path: Path, row: dict[str, object]
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "scoring_mode": "first_token",
                    **row,
                }
            )
            + "\n"
        )

        state = load_resume_state(
            path, "prompt", "gen", expected_scoring_mode="first_token"
        )

        assert state.completed_count == 0

    def test_legacy_loglikelihood_row_without_choices_is_completed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "scoring_mode": "first_token",
                    "gold": 0,
                    "logprobs": [-0.5],
                }
            )
            + "\n"
        )

        state = load_resume_state(
            path, "prompt", "gen", expected_scoring_mode="first_token"
        )

        assert state.completed_indices == {"q1": {0}}

    def test_huge_integer_logprob_remains_retryable(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "scoring_mode": "first_token",
                    "choices": ["A"],
                    "gold": 0,
                    "logprobs": [10**1000],
                }
            )
            + "\n"
        )

        state = load_resume_state(
            path, "prompt", "gen", expected_scoring_mode="first_token"
        )

        assert state.completed_count == 0

    def test_expected_scoring_mode_error_suggests_migration(
        self, tmp_path: Path
    ) -> None:
        """Legacy rows without scoring_mode fail with a migration hint."""
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"doc_id": "q1", "prompt": "p", "gen": "A"}) + "\n")

        with pytest.raises(ValueError, match="scoring_mode='generate'"):
            load_resume_state(path, "prompt", "gen", expected_scoring_mode="generate")

    def test_generate_resume_requires_generation_even_with_logprobs(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "doc_id": "q1",
                    "prompt": "p",
                    "choices": ["A"],
                    "gold": 0,
                    "logprobs": [-0.5],
                    "scoring_mode": "generate",
                }
            )
            + "\n"
        )

        state = load_resume_state(
            path, "prompt", "gen", expected_scoring_mode="generate"
        )

        assert state.completed_count == 0

    @pytest.mark.parametrize(
        "logprobs",
        [
            [],  # empty list: no scores recorded
            "not-a-list",  # input data carrying an unrelated logprobs field
            0.5,  # bare number, not a per-choice list
            ["A", "B"],  # non-numeric elements
            None,  # explicit null
            [None, None],  # no choice received a finite score
            [True],  # bool is an int subclass but not a valid score
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

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_resume_number_raises(
        self, tmp_path: Path, constant: str
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            f'{{"doc_id": "q1", "prompt": "p", "logprobs": [{constant}]}}\n'
        )

        with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
            load_resume_state(path, "prompt", "gen")

    def test_legacy_failure_row_remains_retryable(self, tmp_path: Path) -> None:
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
        assert load_resume_state(path, "prompt", "gen").completed_count == 0

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
        assert state.completed_indices == {"q1": {0}}
        assert path.read_text().endswith("\n")

        append_jsonl(path, [{"doc_id": "q2", "prompt": "p2", "gen": "b"}])

        assert load_resume_state(path, "prompt", "gen").completed_indices == {
            "q1": {0},
            "q2": {0},
        }

    def test_append_repairs_unterminated_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text('{"doc_id": "q1", "gen": "a"}')

        append_jsonl(path, [{"doc_id": "q2", "gen": "b"}])

        assert load_resume_state(path, "prompt", "gen").completed_indices == {
            "q1": {0},
            "q2": {0},
        }

    @pytest.mark.parametrize("ending", ["", "\r"])
    def test_repair_adds_newline_to_complete_final_record(
        self, tmp_path: Path, ending: str
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": "q1", "prompt": "p", "gen": "a"}) + ending
        )

        load_resume_state(path, "prompt", "gen", repair_truncated_last_line=True)
        append_jsonl(path, [{"doc_id": "q2", "prompt": "p2", "gen": "b"}])

        assert load_resume_state(path, "prompt", "gen").completed_count == 2

    def test_read_only_repair_fallback_does_not_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "out.jsonl"
        content = json.dumps({"doc_id": "q1", "prompt": "p", "gen": "a"})
        path.write_text(content)
        original_open = Path.open

        def open_read_only(
            target: Path, mode: str = "r", *args: object, **kwargs: object
        ):
            if target == path and mode == "r+b":
                raise PermissionError("read-only filesystem")
            return original_open(target, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_read_only)

        state = load_resume_state(
            path, "prompt", "gen", repair_truncated_last_line=True
        )

        assert state.completed_indices == {"q1": {0}}
        assert path.read_text() == content

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
        assert load_resume_state(path, "question", "response").completed_indices == {
            "q1": {0},
            "q2": {0},
        }

    def test_completed_row_requires_document_id(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"prompt": "p", "gen": ["a"]}) + "\n")
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            load_resume_state(path, "prompt", "gen")

    @pytest.mark.parametrize("doc_id", [1, True, [], {}])
    def test_completed_row_requires_string_document_id(
        self, tmp_path: Path, doc_id: object
    ) -> None:
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"doc_id": doc_id, "prompt": "p", "gen": "a"}) + "\n"
        )
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

    def test_conflicting_n_samples_for_document_are_rejected(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.jsonl"
        rows = [
            {
                "doc_id": "q1",
                "prompt": "p",
                "gen": "a",
                "sample_index": 0,
                "n_samples": 2,
            },
            {
                "doc_id": "q1",
                "prompt": "p",
                "gen": "b",
                "sample_index": 1,
                "n_samples": 3,
            },
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        with pytest.raises(ValueError, match="conflicting n_samples"):
            load_resume_state(path, "prompt", "gen")


class TestExpansion:
    def test_copies_each_document_requested_number_of_times(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = prepare_sample_requests(raw, _resume(), "prompt", 3)
        assert len(expanded) == 3
        assert [item["sample_index"] for item in expanded] == [0, 1, 2]
        assert {item["n_samples"] for item in expanded} == {3}
        assert (
            len(
                {
                    derive_request_seed(
                        123, item["doc_id"], item["prompt"], item["sample_index"]
                    )
                    for item in expanded
                }
            )
            == 3
        )

    @pytest.mark.parametrize("doc_id", [1, True, [], {}])
    def test_input_requires_string_document_id(self, doc_id: object) -> None:
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            prepare_sample_requests(
                [{"doc_id": doc_id, "prompt": "p"}], _resume(), "prompt", 1
            )

    def test_resume_uses_exact_completed_indices(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        expanded = prepare_sample_requests(
            raw,
            _resume(completed_indices={"q1": {0, 2}}),
            "prompt",
            4,
        )
        assert len(expanded) == 2
        assert [item["sample_index"] for item in expanded] == [1, 3]

    def test_resume_rejects_changed_n_samples(self) -> None:
        resume_state = _resume(completed_indices={"q1": {0}})
        resume_state.n_samples_by_document["q1"] = 2

        with pytest.raises(ValueError, match="n_samples=2"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": "p"}],
                resume_state,
                "prompt",
                3,
            )

    def test_all_completed_yields_nothing(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        assert (
            prepare_sample_requests(
                raw,
                _resume(completed_indices={"q1": {0, 1}}),
                "prompt",
                2,
            )
            == []
        )

    def test_too_many_completed_rows_is_rejected(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p"}]
        with pytest.raises(ValueError, match="outside requested"):
            prepare_sample_requests(
                raw,
                _resume(completed_indices={"q1": {0, 2}}),
                "prompt",
                2,
            )

    @pytest.mark.parametrize("count", [0, -1, True])
    def test_rejects_non_positive_generation_count(self, count: int) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": "p"}],
                _resume(),
                "prompt",
                count,
            )

    def test_requires_document_id_and_prompt(self) -> None:
        with pytest.raises(ValueError, match="missing required 'doc_id'"):
            prepare_sample_requests([{"prompt": "p"}], _resume(), "prompt", 1)
        with pytest.raises(ValueError, match="non-empty string prompt"):
            prepare_sample_requests([{"doc_id": "q1"}], _resume(), "prompt", 1)

    @pytest.mark.parametrize("prompt", [{"text": "p"}, ["p"], 42])
    def test_rejects_non_string_prompts(self, prompt: object) -> None:
        with pytest.raises(ValueError, match="non-empty string prompt"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": prompt}],
                _resume(),
                "prompt",
                1,
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
            )

    def test_base_seed_must_be_a_non_negative_integer(self) -> None:
        with pytest.raises(ValueError, match="base_seed must be non-negative"):
            derive_request_seed(-1, "q1", "p", 0)

    def test_changed_prompt_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="changed prompt"):
            prepare_sample_requests(
                [{"doc_id": "q1", "prompt": "new"}],
                _resume(completed_indices={"q1": {0}}, prompts={"q1": "old"}),
                "prompt",
                2,
            )

    def test_copies_are_independent(self) -> None:
        raw = [{"doc_id": "q1", "prompt": "p", "gen": ["existing"]}]
        expanded = prepare_sample_requests(raw, _resume(), "prompt", 2)
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
