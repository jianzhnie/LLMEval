"""Tests for llmeval.utils.config."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from llmeval.utils.config import (
    DataArguments,
    EvalTaskArguments,
    GenerationArguments,
    MCInferConfig,
    OfflineInferArguments,
    OnlineInferArguments,
    PromptArguments,
    ServerArguments,
    VLLMEngineArguments,
    VLLMGenerationArguments,
)


class TestDataArguments:
    def test_defaults_are_valid(self, tmp_path: Path) -> None:
        in_file = tmp_path / "input.jsonl"
        in_file.touch()  # DataArguments now raises on missing input
        args = DataArguments(input_file=str(in_file))
        assert args.input_file == str(in_file)

    def test_config_construction_does_not_create_output_dir(
        self, tmp_path: Path
    ) -> None:
        in_file = tmp_path / "x.jsonl"
        in_file.touch()
        out = tmp_path / "sub" / "out.jsonl"
        DataArguments(input_file=str(in_file), output_file=str(out))
        assert not out.parent.exists()

    def test_missing_input_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"does not exist"):
            DataArguments(input_file=str(tmp_path / "nonexistent.jsonl"))


def test_eval_result_path_overrides_legacy_cache_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.touch()
    result_path = tmp_path / "results.jsonl"

    args = EvalTaskArguments(
        input_path=str(input_path), result_path=str(result_path), code_k_values="1,5,5"
    )

    assert args.cache_path == str(result_path)
    assert args.code_k_values_tuple == (1, 5)


def test_eval_code_k_values_reject_invalid_values(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.touch()

    with pytest.raises(ValueError, match="positive integers"):
        EvalTaskArguments(input_path=str(input_path), code_k_values="1,0")


class TestPromptArguments:
    def test_default_resolves_system_prompt(self) -> None:
        args = PromptArguments()
        assert args.system_prompt is None  # "empty" maps to None

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid system prompt type"):
            PromptArguments(system_prompt_type="nonexistent")

    def test_deepseek_r1_resolves(self) -> None:
        args = PromptArguments(system_prompt_type="deepseek_r1")
        assert args.system_prompt is not None
        assert "think" in args.system_prompt.lower()

    @pytest.mark.parametrize("key", ["input_key", "label_key", "response_key"])
    def test_empty_record_key_raises(self, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            PromptArguments(**{key: "  "})


class TestGenerationArguments:
    def test_defaults_valid(self) -> None:
        args = GenerationArguments()
        assert 0 < args.temperature <= 2.0
        assert args.n_samples > 0
        assert args.max_completion_tokens == 32768
        assert not hasattr(args, "max_tokens")

    @pytest.mark.parametrize("temp", [-0.1, 2.1])
    def test_invalid_temperature_raises(self, temp: float) -> None:
        with pytest.raises(ValueError, match=r"[Tt]emperature"):
            GenerationArguments(temperature=temp)

    def test_zero_temperature_is_valid_for_online_backends(self) -> None:
        assert GenerationArguments(temperature=0.0).temperature == 0.0

    def test_seed_is_shared_by_generation_backends(self) -> None:
        assert GenerationArguments(seed=42).seed == 42

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            GenerationArguments(seed=-1)


class TestInferenceSpecificArguments:
    def test_online_does_not_expose_offline_only_fields(self) -> None:
        online_fields = {item.name for item in fields(OnlineInferArguments)}
        assert not online_fields.intersection(
            {
                "cache_dir",
                "batch_size",
                "fail_fast",
                "do_sample",
                "top_k",
                "enable_thinking",
                "repetition_penalty",
                "skip_special_tokens",
            }
        )

    def test_offline_keeps_and_validates_local_fields(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.jsonl"
        input_file.touch()
        args = OfflineInferArguments(input_file=str(input_file), temperature=0.0)

        assert args.batch_size == 128
        assert args.fail_fast is True
        assert args.temperature == 0.0
        assert args.top_k == 40
        assert args.skip_special_tokens is True
        assert args.repetition_penalty == 1.0

        with pytest.raises(ValueError, match="positive integer"):
            OfflineInferArguments(input_file=str(input_file), batch_size=0)


class TestVLLMEngineArguments:
    def test_defaults_valid(self) -> None:
        args = VLLMEngineArguments()
        assert args.tensor_parallel_size >= 1

    def test_invalid_gpu_util_raises(self) -> None:
        with pytest.raises(ValueError, match="GPU memory"):
            VLLMEngineArguments(gpu_memory_utilization=0.0)


class TestMCAndEvaluationP0Config:
    def test_mc_generation_defaults_to_one_sample_and_auto_scoring(self) -> None:
        config = MCInferConfig()
        assert config.n_samples == 1
        assert config.loglikelihood_mode == "first_token"

    def test_invalid_mc_aggregation_raises(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.jsonl"
        input_path.write_text("{}\n")
        with pytest.raises(ValueError, match="mc_aggregation"):
            EvalTaskArguments(
                input_path=str(input_path),
                mc_aggregation="invalid",
            )

    def test_rope_scaling_parsed(self) -> None:
        args = VLLMEngineArguments(rope_scaling='{"type": "dynamic", "factor": 2.0}')
        assert args.rope_scaling_dict == {"type": "dynamic", "factor": 2.0}

    def test_invalid_rope_scaling_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            VLLMEngineArguments(rope_scaling="not json")


class TestVLLMGenerationArguments:
    def test_defaults_are_valid(self) -> None:
        args = VLLMGenerationArguments()
        assert args.top_k == 40
        assert args.repetition_penalty == 1.0

    def test_zero_repetition_penalty_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            VLLMGenerationArguments(repetition_penalty=0.0)


class TestServerArguments:
    def test_env_api_key_picked_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        args = ServerArguments()
        assert args.api_key == "test-key-123"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="http"):
            ServerArguments(base_url="ftp://bad.url")

    def test_empty_model_name_raises(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ServerArguments(model_name="  ")

    def test_zero_max_workers_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ServerArguments(max_workers=0)

    def test_tool_choice_is_normalized(self) -> None:
        assert ServerArguments(tool_choice=" AUTO ").tool_choice == "auto"

    def test_invalid_tool_choice_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_choice"):
            ServerArguments(tool_choice="function_name")

    def test_extra_body_defaults_to_empty_object(self) -> None:
        args = ServerArguments()
        assert args.extra_body_dict == {}

    def test_extra_body_parses_provider_extensions(self) -> None:
        args = ServerArguments(
            extra_body='{"top_k": 40, "chat_template_kwargs": {"enable_thinking": true}}'
        )
        assert args.extra_body_dict == {
            "top_k": 40,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    @pytest.mark.parametrize("extra_body", ["not-json", "[]", "null"])
    def test_extra_body_must_be_a_json_object(self, extra_body: str) -> None:
        with pytest.raises(ValueError, match="extra_body"):
            ServerArguments(extra_body=extra_body)


class TestMCInferConfig:
    def test_reuses_shared_argument_classes(self) -> None:
        args = MCInferConfig()

        assert isinstance(args, DataArguments)
        assert isinstance(args, PromptArguments)
        assert isinstance(args, GenerationArguments)
        assert isinstance(args, ServerArguments)

    def test_defaults_valid(self) -> None:
        args = MCInferConfig()
        assert args.mode == "loglikelihood"
        assert args.max_workers > 0
        assert args.n_shot == 0
        assert args.max_completion_tokens == 2048
        assert args.top_p == 0.95
        assert args.extra_body_dict == {}
        assert not hasattr(args, "max_tokens")

    def test_invalid_tool_choice_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_choice"):
            MCInferConfig(tool_choice="function_name")

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mc-test")
        assert MCInferConfig().api_key == "sk-mc-test"

    def test_api_key_default_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert MCInferConfig().api_key == "EMPTY"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            MCInferConfig(mode="bogus")

    def test_nonempty_input_reuses_data_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            MCInferConfig(input_file=str(tmp_path / "missing.jsonl"))

    @pytest.mark.parametrize("key", ["input_key", "label_key", "response_key"])
    def test_empty_record_key_raises(self, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            MCInferConfig(**{key: "\t"})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_workers": 0},
            {"request_timeout": -1},
            {"max_retries": -1},
            {"max_completion_tokens": 0},
            {"temperature": 2.1},
            {"top_p": 1.1},
            {"n_shot": -1},
            {"base_url": "  "},
            {"model_name": ""},
            {"extra_body": "[]"},
        ],
    )
    def test_invalid_values_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            MCInferConfig(**kwargs)


class TestEvalTaskArguments:
    def test_does_not_expose_unused_input_key(self) -> None:
        assert "input_key" not in {item.name for item in fields(EvalTaskArguments)}

    def test_valid_task_names(self, tmp_path: Path) -> None:
        input_f = tmp_path / "data.jsonl"
        input_f.write_text('{"prompt": "q", "answer": "a"}\n')
        for task in [
            "math_opensource/aime24",
            "math_opensource/math500",
            "math_opensource/hmmt25",
        ]:
            args = EvalTaskArguments(input_path=str(input_f), task_name=task)
            assert args.task_name == task

    def test_evaluation_output_schema_defaults_to_compact(self, tmp_path: Path) -> None:
        input_f = tmp_path / "data.jsonl"
        input_f.write_text('{"prompt": "q", "answer": "a"}\n')
        args = EvalTaskArguments(input_path=str(input_f))
        assert args.output_schema == "compact"

    def test_invalid_output_schema_raises(self, tmp_path: Path) -> None:
        input_f = tmp_path / "data.jsonl"
        input_f.write_text("{}\n")
        with pytest.raises(ValueError, match="output_schema"):
            EvalTaskArguments(input_path=str(input_f), output_schema="full")

    def test_task_validation_is_delegated_to_registry(self, tmp_path: Path) -> None:
        input_f = tmp_path / "data.jsonl"
        input_f.write_text("{}\n")
        args = EvalTaskArguments(input_path=str(input_f), task_name="custom/task")
        assert args.task_name == "custom/task"

    def test_missing_input_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            EvalTaskArguments(input_path=str(tmp_path / "nope.jsonl"))
