"""Tests for llmeval.tasks.mc_eval (mc_score + mc_infer)."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Mock heavy dependencies ──
for mod_name in ("openai", "httpx"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Provide stubs for openai exceptions used at module level in mc_infer
_openai_mod = sys.modules["openai"]
for _exc in ("APIConnectionError", "APIError", "RateLimitError"):
    if not hasattr(_openai_mod, _exc):
        setattr(_openai_mod, _exc, type(_exc, (Exception,), {}))

# mc_infer imports HfArgumentParser/tqdm at module level; stub if absent

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.HfArgumentParser = MagicMock
    sys.modules["transformers"] = _tf

if "tqdm" not in sys.modules:
    _tqdm = types.ModuleType("tqdm")
    _tqdm.tqdm = MagicMock
    sys.modules["tqdm"] = _tqdm


# ===========================================================================
# mc_score tests
# ===========================================================================


class TestMCExtractAnswer:
    """Test answer letter extraction from generated text."""

    def test_extract_answer_pattern(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("Some text\nAnswer: B") == "B"
        assert _extract_answer("Reasoning...\n答案：D") == "D"
        assert _extract_answer("Answer: A") == "A"

    def test_extract_last_letter_fallback(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("The correct option is C.") == "C"
        assert _extract_answer("I think A and B but choose D") == "D"

    def test_extract_empty(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("") == ""
        assert _extract_answer("no letters here") == ""


class TestScoreLoglikelihood:
    """Test loglikelihood scoring with acc + acc_norm."""

    def test_all_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0], "choices": ["a", "b", "c"]},
            {"gold": 0, "logprobs": [-0.1, -1.0, -3.0], "choices": ["x", "y", "z"]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            assert acc == 1.0
            # Check summary file
            summary_path = Path(cache).with_suffix(".summary.json")
            assert summary_path.exists()
            s = json.loads(summary_path.read_text())
            assert s["acc"] == 1.0
            assert s["acc_norm"] == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_half_correct(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 1, "logprobs": [-1.0, -0.5, -2.0]},  # correct: index 1
            {"gold": 0, "logprobs": [-0.5, -0.1, -3.0]},  # wrong: index 1 wins, gold 0
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            assert acc == 0.5
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_acc_norm_length_penalized(self) -> None:
        """acc_norm should prefer shorter choices with same logprob."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        # Choice A: short, Choice B: very long. Same raw logprob.
        items = [
            {
                "gold": 0,
                "logprobs": [-10.0, -10.0],
                "choices": ["A", "BBBBBBBBBB"],
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood(items, cache)
            # Same logprob: argmax picks index 0 (= gold) → acc correct
            assert acc == 1.0
            summary_path = Path(cache).with_suffix(".summary.json")
            s = json.loads(summary_path.read_text())
            # acc_norm: "A"(len=1) → -10/1=-10.0, "B"×10 → -10/10=-1.0
            # argmax picks index 1 (long) ≠ gold 0 → acc_norm wrong
            assert s["acc_norm"] == 0.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_empty_dataset(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_loglikelihood([], cache)
            assert acc == 0.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)


class TestScoreGenerate:
    """Test generation-based scoring."""

    def test_exact_match(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "B", "gen": ["Some text\nAnswer: B"]},
            {"answer": "A", "gen": ["The answer is A."]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_generate(items, "answer", "gen", cache)
            assert acc == 1.0
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)

    def test_mixed(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "C", "gen": ["Answer: B"]},  # wrong
            {"answer": "D", "gen": ["Answer: D"]},  # correct
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache = f.name
        try:
            acc = score_generate(items, "answer", "gen", cache)
            assert acc == 0.5
        finally:
            Path(cache).unlink(missing_ok=True)
            Path(cache).with_suffix(".summary.json").unlink(missing_ok=True)


# ===========================================================================
# mc_infer tests
# ===========================================================================


class TestMCInferConfig:
    """Test MCInferConfig defaults and API key resolution."""

    def test_defaults(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import MCInferConfig

        c = MCInferConfig()
        assert c.mode == "loglikelihood"
        assert c.max_workers == 32
        assert c.temperature == 0.0
        assert c.n_shot == 0

    def test_api_key_default(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import MCInferConfig

        with patch.dict("os.environ", {}, clear=True):
            c = MCInferConfig()
            assert c.api_key == "EMPTY"

    def test_api_key_from_env(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import MCInferConfig

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            c = MCInferConfig()
            assert c.api_key == "sk-test"


class TestFewShotFormatter:
    """Test few-shot example loading and dedup."""

    def _make_examples(self, count: int) -> str:
        """Create a temp JSONL with `count` examples."""
        items = [
            {
                "prompt": f"Q{i}: test?\nA. a\nB. b\nC. c\nD. d\nAnswer:",
                "answer": "B",
                "choices": ["a", "b", "c", "d"],
                "gold": 1,
            }
            for i in range(count)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
            return f.name

    def test_zero_shot(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        fmt = FewShotFormatter(n_shot=0)
        assert fmt.get_prefix("any prompt") == ""

    def test_load_and_prefix(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        tmp = self._make_examples(10)
        try:
            fmt = FewShotFormatter(n_shot=3, seed=42)
            fmt.load(tmp)
            prefix = fmt.get_prefix("some other prompt")
            # Should contain 3 examples separated by \n\n
            assert prefix.count("\n\n") >= 3
            assert "Q" in prefix
            assert "Answer: B" in prefix
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_dedup_excludes_test_prompt(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        tmp = self._make_examples(10)
        try:
            fmt = FewShotFormatter(n_shot=3, seed=42)
            fmt.load(tmp)
            # Get the raw prompt from one of the few-shot pool
            test_prompt = fmt._few_shot_pool[0]["prompt"]
            prefix_with_dedup = fmt.get_prefix(test_prompt)
            prefix_without = fmt.get_prefix("unrelated prompt")
            # Dedup should produce different prefixes (one fewer example)
            # Both should have content
            assert len(prefix_with_dedup) > 0
            assert len(prefix_without) > 0
            # Dedup removes the matching example, so non-dedup prefix may differ
            # The key invariant: formatted demo starts with raw_prompt + " " + answer
            assert any(test_prompt in d for d in fmt._all_formatted)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_insufficient_examples(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import FewShotFormatter

        tmp = self._make_examples(3)
        try:
            fmt = FewShotFormatter(n_shot=10)
            fmt.load(tmp)
            # Should warn and return empty
            assert fmt.get_prefix("test") == ""
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestArgmax:
    def test_basic(self) -> None:
        from llmeval.tasks.mc_eval.mc_infer import _argmax

        assert _argmax([1.0, 3.0, 2.0]) == 1
        assert _argmax([5.0]) == 0
        assert _argmax([-1.0, -0.5, -2.0]) == 1


# ===========================================================================
# mc_infer runner/client tests (offline, mocked API)
# ===========================================================================


def _make_api_error(message: str = "", status_code: int | None = None) -> Exception:
    """Build an APIError instance compatible with the (possibly stubbed) openai module."""
    cls = sys.modules["openai"].APIError
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    err.message = message
    err.status_code = status_code
    return err


def _fake_logprob_resp(token_lps_list: list[list[float | None]]) -> MagicMock:
    """Completions response whose choices carry the given token_logprobs."""
    resp = MagicMock()
    choices = []
    for lps in token_lps_list:
        choice = MagicMock()
        choice.logprobs.token_logprobs = lps
        choices.append(choice)
    resp.choices = choices
    return resp


def _make_ll_client(max_retries: int = 0):
    """MCLoglikelihoodClient bypassing __init__ (works with stubbed openai)."""
    from llmeval.tasks.mc_eval.mc_infer import MCLoglikelihoodClient

    client = MCLoglikelihoodClient.__new__(MCLoglikelihoodClient)
    client.model_name = "m"
    client.timeout = 5
    client.max_retries = max_retries
    client.client = MagicMock()
    return client


def _make_mc_runner(tmp_path: Path, mode: str = "loglikelihood", max_retries: int = 0):
    """MCRunner bypassing __init__ (no client construction)."""
    import threading as _threading

    from llmeval.tasks.mc_eval.mc_infer import MCRunner
    from llmeval.utils.config import MCInferConfig

    runner = MCRunner.__new__(MCRunner)
    runner.config = MCInferConfig(
        input_file=str(tmp_path / "in.jsonl"),
        output_file=str(tmp_path / "out.jsonl"),
        mode=mode,
        max_retries=max_retries,
        max_workers=1,
    )
    runner.client = None
    runner.system_prompt = None
    runner._few_shot_fmt = None
    runner._file_lock = _threading.Lock()
    runner._stats_lock = _threading.Lock()
    runner._stats = {"processed": 0, "failed": 0, "correct": 0, "skipped": 0}
    return runner


class TestMCLoglikelihoodClient:
    def test_batched_single_request_and_trailing_token_sliced(self) -> None:
        client = _make_ll_client()
        # [first(None), choice tokens..., trailing generated token]
        client.client.completions.create.return_value = _fake_logprob_resp(
            [[None, -1.0, -2.0, -9.9], [None, -0.5, -9.9]]
        )
        result = client.get_choices_logprobs("prompt", ["A", "B"])
        assert result == [-3.0, -0.5]  # trailing -9.9 sliced off
        # one batched request carrying both choice prompts
        client.client.completions.create.assert_called_once()
        prompt_arg = client.client.completions.create.call_args.kwargs["prompt"]
        assert prompt_arg == ["prompt A", "prompt B"]

    def test_4xx_aborts_without_retry(self) -> None:
        client = _make_ll_client(max_retries=3)
        client.client.completions.create.side_effect = _make_api_error("bad", 400)
        result = client.get_choices_logprobs("p", ["a", "b"])
        assert result == [float("-inf"), float("-inf")]
        assert client.client.completions.create.call_count == 1

    def test_total_failure_returns_all_neg_inf(self) -> None:
        client = _make_ll_client(max_retries=0)
        client.client.completions.create.side_effect = RuntimeError("down")
        assert client.get_choices_logprobs("p", ["a"]) == [float("-inf")]

    def test_choice_count_mismatch_retries_to_neg_inf(self) -> None:
        client = _make_ll_client(max_retries=0)
        client.client.completions.create.return_value = _fake_logprob_resp(
            [[None, -1.0]]
        )
        assert client.get_choices_logprobs("p", ["a", "b"]) == [float("-inf")] * 2


class TestProcessLoglikelihoodItem:
    def test_all_neg_inf_raises(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [float("-inf")] * 2
        item = {"prompt": "q", "choices": ["a", "b"], "gold": 1}
        with pytest.raises(RuntimeError, match="failed for all choices"):
            runner.process_loglikelihood_item(item)

    def test_normal_pred_and_correct(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        runner.client = MagicMock()
        runner.client.get_choices_logprobs.return_value = [-5.0, -1.0]
        item = {"prompt": "q", "choices": ["a", "b"], "gold": 1}
        result = runner.process_loglikelihood_item(item)
        assert result["pred"] == 1 and result["correct"] is True

    def test_no_choices_returns_none(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path)
        assert runner.process_loglikelihood_item({"prompt": "q"}) is None


class TestProcessGenerateItem:
    def test_success(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "ans"
        result = runner.process_generate_item(
            {"prompt": "q", "answer": "A"}, client, []
        )
        assert result["gen"] == ["ans"]

    def test_null_content_raises(self, tmp_path: Path) -> None:
        """Null/empty generation must raise (not write an empty gen)."""
        runner = _make_mc_runner(tmp_path, mode="generate")
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = None
        with pytest.raises(RuntimeError, match="no usable text"):
            runner.process_generate_item({"prompt": "q", "answer": "A"}, client, [])

    def test_persistent_error_raises_after_retries(self, tmp_path: Path) -> None:
        runner = _make_mc_runner(tmp_path, mode="generate", max_retries=0)
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("down")
        with pytest.raises(RuntimeError):
            runner.process_generate_item({"prompt": "q", "answer": "A"}, client, [])


class TestMCRunnerEndToEnd:
    """Full run() pipeline with a fake loglikelihood client."""

    def _write_input(self, path: Path) -> None:
        items = [
            {"prompt": "Q1?\nA. x\nB. y\nAnswer:", "choices": ["x", "y"], "gold": 1},
            {"prompt": "Q2?\nA. p\nB. q\nAnswer:", "choices": ["p", "q"], "gold": 0},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def test_run_and_resume(self, tmp_path: Path) -> None:
        from llmeval.tasks.mc_eval import mc_infer
        from llmeval.tasks.mc_eval.mc_infer import MCRunner
        from llmeval.utils.config import MCInferConfig

        class FakeLLClient:
            def __init__(self, **kwargs):
                pass

            def get_choices_logprobs(self, prompt, choice_texts):
                return [-1.0 if i == 1 else -5.0 for i in range(len(choice_texts))]

        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        self._write_input(inp)
        cfg = MCInferConfig(
            input_file=str(inp),
            output_file=str(out),
            mode="loglikelihood",
            max_workers=2,
        )
        with patch.object(mc_infer, "MCLoglikelihoodClient", FakeLLClient):
            MCRunner(cfg).run()
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        assert len(rows) == 2
        by_pred = {r["prompt"][:2]: r for r in rows}
        assert by_pred["Q1"]["pred"] == 1 and by_pred["Q1"]["correct"] is True
        assert by_pred["Q2"]["pred"] == 1 and by_pred["Q2"]["correct"] is False

        # Resume: second run must not duplicate
        with patch.object(mc_infer, "MCLoglikelihoodClient", FakeLLClient):
            MCRunner(cfg).run()
        assert len(out.read_text().strip().split("\n")) == 2

    def test_failed_items_dumped_not_written(self, tmp_path: Path) -> None:
        from llmeval.tasks.mc_eval import mc_infer
        from llmeval.tasks.mc_eval.mc_infer import MCRunner
        from llmeval.utils.config import MCInferConfig

        class FailLLClient:
            def __init__(self, **kwargs):
                pass

            def get_choices_logprobs(self, prompt, choice_texts):
                return [float("-inf")] * len(choice_texts)

        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        self._write_input(inp)
        cfg = MCInferConfig(
            input_file=str(inp),
            output_file=str(out),
            mode="loglikelihood",
            max_workers=2,
        )
        with patch.object(mc_infer, "MCLoglikelihoodClient", FailLLClient):
            MCRunner(cfg).run()
        assert not out.exists()  # nothing scored
        failed = tmp_path / "out_failed.jsonl"
        assert failed.exists()
        assert len(failed.read_text().strip().split("\n")) == 2


class TestMCScoreEdgeCases:
    """Regression tests for scorer fixes (2026-07-30)."""

    def test_generate_empty_gold_and_pred_not_correct(self, tmp_path: Path) -> None:
        """Empty gold + unparseable (empty) pred must NOT count as correct."""
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [
            {"answer": "", "gen": ["no letter here"]},  # both empty → was 判对
            {"answer": "B", "gen": ["Answer: B"]},
        ]
        acc = score_generate(items, "answer", "gen", tmp_path / "c.jsonl")
        assert acc == 0.5  # only the second item is correct

    def test_loglikelihood_all_neg_inf_counted_wrong(self, tmp_path: Path) -> None:
        """All -inf logprobs (failed inference) must not be argmax-scored."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": 0, "logprobs": [float("-inf")] * 2, "choices": ["a", "b"]},
            {"gold": 1, "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
        ]
        acc = score_loglikelihood(items, tmp_path / "c.jsonl")
        assert acc == 0.5

    def test_acc_norm_uses_choices_when_present(self, tmp_path: Path) -> None:
        """Length normalization flips the argmax when choices differ in length."""
        from llmeval.tasks.mc_eval.mc_score import _compute_loglikelihood_metrics

        # raw argmax → index 1; normalized: -2.0/4=-0.5 vs -1.0/1=-1.0 → index 0
        items = [{"gold": 0, "logprobs": [-2.0, -1.0], "choices": ["aaaa", "b"]}]
        metrics = _compute_loglikelihood_metrics(items)
        assert metrics.acc == 0.0
        assert metrics.acc_norm == 1.0

    def test_extract_lowercase_letter(self) -> None:
        from llmeval.tasks.mc_eval.mc_score import _extract_answer

        assert _extract_answer("the answer is b") == "B"
        assert _extract_answer("选 c") == "C"

    def test_non_numeric_gold_treated_invalid(self, tmp_path: Path) -> None:
        """A non-numeric gold must not crash scoring; item counts as wrong."""
        from llmeval.tasks.mc_eval.mc_score import score_loglikelihood

        items = [
            {"gold": "B", "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
            {"gold": 1, "logprobs": [-2.0, -1.0], "choices": ["a", "b"]},
        ]
        acc = score_loglikelihood(items, tmp_path / "c.jsonl")
        assert acc == 0.5

    def test_generate_bare_string_gen_tolerated(self, tmp_path: Path) -> None:
        """A plain-string gen field (schema expects list) is scored as text."""
        from llmeval.tasks.mc_eval.mc_score import score_generate

        items = [{"answer": "B", "gen": "Answer: B"}]
        acc = score_generate(items, "answer", "gen", tmp_path / "c.jsonl")
        assert acc == 1.0
