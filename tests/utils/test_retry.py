"""Tests for llmeval.utils.retry — the shared retry policy and attempt loop.

Covers the should_retry tri-state contract (True / None / raises ClientError)
and the call_with_retry executor used by the online and MC clients.  The
429-must-retry cases guard against the APIError-subclass misclassification
(RateLimitError carries a 4xx status code but is always worth retrying).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Mock heavy dependencies (same pattern as the other test modules) ──
def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


for mod_name in ("openai", "httpx"):
    if mod_name not in sys.modules and not importlib.util.find_spec(mod_name):
        sys.modules[mod_name] = _make_stub(mod_name)

_openai_mod = sys.modules.get("openai")
if _openai_mod is not None:
    for _exc in ("APIConnectionError", "APIError", "RateLimitError"):
        if not hasattr(_openai_mod, _exc):
            setattr(_openai_mod, _exc, type(_exc, (Exception,), {}))

from llmeval.utils.retry import (
    ClientError,
    MalformedResponseError,
    call_with_retry,
    error_message,
    is_context_length_error,
    non_retryable_client_error,
    retry_backoff,
    should_retry,
)

# ── helpers ───────────────────────────────────────────────────────


def _make_error(cls_name: str, message: str = "", status_code: int | None = None):
    """Build an openai exception instance without invoking its __init__."""
    cls = getattr(sys.modules["openai"], cls_name)
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    err.message = message
    err.status_code = status_code
    return err


def _api_error(message: str = "", status_code: int | None = None):
    return _make_error("APIError", message, status_code)


def _rate_limit_error(message: str = "rate limited"):
    return _make_error("RateLimitError", message, 429)


def _connection_error(message: str = "connection refused"):
    return _make_error("APIConnectionError", message)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff sleeps are irrelevant to behavior — skip them."""
    monkeypatch.setattr(time, "sleep", lambda s: None)


# ── error classifiers ─────────────────────────────────────────────


class TestClassifiers:
    def test_context_length_detected(self) -> None:
        assert is_context_length_error(
            _api_error("This model's maximum context length is 8192")
        )

    def test_context_length_not_detected(self) -> None:
        assert not is_context_length_error(_api_error("some other failure"))

    def test_non_retryable_4xx(self) -> None:
        reason = non_retryable_client_error(_api_error("bad request", 400))
        assert reason is not None and "non-retryable" in reason

    def test_408_stays_retryable(self) -> None:
        assert non_retryable_client_error(_api_error("timeout", 408)) is None

    def test_5xx_stays_retryable(self) -> None:
        assert non_retryable_client_error(_api_error("boom", 500)) is None

    def test_missing_status_stays_retryable(self) -> None:
        assert non_retryable_client_error(_api_error("weird")) is None


# ── should_retry tri-state contract ───────────────────────────────


class TestShouldRetry:
    def test_rate_limit_retries_and_never_raises(self) -> None:
        """429 is an APIError subclass with a 4xx status — must NOT be fatal."""
        assert should_retry(_rate_limit_error(), 0, 3) is True

    def test_rate_limit_exhaustion_raises(self) -> None:
        with pytest.raises(ClientError, match="Max retries exceeded"):
            should_retry(_rate_limit_error(), 3, 3)

    def test_connection_error_retries(self) -> None:
        assert should_retry(_connection_error(), 0, 2) is True

    def test_non_retryable_4xx_raises_immediately(self) -> None:
        with pytest.raises(ClientError, match="non-retryable"):
            should_retry(_api_error("invalid", 400), 0, 3)

    def test_context_length_returns_none(self) -> None:
        err = _api_error("This model's maximum context length is 8192", 400)
        assert should_retry(err, 0, 3) is None

    def test_context_length_returns_none_even_at_last_attempt(self) -> None:
        """Context-length is an unfit prompt, not an exhaustion case."""
        err = _api_error("This model's maximum context length is 8192", 400)
        assert should_retry(err, 3, 3) is None

    def test_rate_limit_wrapped_context_length_returns_none(self) -> None:
        err = _rate_limit_error("maximum context length exceeded")
        assert should_retry(err, 0, 3) is None

    def test_5xx_retries_then_exhaustion_raises(self) -> None:
        err = _api_error("boom", 500)
        assert should_retry(err, 0, 1) is True
        with pytest.raises(ClientError, match="Max retries exceeded"):
            should_retry(err, 1, 1)

    def test_unexpected_error_is_reraised_without_retry(self) -> None:
        """Non-APIError exceptions are programming errors — surface, don't retry."""
        with pytest.raises(RuntimeError, match="weird"):
            should_retry(RuntimeError("weird"), 0, 2)

    def test_unexpected_error_is_reraised_even_at_first_attempt(self) -> None:
        with pytest.raises(RuntimeError, match="weird"):
            should_retry(RuntimeError("weird"), 0, 3)

    def test_malformed_response_retries(self) -> None:
        """A structurally broken payload is transient — worth retrying."""
        assert should_retry(MalformedResponseError("empty choices"), 0, 3) is True

    def test_malformed_response_exhaustion_raises(self) -> None:
        with pytest.raises(ClientError, match="Max retries exceeded"):
            should_retry(MalformedResponseError("empty choices"), 3, 3)


# ── call_with_retry attempt loop ──────────────────────────────────


class TestCallWithRetry:
    @pytest.mark.parametrize("max_retries", [-1, 1.5, True])
    def test_invalid_retry_count_is_rejected(self, max_retries: object) -> None:
        fn = MagicMock()
        with pytest.raises(ValueError, match="max_retries"):
            call_with_retry(fn, max_retries)  # type: ignore[arg-type]
        fn.assert_not_called()

    def test_fail_fast_exception_is_not_retried(self) -> None:
        class AlignmentError(ValueError):
            pass

        fn = MagicMock(side_effect=AlignmentError("token boundary"))
        with pytest.raises(AlignmentError):
            call_with_retry(fn, 3, fail_fast_exceptions=(AlignmentError,))
        assert fn.call_count == 1

    def test_success_first_attempt(self) -> None:
        fn = MagicMock(return_value="ok")
        assert call_with_retry(fn, 3) == "ok"
        assert fn.call_count == 1

    def test_retry_then_success(self) -> None:
        fn = MagicMock(side_effect=[_connection_error(), "ok"])
        assert call_with_retry(fn, 3) == "ok"
        assert fn.call_count == 2

    def test_context_length_returns_none(self) -> None:
        fn = MagicMock(
            side_effect=_api_error("This model's maximum context length is 8192", 400)
        )
        assert call_with_retry(fn, 3) is None
        assert fn.call_count == 1  # unfit prompt: no retries at all

    def test_non_retryable_raises_after_one_attempt(self) -> None:
        fn = MagicMock(side_effect=_api_error("invalid", 400))
        with pytest.raises(ClientError, match="non-retryable"):
            call_with_retry(fn, 3)
        assert fn.call_count == 1

    def test_exhaustion_raises_and_counts_attempts(self) -> None:
        fn = MagicMock(side_effect=_api_error("boom", 500))
        with pytest.raises(ClientError, match="Max retries exceeded"):
            call_with_retry(fn, 2)
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_non_api_error_propagates_without_retry(self) -> None:
        fn = MagicMock(side_effect=RuntimeError("down"))
        with pytest.raises(RuntimeError, match="down"):
            call_with_retry(fn, 3)
        assert fn.call_count == 1

    def test_malformed_response_retried_then_success(self) -> None:
        fn = MagicMock(side_effect=[MalformedResponseError("empty choices"), "ok"])
        assert call_with_retry(fn, 3) == "ok"
        assert fn.call_count == 2

    def test_malformed_response_exhaustion_raises(self) -> None:
        fn = MagicMock(side_effect=MalformedResponseError("empty choices"))
        with pytest.raises(ClientError, match="Max retries exceeded"):
            call_with_retry(fn, 2)
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_return_value_passthrough(self) -> None:
        payload: dict[str, Any] = {"choices": [1, 2]}
        assert call_with_retry(lambda: payload, 1) is payload


# ===========================================================================
# Low-level helper coverage (merged from the former test_api_retry.py)
# ===========================================================================


def _duck_error(message: str = "", status_code: int | None = None) -> Exception:
    """Duck-typed APIError stand-in (the retry helpers only use getattr)."""
    err = Exception(message)
    err.message = message  # type: ignore[attr-defined]
    err.status_code = status_code  # type: ignore[attr-defined]
    return err


class TestClientError:
    def test_message_and_original(self) -> None:
        orig = ValueError("boom")
        err = ClientError("wrapped", orig)
        assert isinstance(err, RuntimeError)
        assert str(err) == "wrapped"
        assert err.original_error is orig

    def test_original_optional(self) -> None:
        assert ClientError("solo").original_error is None


class TestErrorMessage:
    def test_uses_message_attr(self) -> None:
        assert error_message(_duck_error("hello", 500)) == "hello"

    def test_falls_back_to_str_when_message_none(self) -> None:
        err = _duck_error("", 500)
        err.message = None  # type: ignore[attr-defined]
        assert error_message(err) == str(err)


class TestIsContextLengthError:
    def test_matches(self) -> None:
        err = _duck_error("This model's maximum context length is 8192 tokens")
        assert is_context_length_error(err) is True

    def test_no_match(self) -> None:
        assert is_context_length_error(_duck_error("rate limited")) is False


class TestNonRetryableClientError:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_non_retryable(self, status: int) -> None:
        reason = non_retryable_client_error(_duck_error("bad", status))
        assert reason is not None
        assert f"status={status}" in reason

    @pytest.mark.parametrize("status", [408, 500, 502, 503, None])
    def test_retryable_statuses(self, status: int | None) -> None:
        assert non_retryable_client_error(_duck_error("x", status)) is None


class TestRetryBackoff:
    def test_sleeps_with_exponential_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", slept.append)

        retry_backoff(attempt=0, max_retries=3, reason="test")

        assert len(slept) == 1
        # 2 ** (attempt + 1) + jitter(0..5)
        assert 2.0 <= slept[0] <= 7.0

    def test_backoff_grows_with_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", slept.append)

        retry_backoff(attempt=0, max_retries=3, reason="a")
        retry_backoff(attempt=2, max_retries=3, reason="b")

        # attempt 2 base is 2 ** 3 = 8, attempt 0 base is 2 — disjoint ranges
        assert slept[0] <= 7.0 < slept[1]
