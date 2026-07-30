"""Tests for llmeval.utils.api_retry (shared retry utilities)."""

from __future__ import annotations

import time

import pytest

from llmeval.utils.api_retry import (
    ClientError,
    error_message,
    is_context_length_error,
    non_retryable_client_error,
    retry_backoff,
)


def _make_error(message: str = "", status_code: int | None = None) -> Exception:
    """Duck-typed APIError stand-in (the api_retry helpers only use getattr)."""
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
        assert error_message(_make_error("hello", 500)) == "hello"

    def test_falls_back_to_str_when_message_none(self) -> None:
        err = _make_error("", 500)
        err.message = None  # type: ignore[attr-defined]
        assert error_message(err) == str(err)


class TestIsContextLengthError:
    def test_matches(self) -> None:
        err = _make_error("This model's maximum context length is 8192 tokens")
        assert is_context_length_error(err) is True

    def test_no_match(self) -> None:
        assert is_context_length_error(_make_error("rate limited")) is False


class TestNonRetryableClientError:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_non_retryable(self, status: int) -> None:
        reason = non_retryable_client_error(_make_error("bad", status))
        assert reason is not None
        assert f"status={status}" in reason

    @pytest.mark.parametrize("status", [408, 500, 502, 503, None])
    def test_retryable_statuses(self, status: int | None) -> None:
        assert non_retryable_client_error(_make_error("x", status)) is None


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
