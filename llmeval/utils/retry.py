"""Shared retry utilities for OpenAI-compatible API clients.

Used by llmeval/inference/online.py and llmeval/inference/mc.py
so both classify and back off from API failures identically.

``should_retry`` holds the classification policy (the single definition);
``call_with_retry`` runs an attempt loop under that policy so call sites
share one loop implementation as well.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from openai import APIConnectionError, APIError, RateLimitError

from llmeval.utils.log import init_logger

logger = init_logger("api_retry")

T = TypeVar("T")


class ClientError(RuntimeError):
    """Custom exception class for client-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize ClientError with message and optional original error.

        Args:
            message: Error message describing the issue
            original_error: The original exception that caused this error
        """
        super().__init__(message)
        self.original_error = original_error


class MalformedResponseError(RuntimeError):
    """The API response payload is structurally malformed.

    Raised by response-structure probes (empty choices, missing fields) to
    mark a non-APIError failure that IS worth retrying: a malformed payload
    is a transient backend quirk, not a programming error, so it follows the
    normal bounded retry path instead of being re-raised immediately.
    """


def error_message(e: APIError) -> str:
    """Best-effort human-readable message from an APIError (message may be None)."""
    return getattr(e, "message", None) or str(e)


# Known server phrasings for prompt-exceeds-max-context rejections.
_CONTEXT_LENGTH_PHRASES = (
    "maximum context length",
    "max context length",
    "context length exceeded",
    "context window",
    "context_length",
)


def is_context_length_error(e: APIError) -> bool:
    """Whether the error is a prompt-exceeds-max-context-length rejection."""
    msg = error_message(e).lower()
    return any(phrase in msg for phrase in _CONTEXT_LENGTH_PHRASES)


def non_retryable_client_error(e: APIError) -> str | None:
    """Return a reason string for non-retryable 4xx errors, else None.

    4xx client errors (invalid request, auth, not found, ...) never succeed on
    retry, so retrying only wastes backoff time. 408 Request Timeout stays
    retryable; 429 is a RateLimitError subclass handled by callers separately.
    """
    status_code = getattr(e, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 408:
        return f"non-retryable API error (status={status_code}): {error_message(e)}"
    return None


def retry_backoff(attempt: int, max_retries: int, reason: str) -> None:
    """Sleep with exponential backoff plus jitter before the next retry.

    Args:
        attempt: Zero-based index of the attempt that just failed
        max_retries: Configured maximum number of retries (for logging)
        reason: Short failure description included in the warning log
    """
    sleep_time = min(60, 2 ** (attempt + 1)) + random.randint(0, 5)
    logger.warning(
        f"{reason} on attempt {attempt + 1}/{max_retries + 1}. "
        f"Sleeping for {sleep_time:.2f}s."
    )
    time.sleep(sleep_time)


def should_retry(exc: Exception, attempt: int, max_retries: int) -> bool | None:
    """Classify a failed API call; back off when it is worth retrying.

    This is the single retry policy shared by the online and MC clients.

    Returns
    -------
    ``True``
        The error is retryable and backoff has already been applied — the
        caller should make another attempt.
    ``None``
        Context-length rejection: the prompt can never fit, so the caller
        maps it to an empty result instead of retrying or failing.

    Raises
    ------
    ClientError
        Non-retryable 4xx (retrying cannot succeed — fail fast), or
        *max_retries* exhausted.
    Exception
        Any other non-APIError (TypeError, KeyError, ...) is re-raised
        as-is: these are programming errors, not transient failures, and
        retrying them would only hide the root cause. The one exception is
        :class:`MalformedResponseError`, which marks a structurally broken
        response payload and follows the normal bounded retry path.
    """
    # Some OpenAI-compatible gateways wrap prompt-length rejections in a 429.
    # Classify by the explicit message before applying subclass retry rules.
    if isinstance(exc, APIError) and is_context_length_error(exc):
        logger.warning("Max context length exceeded, returning empty result")
        return None

    # Connection / rate-limit errors are APIError subclasses, so they must be
    # excluded from the remaining fatal-4xx checks.
    if isinstance(exc, APIError) and not isinstance(
        exc, APIConnectionError | RateLimitError
    ):
        # 4xx (except 408/429): retrying can never succeed.
        non_retryable = non_retryable_client_error(exc)
        if non_retryable:
            raise ClientError(non_retryable, exc) from exc

    # Non-APIError exceptions other than malformed-response probes are
    # programming errors, not transient API failures — surface them
    # immediately instead of retrying.
    if not isinstance(exc, APIError | MalformedResponseError):
        raise exc

    if attempt >= max_retries:
        raise ClientError(f"Max retries exceeded: {exc!s}", exc) from exc

    if isinstance(exc, MalformedResponseError):
        reason = f"Malformed response: {exc!s}"
    elif isinstance(exc, APIConnectionError | RateLimitError):
        reason = f"{type(exc).__name__}: {exc!s}"
    else:
        reason = f"API error: {exc!s}"
    retry_backoff(attempt, max_retries, reason)
    return True


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int,
    *,
    fail_fast_exceptions: tuple[type[BaseException], ...] = (),
) -> T | None:
    """Run *fn* under the shared retry policy (see :func:`should_retry`).

    This is the single attempt loop shared by the online and MC clients; the
    per-call work (building and issuing the request) stays in *fn*.

    Returns
    -------
    fn()'s result, or ``None`` for a context-length rejection (the prompt
    can never fit — callers map this to an empty result).

    Raises
    ------
    ClientError
        Non-retryable 4xx (fail fast), or *max_retries* exhausted.
    Exception
        Non-APIError exceptions from *fn* are re-raised without retrying
        (see :func:`should_retry`).
    """
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError(
            f"max_retries must be a non-negative integer, got {max_retries!r}"
        )
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if isinstance(e, fail_fast_exceptions):
                raise
            if should_retry(e, attempt, max_retries) is None:
                return None  # context-length rejection → empty result
    raise AssertionError("retry loop exhausted without returning or raising")
