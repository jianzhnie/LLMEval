"""Shared retry utilities for OpenAI-compatible API clients.

Used by llmeval/vllm/online_server.py and llmeval/tasks/mc_eval/mc_infer.py
so both classify and back off from API failures identically.
"""

from __future__ import annotations

import random
import time

from openai import APIError

from llmeval.utils.logger import init_logger

logger = init_logger("api_retry")


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


def error_message(e: APIError) -> str:
    """Best-effort human-readable message from an APIError (message may be None)."""
    return getattr(e, "message", None) or str(e)


def is_context_length_error(e: APIError) -> bool:
    """Whether the error is a prompt-exceeds-max-context-length rejection."""
    return "maximum context length" in error_message(e)


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
    sleep_time = (2 ** (attempt + 1)) + random.randint(0, 5)
    logger.warning(
        f"{reason} on attempt {attempt + 1}/{max_retries + 1}. "
        f"Sleeping for {sleep_time:.2f}s."
    )
    time.sleep(sleep_time)


def should_retry(exc: Exception, attempt: int, max_retries: int) -> bool | None:
    """Decide whether *exc* is retryable after a failed API call.

    Returns
    -------
    ``True`` when the caller should retry (backoff is already applied).
    ``False`` when the error is fatal and retries should stop.
    ``None`` when this helper cannot classify the exception (caller decides).
    """
    from openai import APIConnectionError, APIError, RateLimitError

    if isinstance(exc, (APIConnectionError, RateLimitError)):
        if attempt < max_retries:
            retry_backoff(attempt, max_retries, f"{type(exc).__name__}: {exc!s}")
            return True
        return False

    if isinstance(exc, APIError):
        non_retryable = non_retryable_client_error(exc)
        if non_retryable:
            logger.warning(f"Request aborted: {non_retryable}")
            return False
        if attempt < max_retries:
            retry_backoff(attempt, max_retries, f"API error: {exc!s}")
            return True
        return False

    if attempt < max_retries:
        retry_backoff(attempt, max_retries, f"Unexpected {type(exc).__name__}: {exc!s}")
        return True
    return False
