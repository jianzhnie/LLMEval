"""Backend-independent request and result schemas for model scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

LOGLIKELIHOOD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoglikelihoodRequest:
    """A context with one or more continuations to score.

    The scoring boundary follows ``lm-evaluation-harness``: trailing whitespace
    belongs to the continuation for tokenization purposes. Metric normalization
    still uses the original continuation text.
    """

    context: str
    continuations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, str):
            raise TypeError("loglikelihood context must be a string")
        if not isinstance(self.continuations, tuple) or not self.continuations:
            raise ValueError("loglikelihood request requires at least one continuation")
        if any(
            not isinstance(continuation, str) or not continuation
            for continuation in self.continuations
        ):
            raise ValueError("loglikelihood continuations must be non-empty strings")

    @property
    def scoring_context(self) -> str:
        """Context after moving trailing whitespace to each continuation."""
        return self.context.rstrip()

    @property
    def continuation_prefix(self) -> str:
        """Trailing context whitespace scored with every continuation."""
        return self.context[len(self.scoring_context) :]

    def scored_continuation(self, index: int) -> str:
        """Return the exact text whose token logprobs must be summed."""
        return f"{self.continuation_prefix}{self.continuations[index]}"

    def cache_identity(self) -> dict[str, Any]:
        """Return the backend-independent portion of a cache key."""
        return {
            "schema_version": LOGLIKELIHOOD_SCHEMA_VERSION,
            "context": self.context,
            "continuations": list(self.continuations),
        }


@dataclass(frozen=True)
class ChoiceLoglikelihood:
    """Validated token-level score for one continuation."""

    continuation: str
    scored_text: str
    token_logprobs: tuple[float, ...] = ()
    token_texts: tuple[str, ...] = ()
    token_ids: tuple[int, ...] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error is not None:
            if self.token_logprobs or self.token_texts or self.token_ids is not None:
                raise ValueError("failed choice results cannot contain token scores")
            return
        if not self.token_logprobs:
            raise ValueError("successful choice result requires token logprobs")
        if len(self.token_texts) != len(self.token_logprobs):
            raise ValueError("token text and logprob lengths must match")
        if "".join(self.token_texts) != self.scored_text:
            raise ValueError("token text does not reconstruct the scored continuation")
        if self.token_ids is not None and len(self.token_ids) != len(
            self.token_logprobs
        ):
            raise ValueError("token ID and logprob lengths must match")
        if any(not math.isfinite(value) for value in self.token_logprobs):
            raise ValueError("token logprobs must be finite")

    @property
    def complete(self) -> bool:
        """Whether this choice has a complete, aligned token score."""
        return self.error is None

    @property
    def total_logprob(self) -> float:
        """Sum token logprobs for the complete continuation."""
        return sum(self.token_logprobs) if self.complete else float("-inf")

    @property
    def token_count(self) -> int:
        return len(self.token_logprobs)

    @property
    def char_count(self) -> int:
        return len(self.continuation)

    @property
    def byte_count(self) -> int:
        return len(self.continuation.encode("utf-8"))

    @classmethod
    def failure(
        cls, continuation: str, scored_text: str, error: str
    ) -> ChoiceLoglikelihood:
        """Build an explicitly failed choice result."""
        return cls(continuation=continuation, scored_text=scored_text, error=error)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this choice for a content-addressed cache entry."""
        return {
            "continuation": self.continuation,
            "scored_text": self.scored_text,
            "token_logprobs": list(self.token_logprobs),
            "token_texts": list(self.token_texts),
            "token_ids": list(self.token_ids) if self.token_ids is not None else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChoiceLoglikelihood:
        """Deserialize and validate a cached choice result."""
        continuation = value.get("continuation")
        scored_text = value.get("scored_text")
        raw_logprobs = value.get("token_logprobs")
        raw_tokens = value.get("token_texts")
        raw_ids = value.get("token_ids")
        if not isinstance(continuation, str) or not isinstance(scored_text, str):
            raise ValueError("cached choice text fields must be strings")
        if not isinstance(raw_logprobs, list) or not isinstance(raw_tokens, list):
            raise ValueError("cached choice token fields must be lists")
        if any(
            isinstance(item, bool) or not isinstance(item, int | float)
            for item in raw_logprobs
        ):
            raise ValueError("cached choice logprobs must be numeric")
        if any(not isinstance(item, str) for item in raw_tokens):
            raise ValueError("cached choice tokens must be strings")
        if raw_ids is not None and not isinstance(raw_ids, list):
            raise ValueError("cached choice token IDs must be a list or null")
        if raw_ids is not None and any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_ids
        ):
            raise ValueError("cached choice token IDs must be integers")
        return cls(
            continuation=continuation,
            scored_text=scored_text,
            token_logprobs=tuple(float(item) for item in raw_logprobs),
            token_texts=tuple(raw_tokens),
            token_ids=tuple(raw_ids) if raw_ids is not None else None,
            error=(str(value["error"]) if value.get("error") is not None else None),
        )


@dataclass(frozen=True)
class LoglikelihoodResult:
    """Aligned continuation results for one :class:`LoglikelihoodRequest`."""

    request: LoglikelihoodRequest
    choices: tuple[ChoiceLoglikelihood, ...]
    exact: bool
    error: str | None = None

    def __post_init__(self) -> None:
        if len(self.choices) != len(self.request.continuations):
            raise ValueError("choice result count does not match the request")
        for index, choice in enumerate(self.choices):
            if choice.continuation != self.request.continuations[index]:
                raise ValueError(
                    "choice result continuation order does not match request"
                )
            if choice.scored_text != self.request.scored_continuation(index):
                raise ValueError("choice result scoring text does not match request")
        if self.exact and (self.error is not None or not all(self.complete_choices)):
            raise ValueError("an exact result requires every choice to be complete")

    @property
    def complete_choices(self) -> tuple[bool, ...]:
        return tuple(choice.complete for choice in self.choices)

    @property
    def complete(self) -> bool:
        """Whether the result is exact and every requested choice was scored."""
        return self.exact and self.error is None and all(self.complete_choices)

    @classmethod
    def failure(cls, request: LoglikelihoodRequest, error: str) -> LoglikelihoodResult:
        """Build a result that fails every continuation with the same reason."""
        return cls(
            request=request,
            choices=tuple(
                ChoiceLoglikelihood.failure(
                    continuation,
                    request.scored_continuation(index),
                    error,
                )
                for index, continuation in enumerate(request.continuations)
            ),
            exact=False,
            error=error,
        )

    def to_cache_value(self) -> dict[str, Any]:
        """Serialize a complete exact result for persistent caching."""
        if not self.complete:
            raise ValueError("incomplete loglikelihood results cannot be cached")
        return {
            "result_schema_version": LOGLIKELIHOOD_SCHEMA_VERSION,
            "request": self.request.cache_identity(),
            "choices": [choice.to_dict() for choice in self.choices],
            "exact": True,
        }

    @classmethod
    def from_cache_value(
        cls, request: LoglikelihoodRequest, value: dict[str, Any]
    ) -> LoglikelihoodResult:
        """Deserialize a cache entry and reject stale or misaligned shapes."""
        if value.get("result_schema_version") != LOGLIKELIHOOD_SCHEMA_VERSION:
            raise ValueError("unsupported loglikelihood cache schema")
        if value.get("request") != request.cache_identity():
            raise ValueError("cached loglikelihood request does not match")
        raw_choices = value.get("choices")
        if not isinstance(raw_choices, list):
            raise ValueError("cached loglikelihood choices must be a list")
        if value.get("exact") is not True:
            raise ValueError("cached loglikelihood result must be exact")
        result = cls(
            request=request,
            choices=tuple(
                ChoiceLoglikelihood.from_dict(choice)
                for choice in raw_choices
                if isinstance(choice, dict)
            ),
            exact=True,
        )
        if not result.complete:
            raise ValueError("cached loglikelihood result is incomplete")
        return result
