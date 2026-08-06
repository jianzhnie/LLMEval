"""Backend-independent request and result schemas for model scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass


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
