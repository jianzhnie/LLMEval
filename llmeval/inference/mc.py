"""Multiple-choice inference: answer-token loglikelihood and generation modes.

This module mirrors llmeval/inference/online.py in structure, naming, and
documentation style so the two can be reviewed side by side with a file diff:
- MCLoglikelihoodClient  ~  InferenceClient
- MCRunner               ~  InferenceRunner
- main() (HfArgumentParser) ~  main()

Shared utilities (ClientError, retry classification, backoff) live in
llmeval/utils/retry.py; shared JSONL and resume helpers live in
llmeval/inference/common.py; the configuration dataclass lives in
llmeval/utils/config.py (MCInferArguments).
MC-specific pieces (kept deliberately): FewShotFormatter, answer-token
logprobs, and per-mode worker methods.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import openai
from tqdm import tqdm
from transformers import HfArgumentParser

from llmeval.inference.common import (
    CONTEXT_LENGTH_ERROR,
    append_jsonl,
    build_chat_messages,
    get_request_seed,
    is_explicit_tool_choice,
    is_local_endpoint,
    load_jsonl,
    load_resume_state,
    prepare_sample_requests,
    redact_config_for_logging,
    save_failed_items,
)
from llmeval.inference.schema import (
    ChoiceLoglikelihood,
    LoglikelihoodRequest,
    LoglikelihoodResult,
)
from llmeval.utils.config import MCInferArguments
from llmeval.utils.log import init_logger
from llmeval.utils.retry import (
    ClientError,
    MalformedResponseError,
    call_with_retry,
)

logger = init_logger("mc_infer")


class ContinuationAlignmentError(ValueError):
    """A deterministic backend token-offset mismatch that should not be retried."""


def _aligned_continuation_tokens(
    offsets: Sequence[int | None],
    token_logprobs: Sequence[float | None],
    tokens: Sequence[Any],
    context: str,
    continuation: str,
) -> tuple[list[int], tuple[str, ...]]:
    """Locate a continuation using character or UTF-8 byte offsets."""
    for byte_offsets in (False, True):
        text_length = (
            (lambda value: len(value.encode("utf-8"))) if byte_offsets else len
        )
        start = text_length(context)
        end = start + text_length(continuation)
        selected = [
            index
            for index, (offset, logprob) in enumerate(
                zip(offsets, token_logprobs, strict=True)
            )
            if offset is not None and start <= offset < end and logprob is not None
        ]
        if not selected or offsets[selected[0]] != start:
            continue

        selected_tokens = tuple(str(tokens[index]) for index in selected)
        expected_offset = start
        aligned = True
        for selected_index, token in zip(selected, selected_tokens, strict=True):
            if offsets[selected_index] != expected_offset:
                aligned = False
                break
            expected_offset += text_length(token)
        if (
            aligned
            and expected_offset == end
            and "".join(selected_tokens) == continuation
        ):
            return selected, selected_tokens

    raise ContinuationAlignmentError(
        "continuation token offsets do not align in characters or UTF-8 bytes"
    )


# ===========================================================================
# Few-shot formatter (MC-specific)
# ===========================================================================


class FewShotFormatter:
    """Load and format few-shot examples for MC prompts.

    Each demonstration is formatted by :meth:`_format_demo` as
    ``f"{prompt} {answer}"`` — the stored prompt already ends with
    "Answer:", so only the answer text is appended. Demonstrations are
    joined with a blank line in :meth:`get_prefix`.
    """

    def __init__(self, n_shot: int, few_shot_file: str = "", seed: int = 42) -> None:
        self.n_shot = n_shot
        self.few_shot_file = few_shot_file
        self.seed = seed
        self._few_shot_pool: list[dict[str, Any]] = []
        self._all_formatted: list[str] = []

    def load(self) -> None:
        """Load all demonstrations from the configured dev file.

        Sampling happens per test document in :meth:`get_prefix`, matching
        lm-evaluation-harness semantics and preventing one global pool from
        coupling every prompt in a run.
        """
        if self.n_shot <= 0:
            return

        try:
            items = load_jsonl(self.few_shot_file)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load few-shot data from {self.few_shot_file!r}: {exc}"
            ) from exc

        if len(items) < self.n_shot:
            raise ValueError(
                "Few-shot pool is too small: "
                f"available={len(items)}, required={self.n_shot}, "
                f"file={self.few_shot_file!r}"
            )

        self._few_shot_pool = items
        self._all_formatted = [self._format_demo(it) for it in items]
        logger.info(f"Loaded {len(items)} few-shot examples (seed={self.seed})")

    def get_prefix(self, test_prompt: str, document_id: str = "") -> str:
        """Get a deterministic, per-document prefix excluding the test item."""
        if self.n_shot <= 0:
            return ""

        candidates = [
            index
            for index, item in enumerate(self._few_shot_pool)
            if item.get("prompt", "") != test_prompt
            and (
                not str(document_id) or str(item.get("doc_id", "")) != str(document_id)
            )
        ]
        if len(candidates) < self.n_shot:
            raise ValueError(
                "Few-shot candidates are insufficient after excluding the test "
                f"document: doc_id={document_id or '<unknown>'!r}, "
                f"available={len(candidates)}, required={self.n_shot}"
            )
        seed_text = f"{self.seed}:{document_id}:{test_prompt}"
        derived_seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16)
        selected = random.Random(derived_seed).sample(candidates, self.n_shot)
        demos = [self._all_formatted[index] for index in selected]

        return "\n\n".join(demos) + "\n\n" if demos else ""

    @staticmethod
    def _format_demo(item: dict[str, Any]) -> str:
        """Format one few-shot demonstration."""
        prompt = item.get("prompt", "")
        answer = item.get("answer", "")
        # prompt already ends with "Answer:", append the answer
        return f"{prompt} {answer}"


# ===========================================================================
# Loglikelihood Client
# ===========================================================================


class MCLoglikelihoodClient:
    """Client for computing answer-token log-probabilities.

    Mirrors InferenceClient in online.py: same initialization, masked
    API-key logging, and classified retry policy.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 300,
        max_retries: int = 3,
        api_key: str | None = None,
        seed: int = 0,
        organization: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the client with API configuration.

        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint
            model_name: Served model name used in requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for transient failures
            api_key: API key; falls back to the OPENAI_API_KEY env var
            extra_body: Explicit non-standard fields for compatible providers
        """
        self.model_name: str = model_name
        self.base_url: str = base_url
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.seed = seed
        self.extra_body: dict[str, Any] = dict(extra_body or {})
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        if self.api_key == "EMPTY":
            log = logger.debug if is_local_endpoint(base_url) else logger.warning
            log("Using default 'EMPTY' API key.")

        # Initialize OpenAI client with validated configuration.
        # max_retries=0: retries are handled by call_with_retry; letting the
        # SDK retry internally would multiply attempts invisibly.
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(self.timeout),
            organization=organization,
            max_retries=0,
        )
        logger.info(
            f"Using API Key: ***, Timeout: {self.timeout}, "
            f"Max Retries: {self.max_retries}, base_url: {base_url}"
        )

    def get_choices_logprobs(
        self, prompt: str, choice_texts: list[str]
    ) -> list[float] | None:
        """Compute per-answer-token log-probabilities from first-token top_logprobs.

        Chat Completions returns the generated token and its alternatives under
        ``choices[0].logprobs.content[0].top_logprobs``. For each target choice,
        look up its logprob among those alternatives. This directly measures
        P(target_token | prompt).

        The target tokens (typically "A"/"B"/"C"/"D") are looked up in several
        common tokenizer forms: with/without leading space, upper/lower case.

        Args:
            prompt: The shared MC prompt (few-shot prefix + question)
            choice_texts: Candidate answer tokens (e.g. ["A","B","C","D"])

        Returns:
            One logprob per choice, aligned with choice_texts. Choices not found
            among the top predictions get float("-inf"). ``None`` means the
            prompt exceeded the model context and must not be retried.
        """

        def do_request() -> list[float]:
            call_args: dict[str, Any] = {
                "model": self.model_name,
                "messages": build_chat_messages(prompt, None),
                "max_completion_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 20,
                "timeout": self.timeout,
                "seed": self.seed,
            }
            if self.extra_body:
                call_args["extra_body"] = dict(self.extra_body)
            resp = self.client.chat.completions.create(**call_args)
            choices = getattr(resp, "choices", []) or []
            choice_logprobs = getattr(choices[0], "logprobs", None) if choices else None
            content = getattr(choice_logprobs, "content", None) or []
            alternatives = (
                getattr(content[0], "top_logprobs", None) if content else None
            )
            top_dict = {
                str(token): float(logprob)
                for entry in alternatives or []
                if (token := getattr(entry, "token", None)) is not None
                and (logprob := getattr(entry, "logprob", None)) is not None
            }
            results = []
            for target in choice_texts:
                best = float("-inf")
                for form in {
                    target,
                    f" {target}",
                    target.lower(),
                    f" {target.lower()}",
                }:
                    lp = top_dict.get(form)
                    if lp is not None and lp > best:
                        best = lp
                results.append(best)
            return results

        choice_logprobs = call_with_retry(do_request, self.max_retries)
        if choice_logprobs is None:
            return None
        return choice_logprobs

    def score_continuations(self, request: LoglikelihoodRequest) -> LoglikelihoodResult:
        """Score complete continuations and return a validated typed result.

        The traditional Completions endpoint is required because Chat
        Completions cannot echo prompt-token logprobs. Its ``max_tokens`` and
        ``echo`` parameters are therefore intentional, not stale Chat
        Completions fields.

        Structurally malformed responses (missing fields, mismatched choice
        counts) raise :class:`MalformedResponseError` inside the retry loop
        and are retried; deterministic token-offset mismatches
        (:class:`ContinuationAlignmentError`) and exhausted retries return a
        failed (non-exact) result instead.
        """

        def do_request() -> LoglikelihoodResult:
            prompts = [
                f"{request.context}{continuation}"
                for continuation in request.continuations
            ]
            call_args: dict[str, Any] = {
                "model": self.model_name,
                "prompt": prompts,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 20,
                "echo": True,
                "timeout": self.timeout,
                "seed": self.seed,
            }
            if self.extra_body:
                call_args["extra_body"] = dict(self.extra_body)
            response = self.client.completions.create(**call_args)
            completions = getattr(response, "choices", []) or []
            if len(completions) != len(request.continuations):
                raise MalformedResponseError(
                    "completion count does not match continuation count"
                )

            ordered_completions: list[Any | None] = [None] * len(request.continuations)
            for completion in completions:
                index = getattr(completion, "index", None)
                if (
                    type(index) is not int
                    or index < 0
                    or index >= len(request.continuations)
                    or ordered_completions[index] is not None
                ):
                    raise MalformedResponseError(
                        "completion indices are missing, invalid, or duplicated"
                    )
                ordered_completions[index] = completion

            choice_results: list[ChoiceLoglikelihood] = []
            for choice_index, (completion, continuation) in enumerate(
                zip(ordered_completions, request.continuations, strict=True)
            ):
                logprobs = getattr(completion, "logprobs", None)
                offsets = getattr(logprobs, "text_offset", None) if logprobs else None
                token_logprobs = (
                    getattr(logprobs, "token_logprobs", None) if logprobs else None
                )
                if not isinstance(offsets, list | tuple) or not isinstance(
                    token_logprobs, list | tuple
                ):
                    raise MalformedResponseError(
                        "completion is missing token offsets or logprobs"
                    )
                if len(offsets) != len(token_logprobs) or any(
                    offset is not None and not isinstance(offset, int)
                    for offset in offsets
                ):
                    raise MalformedResponseError(
                        "completion token offsets are malformed"
                    )

                tokens = getattr(logprobs, "tokens", None)
                if not isinstance(tokens, list | tuple) or len(tokens) != len(
                    token_logprobs
                ):
                    raise MalformedResponseError(
                        "completion is missing aligned token text"
                    )
                scored_text = request.scored_continuation(choice_index)
                selected_indices, selected_tokens = _aligned_continuation_tokens(
                    offsets,
                    token_logprobs,
                    tokens,
                    request.scoring_context,
                    scored_text,
                )

                backend_ids = getattr(logprobs, "token_ids", None)
                if backend_ids is not None and (
                    not isinstance(backend_ids, list | tuple)
                    or len(backend_ids) != len(token_logprobs)
                ):
                    raise MalformedResponseError("completion token IDs are malformed")
                selected_ids = (
                    tuple(int(backend_ids[index]) for index in selected_indices)
                    if backend_ids is not None
                    else None
                )
                choice_results.append(
                    ChoiceLoglikelihood(
                        continuation=continuation,
                        scored_text=scored_text,
                        token_logprobs=tuple(
                            float(token_logprobs[index]) for index in selected_indices
                        ),
                        token_texts=selected_tokens,
                        token_ids=selected_ids,
                    )
                )
            return LoglikelihoodResult(
                request=request,
                choices=tuple(choice_results),
                exact=True,
            )

        try:
            result = call_with_retry(
                do_request,
                self.max_retries,
                fail_fast_exceptions=(ContinuationAlignmentError,),
            )
        except ContinuationAlignmentError as exc:
            logger.debug("Continuation scoring fallback: %s", exc)
            return LoglikelihoodResult.failure(request, str(exc))
        except ClientError as exc:
            logger.warning("Continuation logprob request failed: %s", exc)
            return LoglikelihoodResult.failure(request, str(exc))
        except ValueError:
            # Schema/invariant errors are programming defects, not alignment
            # fallback signals. Let the per-item runner audit them explicitly.
            raise
        except Exception as exc:
            # Continuation scoring issues several completions for one item.
            # Keep an unexpected backend failure local to that item; the
            # first-token path remains fail-fast for programming errors.
            logger.warning("Continuation logprob request failed: %s", exc)
            return LoglikelihoodResult.failure(request, str(exc))
        if result is None:
            return LoglikelihoodResult.failure(request, CONTEXT_LENGTH_ERROR)
        return result


# ===========================================================================
# Runner
# ===========================================================================


class MCRunner:
    """Orchestrates MC inference with resume, threading, and stats.

    Mirrors InferenceRunner in online.py: same pipeline stages
    (load → resume filter → concurrent processing → report), the same
    thread-safety primitives, and the same failed-item persistence.
    """

    def __init__(self, config: MCInferArguments) -> None:
        """Initialize the runner with client, prompts, and thread safety setup.

        Args:
            config: MC inference configuration (see MCInferArguments)

        Raises:
            RuntimeError: If the loglikelihood client fails to initialize
        """
        self.config: MCInferArguments = config

        # Initialize client with error handling (loglikelihood mode only;
        # generate mode builds a plain OpenAI client per run)
        self.client: MCLoglikelihoodClient | None = None
        if config.mode == "loglikelihood":
            try:
                self.client = MCLoglikelihoodClient(
                    base_url=config.base_url,
                    model_name=config.model_name,
                    timeout=config.request_timeout,
                    max_retries=config.max_retries,
                    api_key=config.api_key,
                    seed=config.seed,
                    organization=config.organization,
                    extra_body=config.extra_body_dict,
                )
            except (OSError, ValueError) as e:
                raise RuntimeError(f"Failed to initialize MC client: {e}") from e

        # System prompt is resolved and validated by MCInferArguments at parse time.
        self.system_prompt: str | None = config.system_prompt

        # Few-shot formatter (per-item dedup)
        self._few_shot_fmt: FewShotFormatter | None = None
        if config.n_shot > 0:
            self._few_shot_fmt = FewShotFormatter(
                config.n_shot, config.few_shot_file, seed=config.seed
            )
            self._few_shot_fmt.load()

        # Initialize thread safety and monitoring
        self._file_lock: threading.Lock = threading.Lock()
        self._stats: dict[str, int] = {
            "processed": 0,
            "failed": 0,
            "correct": 0,
            "skipped": 0,
        }
        self._stats_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    @property
    def effective_loglikelihood_mode(self) -> str:
        """Resolve compatibility ``auto`` to the fast first-token path."""
        return (
            "continuation"
            if self.config.loglikelihood_mode == "continuation"
            else "first_token"
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> list[dict[str, Any]]:
        """Load the dataset and apply resume filtering.

        Mirrors InferenceRunner.load_data: loads raw items, checks previously
        completed samples, and returns only the items still to process.

        Returns:
            List of items remaining to process (empty when all done).

        Raises:
            FileNotFoundError: If the input file does not exist
            json.JSONDecodeError: If the input file contains invalid JSON
        """
        raw_data = load_jsonl(self.config.input_file)
        logger.info(f"Loaded {len(raw_data)} items from input file")

        resume_state = load_resume_state(
            self.config.output_file,
            self.config.input_key,
            self.config.response_key,
            repair_truncated_last_line=self.config.repair_resume,
        )
        target_samples = self.config.n_samples if self.config.mode == "generate" else 1
        remaining = prepare_sample_requests(
            raw_data,
            resume_state,
            self.config.input_key,
            target_samples,
            base_seed=self.config.seed,
            prompt_resolver=self.build_prompt,
        )

        if resume_state.completed_count:
            logger.info(
                "Found %d completed samples.",
                resume_state.completed_count,
            )

        logger.info("Total remaining samples to process: %d", len(remaining))
        return remaining

    def build_prompt(self, item: dict[str, Any]) -> str:
        """Assemble the full prompt (few-shot prefix + raw prompt).

        Used both for resume filtering and inference so the two always agree,
        even when n_shot > 0 changes the prompt that gets written to output.
        """
        prompt = item.get(self.config.input_key) or item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Prompt under {self.config.input_key!r} or 'prompt' must be a "
                "non-empty string"
            )
        fs_prefix = (
            self._few_shot_fmt.get_prefix(prompt, str(item.get("doc_id", "")))
            if self._few_shot_fmt
            else ""
        )
        return fs_prefix + prompt

    # ------------------------------------------------------------------
    # Concurrent processing
    # ------------------------------------------------------------------

    def _process_concurrently(
        self, remaining: list[dict[str, Any]], worker: Any
    ) -> None:
        """Process items concurrently using a thread pool with progress tracking.

        Mirrors InferenceRunner._process_concurrently: writes successful
        results, updates stats, and saves failed items for debugging.

        Args:
            remaining: Data items to process
            worker: Per-item callable (process_loglikelihood_item or a
                process_generate_item binding); returns a result dict,
                None for a skipped item, or raises on failure
        """
        failed_items: list[dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="mc_worker"
        ) as executor:
            futures = {executor.submit(worker, item): item for item in remaining}

            with tqdm(
                total=len(remaining), desc="Processing samples", unit="sample"
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    item = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.warning(f"Item failed: {e}")
                        with self._stats_lock:
                            self._stats["failed"] += 1
                        failed_items.append({"item": item, "error": str(e)})
                    else:
                        if result is None:
                            with self._stats_lock:
                                self._stats["skipped"] += 1
                        else:
                            self._write_result(result)
                            with self._stats_lock:
                                if result.get("error") == CONTEXT_LENGTH_ERROR:
                                    self._stats["failed"] += 1
                                else:
                                    self._stats["processed"] += 1
                                    if result.get("correct"):
                                        self._stats["correct"] += 1
                    pbar.update(1)

        if failed_items:
            logger.warning(f"Total failed tasks: {len(failed_items)}")
            save_failed_items(self.config.output_file, failed_items)

    def _write_result(self, result: dict[str, Any]) -> None:
        """Append one result under the runner's write lock."""
        append_jsonl(self.config.output_file, [result], self._file_lock)

    # ------------------------------------------------------------------
    # Loglikelihood mode
    # ------------------------------------------------------------------

    def run_loglikelihood(self, remaining: list[dict[str, Any]]) -> None:
        """Run the loglikelihood pipeline over the remaining items.

        Args:
            remaining: Items left after resume filtering (see load_data)
        """
        logger.info(
            "effective_loglikelihood_mode=%s",
            self.effective_loglikelihood_mode,
        )
        logger.info("⏳ Processing %d loglikelihood items", len(remaining))
        self._process_concurrently(remaining, self.process_loglikelihood_item)
        self.log_stats()

    def process_loglikelihood_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single MC item via loglikelihood comparison.

        Args:
            item: MC item with prompt, choices, and gold index

        Returns:
            Result dict with choices, per-choice logprobs, prediction, and
            correctness; None when the item has no choices (counted as skipped).
            A context-length rejection returns a permanent-failure row
            (empty logprobs, "error" marker) that is excluded as an inference
            failure and treated as completed by resume.

        Raises:
            RuntimeError: When every choice scored -inf, i.e. the batched
                request failed. Such an item must NOT be scored — argmax
                would silently degrade to always picking choice 0. Raising
                lets the driver count it failed, dump it to *_failed.jsonl,
                and retry it on the next resume run.
        """
        prompt = self.build_prompt(item)
        choices = item.get("choices", [])
        gold = item.get("gold", -1)

        if not choices:
            return None

        if self.client is None:
            raise RuntimeError("Loglikelihood client is not initialized")
        choice_tokens = self._choice_tokens(item, len(choices))
        scoring_mode = self.effective_loglikelihood_mode
        choice_logprobs: list[list[float]] = []
        choice_scored_tokens: list[list[str]] = []
        choice_token_ids: list[list[int] | None] | None = None

        def context_length_result() -> dict[str, Any]:
            return {
                self.config.input_key: prompt,
                "doc_id": item["doc_id"],
                "choices": choices,
                "choice_tokens": choice_tokens,
                "gold": gold,
                "logprobs": [],
                "pred": -1,
                "correct": False,
                "error": CONTEXT_LENGTH_ERROR,
            }

        if scoring_mode == "continuation":
            continuation_result = self.client.score_continuations(
                LoglikelihoodRequest(prompt, tuple(choice_tokens))
            )
            if not continuation_result.exact:
                reason = continuation_result.error or "unknown reason"
                if reason == CONTEXT_LENGTH_ERROR:
                    return context_length_result()
                raise RuntimeError(f"Continuation logprob request failed: {reason}")
            choice_logprobs = [
                list(choice.token_logprobs) for choice in continuation_result.choices
            ]
            choice_scored_tokens = [
                list(choice.token_texts) for choice in continuation_result.choices
            ]
            candidate_token_ids = [
                list(choice.token_ids) if choice.token_ids is not None else None
                for choice in continuation_result.choices
            ]
            if any(token_ids is not None for token_ids in candidate_token_ids):
                choice_token_ids = candidate_token_ids

        if scoring_mode == "first_token":
            logprobs = self.client.get_choices_logprobs(prompt, choice_tokens)
            if logprobs is None:
                return context_length_result()
            choice_logprobs = [
                [score] if score != float("-inf") else [] for score in logprobs
            ]
            choice_scored_tokens = [[] for _ in choice_tokens]
        elif scoring_mode == "continuation":
            logprobs = [
                sum(scores) if scores else float("-inf") for scores in choice_logprobs
            ]
        if all(lp == float("-inf") for lp in logprobs):
            raise RuntimeError("Logprob request failed for all choices")

        pred = max(range(len(logprobs)), key=logprobs.__getitem__) if logprobs else -1
        is_correct = pred == gold
        return {
            self.config.input_key: prompt,
            "doc_id": item["doc_id"],
            "choices": choices,
            "choice_tokens": choice_tokens,
            "gold": gold,
            # JSON has no representation for infinity. Missing top-logprob
            # candidates are persisted as null and restored to -inf by the scorer.
            "logprobs": [
                None if score == float("-inf") else score for score in logprobs
            ],
            "choice_logprobs": choice_logprobs,
            "scoring_mode": scoring_mode,
            "loglikelihood_exact": scoring_mode == "continuation",
            "scoring_approximation": (
                None if scoring_mode == "continuation" else "first_token_top_logprobs"
            ),
            "choice_scored_tokens": choice_scored_tokens,
            "choice_token_ids": choice_token_ids,
            "choice_token_count": (
                [len(scores) for scores in choice_logprobs]
                if scoring_mode == "continuation"
                else None
            ),
            # Harness normalizes MC likelihood by Unicode character count and
            # UTF-8 byte count. Token count is retained only as diagnostics.
            "choice_char_count": [len(token) for token in choice_tokens],
            "choice_byte_count": [
                len(token.encode("utf-8")) for token in choice_tokens
            ],
            "pred": pred,
            "correct": is_correct,
        }

    @staticmethod
    def _choice_tokens(item: dict[str, Any], num_choices: int) -> list[str]:
        """Resolve answer tokens used by first-token loglikelihood scoring."""
        explicit = item.get("choice_tokens")
        if isinstance(explicit, list) and len(explicit) == num_choices:
            return [str(token) for token in explicit]

        choices = item.get("choices", [])
        if isinstance(choices, list) and len(choices) == num_choices:
            as_strings = [str(choice).strip() for choice in choices]
            if all(
                len(choice) == 1 and "A" <= choice.upper() <= "J"
                for choice in as_strings
            ):
                return [choice.upper() for choice in as_strings]

        if num_choices > 10:
            raise ValueError(
                f"Too many choices for letter-token scoring: {num_choices}"
            )
        return [chr(ord("A") + i) for i in range(num_choices)]

    # ------------------------------------------------------------------
    # Generate mode
    # ------------------------------------------------------------------

    def run_generate(self, remaining: list[dict[str, Any]]) -> None:
        """Run the generate pipeline over the remaining items.

        Args:
            remaining: Items left after resume filtering (see load_data)
        """
        logger.info(
            "⏳ Processing %d generation request(s)",
            len(remaining),
        )
        gen_client: openai.OpenAI = openai.OpenAI(
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.request_timeout),
            organization=self.config.organization,
            max_retries=0,  # retries are handled by call_with_retry
        )

        base_messages: list[dict[str, str]] = []
        if self.system_prompt:
            base_messages.append({"role": "system", "content": self.system_prompt})

        self._process_concurrently(
            remaining,
            lambda item: self.process_generate_item(item, gen_client, base_messages),
        )
        self.log_stats()

    def process_generate_item(
        self,
        item: dict[str, Any],
        client: openai.OpenAI,
        base_messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Process a single MC item via text generation.

        Args:
            item: MC item with prompt and answer
            client: OpenAI client shared by all worker threads (thread-safe)
            base_messages: Pre-built system messages prepended to every request

        Returns:
            Result dict with the generated text as a string under the
            configured response key. The 'correct' key is intentionally
            absent — generate mode extracts answers at scoring time; only
            loglikelihood mode computes correctness inline. A context-length
            rejection returns a permanent-failure row (empty gen plus an
            "error" marker) that is excluded as an inference failure and
            treated as completed by resume.

        Raises:
            RuntimeError: When generation produced no usable text (retries
                exhausted, non-retryable 4xx, or null/empty content). An
                empty gen from such a transient failure must NOT be written:
                it would be scored as a wrong answer AND mark the prompt as
                completed so resume never retries it. Raising keeps it
                consistent with the loglikelihood mode's all-"-inf" guard
                (failed, dumped, retried on next run).
        """
        prompt = self.build_prompt(item)
        gold = item.get(self.config.label_key, "")
        messages = [*base_messages, *build_chat_messages(prompt, None)]

        call_args: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "max_completion_tokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "timeout": self.config.request_timeout,
            "seed": get_request_seed(item),
        }
        if self.config.extra_body_dict:
            call_args["extra_body"] = dict(self.config.extra_body_dict)
        # tool_choice: only send when explicitly configured
        if is_explicit_tool_choice(self.config.tool_choice):
            call_args["tool_choice"] = self.config.tool_choice

        def do_request() -> str:
            resp = client.chat.completions.create(**call_args)
            raw_choices = getattr(resp, "choices", []) or []
            if not isinstance(raw_choices, list | tuple):
                try:
                    raw_choices = [raw_choices[0]]
                except (IndexError, TypeError):
                    raw_choices = []
            if not raw_choices:
                raise MalformedResponseError("Generate returned no choices")
            content = getattr(getattr(raw_choices[0], "message", None), "content", None)
            if not content:
                raise RuntimeError("Generate returned empty content")
            return str(content)

        generation: str | None = None
        try:
            generation = call_with_retry(do_request, self.config.max_retries)
        except RuntimeError as e:
            raise RuntimeError(f"Generate produced no usable text: {e}") from e
        if generation is None:
            # Context-length rejection can never succeed on retry: persist a
            # permanent-failure row so resume treats the sample as completed.
            result = dict(item)
            result.pop("_request_seed", None)
            result[self.config.input_key] = prompt
            result[self.config.label_key] = gold
            result[self.config.response_key] = ""
            result["error"] = CONTEXT_LENGTH_ERROR
            return result
        if not generation:
            # Null/empty content from a completed request
            raise RuntimeError("Generate produced no usable text (empty response)")

        result = dict(item)
        result.pop("_request_seed", None)
        result[self.config.input_key] = prompt
        result[self.config.label_key] = gold
        result[self.config.response_key] = generation
        return result

    # ------------------------------------------------------------------
    # Stats and reporting
    # ------------------------------------------------------------------

    def log_stats(self) -> None:
        """Log runtime statistics (and accuracy for loglikelihood mode)."""
        logger.info(
            f"Stats: {self._stats['processed']} processed, "
            f"{self._stats['failed']} failed, "
            f"{self._stats['skipped']} skipped"
        )
        # Quick accuracy summary for loglikelihood mode
        if self.config.mode == "loglikelihood":
            self.print_loglikelihood_summary()

    def print_loglikelihood_summary(self) -> None:
        """Print quick accuracy from in-memory stats."""
        processed = self._stats["processed"]
        correct = self._stats.get("correct", 0)
        failed = self._stats["failed"]
        if processed:
            logger.info(
                f"Accuracy (loglikelihood): {correct}/{processed} = {correct / processed:.2%} "
                f"(failed={failed})"
            )

    def run(self) -> None:
        """Execute the complete MC inference pipeline with monitoring.

        Mirrors InferenceRunner.run: configuration validation, data loading,
        mode dispatch, and a final execution report. Unrecoverable errors are
        propagated to the CLI boundary, which logs them once.

        Raises:
            FileNotFoundError: If the input file is missing
            ValueError: If the configuration is invalid (incl. unknown mode)
            RuntimeError: For backend request or response failures
        """
        start_time = time.perf_counter()

        if not self.config.input_file or not Path(self.config.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        if not self.config.output_file:
            raise ValueError("Output file path is required")

        logger.info("🚀 Initializing MC inference pipeline")
        logger.info(
            "Configuration: %s",
            redact_config_for_logging(dataclasses.asdict(self.config)),
        )

        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        remaining = self.load_data()
        if not remaining:
            logger.info("✅ All items already processed")
            return

        if self.config.mode == "loglikelihood":
            self.run_loglikelihood(remaining)
        elif self.config.mode == "generate":
            self.run_generate(remaining)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        duration = time.perf_counter() - start_time
        total = (
            self._stats["processed"] + self._stats["failed"] + self._stats["skipped"]
        )
        success_rate = (self._stats["processed"] / max(total, 1)) * 100

        logger.info("\n=== Execution Summary ===")
        logger.info(f"Successfully processed: {self._stats['processed']}")
        logger.info(f"Failed: {self._stats['failed']}")
        logger.info(f"Skipped: {self._stats['skipped']}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Total duration: {duration:.2f} seconds")
        logger.info(f"Output file: {self.config.output_file}")
        logger.info("✅ MC inference pipeline completed successfully\n")


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    """Main entry point for the MC inference CLI.

    Mirrors main() in online.py: HfArgumentParser builds MCInferArguments
    directly from the dataclass (field names == CLI flags), then the runner
    executes with standardized exit-code handling.

    Raises:
        SystemExit: 130 on keyboard interrupt, 1 on any fatal error
    """
    start_time = time.perf_counter()
    try:
        # Parse command line arguments into a strongly typed dataclass
        parser = HfArgumentParser(MCInferArguments)  # type: ignore[arg-type]
        (config,) = parser.parse_args_into_dataclasses()

        # Log initialization with formatted argument display
        logger.info(
            "Initializing MCInferArguments with parsed command line arguments..."
        )
        logger.info("\n--- Parsed Arguments ---")
        logger.info(
            json.dumps(redact_config_for_logging(dataclasses.asdict(config)), indent=2)
        )

        # Initialize and run the inference process
        runner = MCRunner(config)
        runner.run()

        # Log successful completion with execution time
        total_time = time.perf_counter() - start_time
        logger.info(
            f"✅ MC inference completed successfully in {total_time:.2f} seconds"
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting gracefully...")
        sys.exit(130)  # Standard exit code for SIGINT
    except FileNotFoundError as e:
        logger.critical(f"File not found error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.critical(f"Invalid value error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"❌ An unrecoverable error occurred during execution: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
