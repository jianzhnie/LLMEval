"""Multiple-choice inference: answer-token loglikelihood and generation modes.

This module mirrors llmeval/inference/online.py in structure, naming, and
documentation style so the two can be reviewed side by side with a file diff:
- MCLoglikelihoodClient  ~  InferenceClient
- MCRunner               ~  InferenceRunner
- main() (HfArgumentParser) ~  main()

Shared utilities (ClientError, retry classification, backoff) live in
llmeval/utils/retry.py; data-loading / resume helpers live in
llmeval/inference/common.py; the configuration dataclass lives in
llmeval/utils/config.py (MCInferConfig).
MC-specific pieces (kept deliberately): FewShotFormatter, answer-token
logprobs via the completions API, and per-mode worker methods.
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
from pathlib import Path
from typing import Any

import httpx
import openai
from tqdm import tqdm
from transformers import HfArgumentParser

from llmeval.inference.common import (
    append_jsonl,
    expand_data_with_resume,
    get_request_seed,
    is_explicit_tool_choice,
    is_local_endpoint,
    load_jsonl,
    load_resume_state,
    redact_config_for_logging,
    save_failed_items,
)
from llmeval.inference.schema import (
    ChoiceLoglikelihood,
    LoglikelihoodRequest,
    LoglikelihoodResult,
)
from llmeval.utils.config import MCInferConfig
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY
from llmeval.utils.retry import ClientError, call_with_retry

logger = init_logger("mc_infer")


class ContinuationAlignmentError(ValueError):
    """A deterministic backend token-offset mismatch that should not be retried."""


class ContinuationLogprobs(list[list[float]]):
    """Aligned continuation scores plus optional backend token metadata."""

    def __init__(
        self,
        scores: list[list[float]],
        *,
        token_texts: list[list[str]] | None = None,
        token_ids: list[list[int] | None] | None = None,
        error: str | None = None,
    ) -> None:
        super().__init__(scores)
        self.token_texts = token_texts or [[] for _ in scores]
        self.token_ids = token_ids or [None for _ in scores]
        self.error = error


# ===========================================================================
# Few-shot formatter (MC-specific)
# ===========================================================================


class FewShotFormatter:
    """Load and format few-shot examples for MC prompts.

    Each few-shot example is formatted as:
        question\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: X\n\n
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
    """Client for computing answer-token log-probabilities via completions.

    Mirrors InferenceClient in online.py: same initialization, masked
    API-key logging, and classified retry policy. The MC-specific part is that
    one request carries ALL choices of an item (batched prompt list).
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 300,
        max_retries: int = 3,
        api_key: str = "",
        seed: int = 0,
        organization: str | None = None,
    ) -> None:
        """Initialize the client with API configuration.

        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint
            model_name: Served model name used in requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for transient failures
            api_key: API key; falls back to the OPENAI_API_KEY env var
        """
        self.model_name: str = model_name
        self.base_url: str = base_url
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.seed = seed
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

    def get_choices_logprobs(self, prompt: str, choice_texts: list[str]) -> list[float]:
        """Compute per-answer-token log-probabilities from first-token top_logprobs.

        Uses echo=False + max_tokens=1 + logprobs=20 to obtain the model's top
        predicted tokens after the prompt. For each target choice, we look up
        its logprob among the predictions. This directly measures
        P(target_token | prompt) without the token-alignment issues that arise
        from echo=True (where prompt tokenization can shift across continuations).

        The target tokens (typically "A"/"B"/"C"/"D") are looked up in several
        common tokenizer forms: with/without leading space, upper/lower case.

        Args:
            prompt: The shared MC prompt (few-shot prefix + question)
            choice_texts: Candidate answer tokens (e.g. ["A","B","C","D"])

        Returns:
            One logprob per choice, aligned with choice_texts. Choices not found
            among the top predictions get float("-inf"). All -inf when the
            request fails after all retries.
        """

        def do_request() -> list[float]:
            resp = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logprobs=20,
                echo=False,
                timeout=self.timeout,
                seed=getattr(self, "seed", 0),
            )
            top_dict: dict[str, float] = (
                resp.choices[0].logprobs.top_logprobs[0]
                if resp.choices
                and resp.choices[0].logprobs
                and resp.choices[0].logprobs.top_logprobs
                else {}
            )
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

        try:
            choice_logprobs = call_with_retry(do_request, self.max_retries)
        except Exception as e:
            logger.warning(f"Logprob request failed: {e}")
            choice_logprobs = None
        if choice_logprobs is None:
            # Request failed (4xx / exhausted retries / context length): the
            # all-"-inf" result marks the item failed downstream, never scored.
            return [float("-inf")] * len(choice_texts)
        return choice_logprobs

    def score_continuations(self, request: LoglikelihoodRequest) -> LoglikelihoodResult:
        """Score complete continuations and return a validated typed result."""

        def do_request() -> LoglikelihoodResult:
            prompts = [
                f"{request.context}{continuation}"
                for continuation in request.continuations
            ]
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompts,
                max_tokens=1,
                temperature=0,
                logprobs=20,
                echo=True,
                timeout=self.timeout,
                seed=getattr(self, "seed", 0),
            )
            completions = getattr(response, "choices", []) or []
            if len(completions) != len(request.continuations):
                raise ValueError("completion count does not match continuation count")

            ordered_completions: list[Any | None] = [None] * len(request.continuations)
            for fallback_index, completion in enumerate(completions):
                response_index = getattr(completion, "index", fallback_index)
                index = (
                    response_index
                    if isinstance(response_index, int)
                    else fallback_index
                )
                if (
                    index < 0
                    or index >= len(request.continuations)
                    or ordered_completions[index] is not None
                ):
                    raise ValueError("completion indices are invalid or duplicated")
                ordered_completions[index] = completion

            choice_results: list[ChoiceLoglikelihood] = []
            scoring_context_length = len(request.scoring_context)
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
                    raise ValueError("completion is missing token offsets or logprobs")
                if len(offsets) != len(token_logprobs) or any(
                    offset is not None and not isinstance(offset, int)
                    for offset in offsets
                ):
                    raise ValueError("completion token offsets are malformed")

                scored_text = request.scored_continuation(choice_index)
                choice_end = scoring_context_length + len(scored_text)
                selected_indices = [
                    index
                    for index, (offset, logprob) in enumerate(
                        zip(offsets, token_logprobs, strict=False)
                    )
                    if offset is not None
                    and scoring_context_length <= offset < choice_end
                    and logprob is not None
                ]
                if (
                    not selected_indices
                    or offsets[selected_indices[0]] != scoring_context_length
                ):
                    raise ContinuationAlignmentError(
                        "continuation does not start on a token boundary"
                    )

                tokens = getattr(logprobs, "tokens", None)
                if not isinstance(tokens, list | tuple) or len(tokens) != len(
                    token_logprobs
                ):
                    raise ValueError("completion is missing aligned token text")
                selected_tokens = tuple(
                    str(tokens[index]) for index in selected_indices
                )
                expected_offset = scoring_context_length
                for selected_index, token in zip(
                    selected_indices, selected_tokens, strict=True
                ):
                    if offsets[selected_index] != expected_offset:
                        raise ContinuationAlignmentError(
                            "continuation token offsets do not match token text"
                        )
                    expected_offset += len(token)
                if expected_offset != choice_end:
                    raise ContinuationAlignmentError(
                        "continuation token offsets do not cover the choice"
                    )

                backend_ids = getattr(logprobs, "token_ids", None)
                if backend_ids is not None and (
                    not isinstance(backend_ids, list | tuple)
                    or len(backend_ids) != len(token_logprobs)
                ):
                    raise ValueError("completion token IDs are malformed")
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
        except (ContinuationAlignmentError, ValueError) as exc:
            logger.debug("Continuation scoring fallback: %s", exc)
            return LoglikelihoodResult.failure(request, str(exc))
        except ClientError as exc:
            logger.warning("Continuation logprob request failed: %s", exc)
            return LoglikelihoodResult.failure(request, str(exc))
        except Exception as exc:
            logger.warning("Continuation logprob request failed: %s", exc)
            return LoglikelihoodResult.failure(request, str(exc))
        if result is None:
            return LoglikelihoodResult.failure(request, "context_length_exceeded")
        return result

    def get_choices_continuation_logprobs(
        self, prompt: str, choice_texts: list[str]
    ) -> ContinuationLogprobs:
        """Compatibility wrapper returning aligned token score lists."""
        request = LoglikelihoodRequest(prompt, tuple(choice_texts))
        result = self.score_continuations(request)
        return ContinuationLogprobs(
            [list(choice.token_logprobs) for choice in result.choices],
            token_texts=[list(choice.token_texts) for choice in result.choices],
            token_ids=[
                list(choice.token_ids) if choice.token_ids is not None else None
                for choice in result.choices
            ],
            error=result.error,
        )


# ===========================================================================
# Runner
# ===========================================================================


class MCRunner:
    """Orchestrates MC inference with resume, threading, and stats.

    Mirrors InferenceRunner in online.py: same pipeline stages
    (load → resume filter → concurrent processing → report), the same
    thread-safety primitives, and the same failed-item persistence.
    """

    def __init__(self, config: MCInferConfig) -> None:
        """Initialize the runner with client, prompts, and thread safety setup.

        Args:
            config: MC inference configuration (see MCInferConfig)

        Raises:
            RuntimeError: If the loglikelihood client fails to initialize
        """
        self.config: MCInferConfig = config

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
                )
            except (OSError, ValueError) as e:
                raise RuntimeError(f"Failed to initialize MC client: {e}") from e

        # Set up system prompt with validation (generate mode)
        self.system_prompt: str | None = None
        if config.system_prompt_type and config.system_prompt_type != "empty":
            self.system_prompt = SYSTEM_PROMPT_FACTORY.get(config.system_prompt_type)
            if not self.system_prompt:
                logger.warning(
                    f"Unknown system_prompt_type: {config.system_prompt_type}"
                )

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
            repair_truncated_last_line=getattr(self.config, "repair_resume", False),
        )
        target_samples = self.config.n_samples if self.config.mode == "generate" else 1
        remaining = expand_data_with_resume(
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
        raw_prompt = item.get(self.config.input_key) or item.get("prompt") or ""
        prompt = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt)
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
        logger.info(
            f"⏳ Processing {len(remaining)} items "
            f"({len(remaining)} batched loglikelihood requests)"
        )
        self._process_concurrently(remaining, self.process_loglikelihood_item)
        self.log_stats()

    def process_loglikelihood_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single MC item via loglikelihood comparison.

        Args:
            item: MC item with prompt, choices, and gold index

        Returns:
            Result dict with choices, per-choice logprobs, prediction, and
            correctness; None when the item has no choices (counted as skipped)

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
        if scoring_mode == "continuation":
            candidate_scores: list[list[float]]
            candidate_scores = self.client.get_choices_continuation_logprobs(
                prompt, choice_tokens
            )
            if (
                isinstance(candidate_scores, list)
                and len(candidate_scores) == len(choice_tokens)
                and all(
                    isinstance(scores, list) and scores for scores in candidate_scores
                )
            ):
                choice_logprobs = candidate_scores
                choice_scored_tokens = getattr(
                    candidate_scores,
                    "token_texts",
                    [[] for _ in candidate_scores],
                )
                candidate_token_ids = getattr(candidate_scores, "token_ids", None)
                if isinstance(candidate_token_ids, list) and any(
                    token_ids is not None for token_ids in candidate_token_ids
                ):
                    choice_token_ids = candidate_token_ids
                scoring_mode = "continuation"
            else:
                raise RuntimeError("Continuation logprob request failed")

        if scoring_mode == "first_token":
            logprobs = self.client.get_choices_logprobs(prompt, choice_tokens)
            choice_logprobs = [
                [score] if score != float("-inf") else [] for score in logprobs
            ]
            choice_scored_tokens = [[] for _ in choice_tokens]
        else:
            logprobs = [
                sum(scores) if scores else float("-inf") for scores in choice_logprobs
            ]
        if all(lp == float("-inf") for lp in logprobs):
            raise RuntimeError("Logprob request failed for all choices")

        pred = max(range(len(logprobs)), key=logprobs.__getitem__) if logprobs else -1
        is_correct = pred == gold
        return {
            self.config.input_key: prompt,
            **({"doc_id": item["doc_id"]} if "doc_id" in item else {}),
            "choices": choices,
            "choice_tokens": choice_tokens,
            "gold": gold,
            "logprobs": logprobs,
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
        total_samples = len(remaining)
        logger.info(
            "⏳ Processing %d item(s), %d generation sample(s)",
            len(remaining),
            total_samples,
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
            Result dict with the generated text in the gen list. The 'correct'
            key is intentionally absent — generate mode extracts answers at
            scoring time; only loglikelihood mode computes correctness inline.

        Raises:
            RuntimeError: When generation produced no usable text (retries
                exhausted, non-retryable 4xx, or null/empty content). An empty
                gen must NOT be written: it would be scored as a wrong answer
                AND mark the prompt as completed so resume never retries it.
                Raising keeps it consistent with the loglikelihood mode's
                all-"-inf" guard (failed, dumped, retried on next run).
        """
        prompt = self.build_prompt(item)
        gold = item.get(self.config.label_key, "")
        messages = [*base_messages, {"role": "user", "content": prompt}]

        call_args: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.request_timeout,
            "seed": get_request_seed(item),
        }
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
                raise RuntimeError("Generate returned no choices")
            content = getattr(getattr(raw_choices[0], "message", None), "content", None)
            if not content:
                raise RuntimeError("Generate returned empty content")
            return str(content)

        generation: str | None = None
        try:
            generation = call_with_retry(do_request, self.config.max_retries)
        except (ClientError, RuntimeError) as e:
            raise RuntimeError(f"Generate produced no usable text: {e}") from e
        if not generation:
            # Context-length rejection (None) or null/empty content ("")
            raise RuntimeError("Generate produced no usable text (empty response)")

        result = {
            self.config.input_key: prompt,
            self.config.label_key: gold,
            self.config.response_key: [generation],
            **({"doc_id": item["doc_id"]} if "doc_id" in item else {}),
            "expected_samples": self.config.n_samples,
        }
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

    Mirrors main() in online.py: HfArgumentParser builds MCInferConfig
    directly from the dataclass (field names == CLI flags), then the runner
    executes with standardized exit-code handling.

    Raises:
        SystemExit: 130 on keyboard interrupt, 1 on any fatal error
    """
    start_time = time.perf_counter()
    try:
        # Parse command line arguments into a strongly typed dataclass
        parser = HfArgumentParser(MCInferConfig)  # type: ignore[arg-type]
        (config,) = parser.parse_args_into_dataclasses()

        # Log initialization with formatted argument display
        logger.info("Initializing MCInferConfig with parsed command line arguments...")
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
