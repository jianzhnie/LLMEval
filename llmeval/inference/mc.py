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
    count_completed_samples_by_identity,
    is_explicit_tool_choice,
    load_jsonl,
    require_document_id,
    save_failed_items,
    validate_document_ids,
)
from llmeval.utils.config import MCInferConfig
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY
from llmeval.utils.retry import ClientError, call_with_retry

logger = init_logger("mc_infer")


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

    def load(self, input_file: str) -> None:
        """Load few-shot examples from dev file.  Call once before get_prefix()."""
        if self.n_shot <= 0:
            return

        source = self.few_shot_file or input_file
        try:
            items = load_jsonl(source)
        except Exception:
            logger.warning(f"Failed to load few-shot from {source}")
            return

        if len(items) < self.n_shot:
            logger.warning(f"Only {len(items)} examples available, need {self.n_shot}")
            return

        rng = random.Random(self.seed)
        # Sample n_shot+1: extra for dedup (lm-eval style)
        selected = rng.sample(items, min(self.n_shot + 1, len(items)))
        self._few_shot_pool = selected
        self._all_formatted = [self._format_demo(it) for it in selected]
        logger.info(f"Loaded {self.n_shot} few-shot examples (seed={self.seed})")

    def get_prefix(self, test_prompt: str) -> str:
        """Get few-shot prefix, excluding any demo matching test_prompt (lm-eval dedup)."""
        if self.n_shot <= 0:
            return ""

        # Filter out the test doc if it appears in few-shot pool
        demos = self._all_formatted
        if test_prompt:
            demos = [
                d
                for i, d in enumerate(demos)
                if self._few_shot_pool[i].get("prompt", "") != test_prompt
            ]
        demos = demos[: self.n_shot]  # trim to n_shot

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
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        # Warn if using default EMPTY key
        if self.api_key == "EMPTY":
            logger.warning("Using default 'EMPTY' API key. This may not be secure.")

        # Initialize OpenAI client with validated configuration
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        masked_key = f"{self.api_key[:4]}***" if len(self.api_key) > 4 else "***"
        logger.info(
            f"Using API Key: {masked_key}, Timeout: {self.timeout}, "
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
        except ClientError as e:
            logger.warning(f"Logprob request failed: {e}")
            choice_logprobs = None
        if choice_logprobs is None:
            # Request failed (4xx / exhausted retries / context length): the
            # all-"-inf" result marks the item failed downstream, never scored.
            return [float("-inf")] * len(choice_texts)
        return choice_logprobs

    def get_choices_continuation_logprobs(
        self, prompt: str, choice_texts: list[str]
    ) -> list[list[float]]:
        """Score complete choice continuations using an echo completion.

        OpenAI-compatible completion APIs that support ``echo=True`` return
        token-level log probabilities for the supplied prompt.  By sending
        ``prompt + choice`` and selecting offsets inside the choice span, the
        scorer obtains the complete continuation likelihood instead of only
        looking at the first token's top-k entries.

        Returns an empty list for a choice whose token-level scores cannot be
        recovered.  A request failure returns one empty list per choice so the
        caller can mark the item as failed rather than selecting index zero.
        """

        def do_request() -> list[list[float]]:
            prompts = [f"{prompt}{choice}" for choice in choice_texts]
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompts,
                max_tokens=1,
                temperature=0,
                logprobs=20,
                echo=True,
                timeout=self.timeout,
            )
            completions = getattr(response, "choices", []) or []
            if len(completions) != len(choice_texts):
                return [[] for _ in choice_texts]

            scores: list[list[float]] = []
            prompt_length = len(prompt)
            for completion, choice_text in zip(completions, choice_texts, strict=True):
                logprobs = getattr(completion, "logprobs", None)
                offsets = getattr(logprobs, "text_offset", None) if logprobs else None
                token_logprobs = (
                    getattr(logprobs, "token_logprobs", None) if logprobs else None
                )
                if not offsets or not token_logprobs:
                    scores.append([])
                    continue

                choice_end = prompt_length + len(choice_text)
                selected = [
                    float(logprob)
                    for offset, logprob in zip(offsets, token_logprobs, strict=False)
                    if offset is not None
                    and prompt_length <= offset < choice_end
                    and logprob is not None
                ]
                scores.append(selected)
            return scores

        try:
            scores = call_with_retry(do_request, self.max_retries)
            return scores if scores is not None else [[] for _ in choice_texts]
        except ClientError as exc:
            logger.warning("Continuation logprob request failed: %s", exc)
            return [[] for _ in choice_texts]


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
            self._few_shot_fmt = FewShotFormatter(config.n_shot, config.few_shot_file)
            self._few_shot_fmt.load(config.input_file)

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

    def get_completed_identity_counts(self) -> dict[tuple[str, str], int]:
        """Return completed counts keyed by prepared ID and rendered prompt."""
        return count_completed_samples_by_identity(
            self.config.output_file,
            self.config.input_key,
            self.config.response_key,
        )

    def get_completed_document_ids(self) -> set[str]:
        """Return stable document IDs already written to the output file."""
        return {document_id for document_id, _ in self.get_completed_identity_counts()}

    def get_completed_prompts(self) -> set[str]:
        """Get completed legacy prompts from existing output (for resume).

        Scans the output file and collects every written (few-shot prefixed)
        prompt, enabling resume for interrupted runs. Malformed lines are
        skipped with a warning instead of aborting the run.

        Returns:
            Set of prompt strings already present in the output file.
        """
        output_path = Path(self.config.output_file)
        if not output_path.exists() or output_path.stat().st_size == 0:
            return set()
        completed = set()
        try:
            with open(output_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            logger.warning(
                                "Skipping non-object JSON on output line %d: %s",
                                line_num,
                                type(item).__name__,
                            )
                            continue
                        if item.get("doc_id"):
                            continue
                        prompt = item.get(self.config.input_key, "")
                        if prompt:
                            completed.add(prompt)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
        except Exception as e:
            logger.error(f"Error reading output file for resume check: {e}")
        return completed

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
        validate_document_ids(raw_data)
        logger.info(f"Loaded {len(raw_data)} items from input file")

        completed_counts = self.get_completed_identity_counts()
        completed_ids = {document_id for document_id, _ in completed_counts}
        completed_prompts = self.get_completed_prompts()

        remaining: list[dict[str, Any]] = []
        for index, raw_item in enumerate(raw_data):
            item = raw_item.copy()
            document_id = require_document_id(item, index)
            target_samples = (
                self.config.n_samples if self.config.mode == "generate" else 1
            )
            rendered_prompt = self.build_prompt(item)
            completed = completed_counts.get((document_id, rendered_prompt), 0)
            if completed == 0 and rendered_prompt in completed_prompts:
                # Legacy MC output did not record per-sample counts. Treat the
                # item as complete, matching the historical resume behavior.
                completed = target_samples
            is_completed = completed >= target_samples
            if not is_completed:
                if self.config.mode == "generate":
                    item["_llmeval_remaining_samples"] = target_samples - completed
                    item["_llmeval_sample_start"] = completed
                else:
                    item["_llmeval_sample_index"] = completed
                remaining.append(item)

        if completed_ids or completed_prompts:
            logger.info(
                "Found %d completed items.",
                len(completed_ids) + len(completed_prompts),
            )

        total_remaining_samples = sum(
            int(item.get("_llmeval_remaining_samples", 1)) for item in remaining
        )
        logger.info("Total remaining items to process: %d", len(remaining))
        logger.info("Total remaining samples to process: %d", total_remaining_samples)
        return remaining

    def build_prompt(self, item: dict[str, Any]) -> str:
        """Assemble the full prompt (few-shot prefix + raw prompt).

        Used both for resume filtering and inference so the two always agree,
        even when n_shot > 0 changes the prompt that gets written to output.
        """
        fs_prefix = (
            self._few_shot_fmt.get_prefix(item.get(self.config.input_key, ""))
            if self._few_shot_fmt
            else ""
        )
        return fs_prefix + item.get(self.config.input_key, "")

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
                            generated = result.get(self.config.response_key)
                            processed_count = (
                                len(generated)
                                if self.config.mode == "generate"
                                and isinstance(generated, list)
                                else 1
                            )
                            with self._stats_lock:
                                self._stats["processed"] += processed_count
                                if result.get("correct"):
                                    self._stats["correct"] += 1
                    pbar.update(1)

        if failed_items:
            logger.warning(f"Total failed tasks: {len(failed_items)}")
            save_failed_items(self.config.output_file, failed_items)

    def _write_result(self, result: dict[str, Any]) -> None:
        """Write result to output file in a thread-safe manner.

        Args:
            result: The result dictionary to write
        """
        with self._file_lock:
            try:
                output_path = Path(self.config.output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # Ensure data is immediately written
            except Exception as e:
                logger.error(f"Error writing batch results: {e}")
                raise OSError(f"Failed to write batch results: {e}") from e

    # ------------------------------------------------------------------
    # Loglikelihood mode
    # ------------------------------------------------------------------

    def run_loglikelihood(self, remaining: list[dict[str, Any]]) -> None:
        """Run the loglikelihood pipeline over the remaining items.

        Args:
            remaining: Items left after resume filtering (see load_data)
        """
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
        scoring_mode = self.config.loglikelihood_mode
        choice_logprobs: list[list[float]] = []
        if scoring_mode in ("auto", "continuation"):
            try:
                candidate_scores = self.client.get_choices_continuation_logprobs(
                    prompt, choice_tokens
                )
            except AttributeError:
                candidate_scores = []
            if (
                isinstance(candidate_scores, list)
                and len(candidate_scores) == len(choice_tokens)
                and all(
                    isinstance(scores, list) and scores for scores in candidate_scores
                )
            ):
                choice_logprobs = candidate_scores
                scoring_mode = "continuation"
            elif scoring_mode == "continuation":
                raise RuntimeError("Continuation logprob request failed")
            else:
                scoring_mode = "first_token"

        if scoring_mode == "first_token":
            logprobs = self.client.get_choices_logprobs(prompt, choice_tokens)
            choice_logprobs = [
                [score] if score != float("-inf") else [] for score in logprobs
            ]
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
            "_llmeval_sample_index": item.get("_llmeval_sample_index", 0),
            "choices": choices,
            "choice_tokens": choice_tokens,
            "gold": gold,
            "logprobs": logprobs,
            "choice_logprobs": choice_logprobs,
            "scoring_mode": scoring_mode,
            "choice_token_count": (
                [len(scores) for scores in choice_logprobs]
                if scoring_mode == "continuation"
                else None
            ),
            "choice_byte_count": (
                [len(token.encode("utf-8")) for token in choice_tokens]
                if scoring_mode == "continuation"
                else None
            ),
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
        total_samples = sum(
            int(item.get("_llmeval_remaining_samples", self.config.n_samples))
            for item in remaining
        )
        logger.info(
            "⏳ Processing %d item(s), %d generation sample(s)",
            len(remaining),
            total_samples,
        )
        gen_client: openai.OpenAI = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.request_timeout),
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
        }
        request_samples = int(
            item.get("_llmeval_remaining_samples", self.config.n_samples)
        )
        if request_samples > 1:
            call_args["n"] = request_samples
        # tool_choice: only send when explicitly configured
        if is_explicit_tool_choice(self.config.tool_choice):
            call_args["tool_choice"] = self.config.tool_choice

        def do_request() -> list[str]:
            resp = client.chat.completions.create(**call_args)
            # Reasoning models may return content=None (thinking exhausted
            # max_tokens); discard empty choices so failed samples are not
            # written as completed generations.
            raw_choices = getattr(resp, "choices", []) or []
            if not isinstance(raw_choices, (list, tuple)):
                try:
                    raw_choices = [raw_choices[0]]
                except (IndexError, TypeError):
                    raw_choices = []
            return [
                choice.message.content
                for choice in raw_choices
                if choice.message.content
            ]

        try:
            generations = call_with_retry(do_request, self.config.max_retries)
        except ClientError as e:
            raise RuntimeError(f"Generate produced no usable text: {e}") from e
        if not generations:
            # Context-length rejection (None) or null/empty content ("")
            raise RuntimeError("Generate produced no usable text (empty response)")

        sample_start = int(item.get("_llmeval_sample_start", 0))
        sample_indices = list(range(sample_start, sample_start + len(generations)))

        return {
            self.config.input_key: prompt,
            self.config.label_key: gold,
            self.config.response_key: generations,
            **({"doc_id": item["doc_id"]} if "doc_id" in item else {}),
            "_llmeval_sample_indices": sample_indices,
        }

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
        mode dispatch, and a final execution report. Any unrecoverable error
        is wrapped in RuntimeError after logging.

        Raises:
            FileNotFoundError: If the input file is missing
            ValueError: If the configuration is invalid (incl. unknown mode)
            RuntimeError: For unrecoverable execution errors
        """
        start_time = time.perf_counter()

        try:
            # Validate configuration
            if not self.config.input_file or not Path(self.config.input_file).exists():
                raise FileNotFoundError(
                    f"Input file not found: {self.config.input_file}"
                )
            if not self.config.output_file:
                raise ValueError("Output file path is required")

            # Initialize execution
            logger.info("🚀 Initializing MC inference pipeline")
            logger.info(f"Configuration: {dataclasses.asdict(self.config)}")

            # Set up output directory
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load and prepare data (resume filtering inside load_data)
            remaining = self.load_data()
            if not remaining:
                logger.info("✅ All items already processed")
                return

            # Execute pipeline (mode dispatch)
            if self.config.mode == "loglikelihood":
                self.run_loglikelihood(remaining)
            elif self.config.mode == "generate":
                self.run_generate(remaining)
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

            # Generate final report
            duration = time.perf_counter() - start_time
            total = (
                self._stats["processed"]
                + self._stats["failed"]
                + self._stats["skipped"]
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

        except Exception as e:
            logger.critical(
                f"❌ Fatal error: {e!s}", exc_info=True, extra={"stats": self._stats}
            )
            raise RuntimeError(f"Pipeline execution failed: {e!s}") from e


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
        logger.info(json.dumps(dataclasses.asdict(config), indent=2))

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
