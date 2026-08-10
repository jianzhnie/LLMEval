"""
Online inference server for OpenAI-compatible APIs.

This module provides a robust client for interacting with OpenAI-compatible APIs,
supporting concurrent requests, retry logic, and resume functionality for large-scale
inference tasks.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
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
    CONTEXT_LENGTH_ERROR,
    append_jsonl,
    build_chat_messages,
    ensure_raw_prompt,
    get_request_seed,
    is_explicit_tool_choice,
    is_local_endpoint,
    load_jsonl,
    load_resume_state,
    prepare_sample_requests,
    redact_config_for_logging,
    save_failed_items,
)
from llmeval.utils.config import OnlineInferArguments
from llmeval.utils.log import init_logger
from llmeval.utils.retry import MalformedResponseError, call_with_retry

logger = init_logger("online_vllm_server", logging.INFO)


class InferenceClient:
    """
    A robust client to interact with OpenAI-compatible APIs.

    This client provides retry logic, error handling, and support for various
    generation parameters including thinking mode for advanced language models.

    Attributes:
        api_key (str): OpenAI API key from environment variables
        client (openai.OpenAI): The OpenAI client instance
        timeout (int): Request timeout in seconds
        base_url (str): Base URL for the OpenAI-compatible API
    """

    def __init__(
        self,
        base_url: str,
        timeout: int,
        max_retries: int = 3,
        tool_choice: str = "none",
        api_key: str | None = None,
        seed: int = 0,
        organization: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the inference client with API configuration.

        Creates a new OpenAI client instance configured with the provided
        base URL and timeout settings. No range validation happens here —
        argument values are validated by the configuration dataclasses at
        parse time.

        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            tool_choice: Tool calling mode: none, auto, or required.
            api_key: API key; falls back to the OPENAI_API_KEY env var and then EMPTY.
            seed: Default request seed when no per-sample seed is provided.
            organization: Optional OpenAI organization ID.
            extra_body: Explicit non-standard fields for compatible providers.
        """
        self.base_url: str = base_url  # Store for potential reconnection
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.tool_choice: str = tool_choice
        self.seed = seed
        self.extra_body = dict(extra_body or {})
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        # Token usage counters, accumulated under _usage_lock in
        # _request_with_retry and reported by the runner's final summary.
        self._usage_lock: threading.Lock = threading.Lock()
        self.usage_stats: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        if self.api_key == "EMPTY":
            log = logger.debug if is_local_endpoint(base_url) else logger.warning
            log("Using default 'EMPTY' API key.")

        # Initialize OpenAI client with validated configuration.
        # max_retries=0: retries are handled by call_with_retry; letting the
        # SDK retry too would nest the two retry loops.
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(self.timeout),
            organization=organization,
            max_retries=0,
        )
        logger.info(
            f"Using API Key: ***, Timeout: {self.timeout}, Max Retries: {self.max_retries}, base_url: {self.base_url}"
        )

    def _prepare_messages(
        self, query: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        """Build OpenAI chat messages from a raw query."""
        ensure_raw_prompt(query)
        return build_chat_messages(query, system_prompt)

    def get_content(
        self,
        query: str,
        system_prompt: str | None,
        model_name: str,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        *,
        seed: int | None = None,
    ) -> str | None:
        """Generate one response.

        Returns the response text (null content is normalized to ""), or
        None for a context-length rejection — a deterministic failure the
        caller persists as a permanent-failure row so resume skips it.
        """
        call_args = self._build_call_args(
            query,
            system_prompt,
            model_name,
            max_completion_tokens,
            temperature,
            top_p,
            seed=seed,
        )
        completion = self._request_with_retry(call_args)
        if completion is None:
            return None  # context length exceeded (logged in retry.should_retry)
        # Reasoning models may return content=None (thinking exhausted
        # max_completion_tokens); normalize to "" so callers can treat it uniformly
        return completion.choices[0].message.content or ""

    def _build_call_args(
        self,
        query: str,
        system_prompt: str | None,
        model_name: str,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Validate inputs and assemble chat.completions call arguments.

        Args:
            query: User's input query (must be non-empty)
            system_prompt: Optional system prompt
            model_name: Served model name (must be non-empty)
            max_completion_tokens: Maximum completion tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Keyword arguments dict for client.chat.completions.create.

        Raises:
            ValueError: If query or model_name is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not model_name:
            raise ValueError("Model name cannot be empty")

        messages = self._prepare_messages(query, system_prompt)
        call_args: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": self.timeout,
            "seed": self.seed if seed is None else seed,
        }
        if self.extra_body:
            call_args["extra_body"] = dict(self.extra_body)
        # tool_choice: only send when explicitly configured (vLLM 0.23+ supports it)
        if is_explicit_tool_choice(self.tool_choice):
            call_args["tool_choice"] = self.tool_choice
        return call_args

    def _request_with_retry(self, call_args: dict[str, Any]) -> Any | None:
        """Execute a chat.completions call under the shared retry policy.

        The attempt loop and classification policy live in
        :func:`llmeval.utils.retry.call_with_retry` (same as mc.py); only the
        request itself — plus a structure probe so malformed responses are
        retried instead of crashing later at content extraction — is defined
        here.

        Args:
            call_args: Keyword arguments for client.chat.completions.create

        Returns:
            The raw completion object, or None on context-length rejection
            (callers map this to empty results).

        Raises:
            ClientError: For non-retryable API issues or exhausted retries
        """

        def do_request() -> Any:
            completion = self.client.chat.completions.create(**call_args)
            # Probe the structure so malformed responses are retried too
            try:
                _ = completion.choices[0].message
            except (AttributeError, IndexError, TypeError) as exc:
                raise MalformedResponseError(
                    f"Malformed response structure: {exc}"
                ) from exc
            self._record_usage(completion)
            return completion

        return call_with_retry(do_request, self.max_retries)

    def _record_usage(self, completion: Any) -> None:
        """Accumulate token usage from a successful chat completion.

        Best-effort: missing or malformed usage fields are ignored so usage
        accounting can never break inference.
        """
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        try:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        with self._usage_lock:
            self.usage_stats["prompt_tokens"] += prompt_tokens
            self.usage_stats["completion_tokens"] += completion_tokens

    def usage_snapshot(self) -> dict[str, int]:
        """Return a consistent copy of accumulated token usage."""
        with self._usage_lock:
            prompt_tokens = self.usage_stats["prompt_tokens"]
            completion_tokens = self.usage_stats["completion_tokens"]
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


class InferenceRunner:
    """Concurrent OpenAI-compatible inference runner with resume support."""

    def __init__(self, args: OnlineInferArguments) -> None:
        """Initialize the client, prompt, locks, and counters."""
        self.args: OnlineInferArguments = args

        # Initialize client with error handling
        try:
            self.client: InferenceClient = InferenceClient(
                base_url=args.base_url,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                tool_choice=args.tool_choice,
                api_key=args.api_key,
                seed=args.seed,
                organization=args.organization,
                extra_body=args.extra_body_dict,
            )
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to initialize inference client: {e}") from e

        # System prompt is resolved and validated by PromptArguments at parse time.
        self.system_prompt: str | None = args.system_prompt

        # Initialize thread safety and monitoring
        self._file_lock: threading.Lock = threading.Lock()
        self._stats: dict[str, int] = {"processed": 0, "failed": 0, "skipped": 0}
        self._stats_lock: threading.Lock = threading.Lock()  # Dedicated lock for stats

    def load_data(self) -> list[dict[str, Any]]:
        """Load input and expand only requests not completed by resume."""
        # Input file validation and loading
        if not os.path.exists(self.args.input_file):
            raise FileNotFoundError(f"Input file not found: {self.args.input_file}")

        # Load raw data
        raw_data: list[dict[str, Any]] = load_jsonl(self.args.input_file)
        logger.info(f"Loaded {len(raw_data)} items from input file")

        # Resume functionality handling
        resume_state = load_resume_state(
            self.args.output_file,
            self.args.input_key,
            self.args.response_key,
            repair_truncated_last_line=self.args.repair_resume,
        )

        if resume_state.completed_count > 0:
            logger.info(
                "Found %d completed samples from previous run.",
                resume_state.completed_count,
            )

        prepared_data = prepare_sample_requests(
            raw_data,
            resume_state,
            self.args.input_key,
            self.args.n_samples,
            base_seed=self.args.seed,
        )
        total_remaining = len(prepared_data)

        if not prepared_data:
            logger.warning("No data to process after preparation")

        logger.info(f"Total remaining samples to process: {total_remaining}")
        return prepared_data

    def _write_result(self, result: dict[str, Any]) -> None:
        """Append one result under the runner's write lock."""
        append_jsonl(self.args.output_file, [result], self._file_lock)

    def _extract_query(self, item: Any) -> str | None:
        """Return a usable query, updating stats for malformed input."""
        if not isinstance(item, dict):
            logger.error(f"Invalid item type: {type(item)}, expected dict")
            with self._stats_lock:
                self._stats["failed"] += 1
            return None

        query = item.get(self.args.input_key) or item.get("prompt")
        if not isinstance(query, str) or not query.strip():
            logger.warning("Query must be a non-empty string")
            with self._stats_lock:
                self._stats["skipped"] += 1
            return None
        return query

    def _build_result(
        self, item: dict[str, Any], response: str
    ) -> dict[str, Any] | None:
        """Build an output row or record an empty-response failure."""
        if not response.strip():
            logger.warning("Empty response received")
            with self._stats_lock:
                self._stats["failed"] += 1
            return None

        result = item.copy()
        result.pop("_request_seed", None)
        result[self.args.response_key] = response
        return result

    def process_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Generate, persist, and account for one expanded request."""
        query = self._extract_query(item)
        if not query:
            return None

        response = self.client.get_content(
            query=query,
            system_prompt=self.system_prompt,
            model_name=self.args.model_name,
            max_completion_tokens=self.args.max_completion_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            seed=get_request_seed(item),
        )

        if response is None:
            # Context-length rejection can never succeed on retry: persist a
            # permanent-failure row so resume treats the sample as completed.
            result = item.copy()
            result.pop("_request_seed", None)
            result[self.args.response_key] = ""
            result["error"] = CONTEXT_LENGTH_ERROR
            self._write_result(result)
            with self._stats_lock:
                self._stats["failed"] += 1
            return result

        result = self._build_result(item, response)
        if not result:
            return None

        self._write_result(result)
        with self._stats_lock:
            self._stats["processed"] += 1

        return result

    def _process_concurrently(self, expanded_data: list[dict[str, Any]]) -> None:
        """Process one independent request per expanded sample."""
        failed_tasks: list[dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.max_workers, thread_name_prefix="inference_worker"
        ) as executor:
            futures = {
                executor.submit(self.process_item, item): item for item in expanded_data
            }

            with tqdm(
                total=len(expanded_data), desc="Processing samples", unit="sample"
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    item = futures[future]
                    try:
                        result = future.result()
                        if result is None:
                            failed_tasks.append(
                                {
                                    "doc_id": item.get("doc_id"),
                                    "error_category": "sample_processing",
                                    "error": "sample was skipped or returned an empty response",
                                }
                            )
                    except Exception as e:
                        logger.warning("Inference sample failed: %s", e)
                        with self._stats_lock:
                            self._stats["failed"] += 1
                        prompt_val = item.get(self.args.input_key) or item.get("prompt")
                        failed_tasks.append(
                            {
                                "doc_id": item.get("doc_id"),
                                self.args.input_key: (
                                    str(prompt_val)[:200]
                                    if prompt_val is not None
                                    else None
                                ),
                                "error": str(e),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                    finally:
                        pbar.update(1)

        if failed_tasks:
            logger.warning(f"Total failed tasks: {len(failed_tasks)}")
            save_failed_items(self.args.output_file, failed_tasks)

    def run(self) -> None:
        """Load, resume, execute, and report online inference."""
        start_time = time.perf_counter()

        if not self.args.output_file:
            raise ValueError("Output file path is required")

        logger.info("🚀 Initializing inference pipeline")
        logger.info(
            "Configuration: %s",
            redact_config_for_logging(dataclasses.asdict(self.args)),
        )

        eval_dataset: list[dict[str, Any]] = self.load_data()
        if not eval_dataset:
            logger.info("✅ All samples already processed")
            return

        output_path = Path(self.args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_samples = len(eval_dataset)
        logger.info(f"⏳ Processing {total_samples} samples")
        self._process_concurrently(eval_dataset)

        duration = time.perf_counter() - start_time
        success_rate = (self._stats["processed"] / total_samples) * 100

        logger.info("\n=== Execution Summary ===")
        logger.info(f"Samples to process this run: {total_samples}")
        logger.info(f"Successfully processed: {self._stats['processed']}")
        logger.info(f"Failed: {self._stats['failed']}")
        logger.info(f"Skipped: {self._stats['skipped']}")
        usage = self.client.usage_snapshot()
        logger.info(f"Prompt tokens: {usage['prompt_tokens']}")
        logger.info(f"Completion tokens: {usage['completion_tokens']}")
        logger.info(f"Total tokens: {usage['total_tokens']}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Total duration: {duration:.2f} seconds")
        logger.info(f"Output file: {self.args.output_file}")
        logger.info("✅ Inference pipeline completed successfully\n")


def main() -> None:
    """Parse online inference arguments and run the CLI."""
    start_time = time.perf_counter()
    try:
        # Parse command line arguments into a strongly typed dataclass
        parser = HfArgumentParser(OnlineInferArguments)  # type: ignore[arg-type]
        (eval_args,) = parser.parse_args_into_dataclasses()

        # Log initialization with formatted argument display
        logger.info(
            "Initializing OnlineInferArguments with parsed command line arguments..."
        )
        logger.info("\n--- Parsed Arguments ---")
        logger.info(
            json.dumps(
                redact_config_for_logging(dataclasses.asdict(eval_args)), indent=2
            )
        )

        # Initialize and run the inference process
        runner = InferenceRunner(eval_args)
        runner.run()

        # Log successful completion with execution time
        total_time = time.perf_counter() - start_time
        logger.info(f"✅ Inference completed successfully in {total_time:.2f} seconds")

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
