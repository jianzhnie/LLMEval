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
import time
from pathlib import Path
from typing import Any

import httpx
import openai
from tqdm import tqdm
from transformers import HfArgumentParser

from llmeval.inference.common import (
    append_jsonl,
    build_chat_messages,
    derive_request_seed,
    ensure_raw_prompt,
    is_local_endpoint,
    load_jsonl,
    load_resume_state,
    prepare_sample_requests,
    redact_config_for_logging,
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
            api_key: API key; falls back to the OPENAI_API_KEY env var and then EMPTY.
            seed: Default request seed when no per-sample seed is provided.
            organization: Optional OpenAI organization ID.
            extra_body: Explicit non-standard fields for compatible providers.
        """
        self.base_url: str = base_url  # Store for potential reconnection
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.seed = seed
        self.extra_body = dict(extra_body or {})
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

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

        A string response is returned verbatim, including an empty string.
        ``None`` is reserved for context-length rejection.
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
        return completion.choices[0].message.content

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
                content = completion.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                raise MalformedResponseError(
                    f"Malformed response structure: {exc}"
                ) from exc
            if content is not None and not isinstance(content, str):
                raise MalformedResponseError("response content is not a string")
            if content is None:
                raise MalformedResponseError("response content is missing")
            if content and not content.strip():
                # Whitespace-only content is a decoding glitch, not an empty
                # answer — retry it. A literal empty string is a legitimate
                # (empty) answer and is persisted as-is.
                raise MalformedResponseError("response content is whitespace-only")
            return completion

        return call_with_retry(do_request, self.max_retries)


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
                api_key=args.api_key,
                seed=args.seed,
                organization=args.organization,
                extra_body=args.extra_body_dict,
            )
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to initialize inference client: {e}") from e

        # System prompt is resolved and validated by PromptArguments at parse time.
        self.system_prompt: str | None = args.system_prompt

        self._stats: dict[str, int] = {"processed": 0, "failed": 0}

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
        )
        total_remaining = len(prepared_data)

        if not prepared_data:
            logger.warning("No data to process after preparation")

        logger.info(f"Total remaining samples to process: {total_remaining}")
        return prepared_data

    def _write_result(self, result: dict[str, Any]) -> None:
        """Append one result from the coordinator thread."""
        append_jsonl(self.args.output_file, [result])

    def _extract_query(self, item: Any) -> str:
        """Return a usable query or reject the sample."""
        if not isinstance(item, dict):
            raise ValueError(f"Invalid item type: {type(item)}, expected dict")

        query = item.get(self.args.input_key) or item.get("prompt")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        return query

    def _build_result(self, item: dict[str, Any], response: str) -> dict[str, Any]:
        """Build one successful output row."""
        result = item.copy()
        result.pop("error", None)
        result[self.args.response_key] = response
        return result

    def process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Generate one result without mutating shared runner state."""
        query = self._extract_query(item)

        sample_index = item.get("sample_index")
        response = self.client.get_content(
            query=query,
            system_prompt=self.system_prompt,
            model_name=self.args.model_name,
            max_completion_tokens=self.args.max_completion_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            seed=derive_request_seed(
                self.args.seed,
                str(item.get("doc_id", "")),
                query,
                sample_index,
            ),
        )

        if response is None:
            raise RuntimeError("Inference produced no response")
        return self._build_result(item, response)

    def _process_concurrently(self, expanded_data: list[dict[str, Any]]) -> None:
        """Process requests and persist successful results in the coordinator."""

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.max_workers, thread_name_prefix="inference_worker"
        )
        futures = [executor.submit(self.process_item, item) for item in expanded_data]
        try:
            with tqdm(
                total=len(expanded_data), desc="Processing samples", unit="sample"
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.warning("Inference sample failed: %s", e)
                        self._stats["failed"] += 1
                    else:
                        # Persistence failures are run-level failures, not model
                        # inference failures, and must stop the run immediately.
                        self._write_result(result)
                        self._stats["processed"] += 1
                    finally:
                        pbar.update(1)
        finally:
            # Don't wait for in-flight API requests when aborting: their
            # results would be discarded anyway. cancel_futures stops pending
            # submissions; running futures finish on their own threads.
            executor.shutdown(wait=False, cancel_futures=True)

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
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Total duration: {duration:.2f} seconds")
        logger.info(f"Output file: {self.args.output_file}")
        if self._stats["failed"]:
            raise RuntimeError(
                f"Inference failed for {self._stats['failed']} sample(s); "
                "successful results were preserved for resume"
            )
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
