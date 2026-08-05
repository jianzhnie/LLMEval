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

from llmeval.cache import build_cache, log_cache_stats
from llmeval.inference.common import (
    expand_group_for_sampling,
    is_explicit_tool_choice,
    is_local_endpoint,
    load_jsonl,
    load_resume_state,
    prepare_data_with_resume,
    redact_config_for_logging,
    sample_count_for_item,
    save_failed_items,
)
from llmeval.utils.config import OnlineInferArguments
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY, is_chat_template_applied
from llmeval.utils.retry import call_with_retry

logger = init_logger("online_vllm_server", logging.INFO)


def _config_for_logging(args: OnlineInferArguments) -> dict[str, Any]:
    """Return online configuration without misleading empty optional fields."""
    payload = redact_config_for_logging(dataclasses.asdict(args))
    if not payload.get("task"):
        payload.pop("task", None)
    return payload


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
        content_cache_dir: str = "",
        force_recompute: bool = False,
        read_only_cache: bool = False,
        model_revision: str | None = None,
        cache_rank: str | None = None,
        organization: str | None = None,
    ) -> None:
        """Initialize the inference client with API configuration and validation.

        Creates a new OpenAI client instance configured with the provided base URL
        and timeout settings. Validates the configuration and ensures required
        environment variables are set.

        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint
            timeout: Request timeout in seconds (must be positive)
            max_retries: Maximum number of retries for requests to VLLM server (must be non-negative)
            tool_choice: Tool calling mode: 'none' (default, disables tools), 'auto', or tool name.
            api_key: API key; falls back to the OPENAI_API_KEY env var and then EMPTY.

        Raises:
            ValueError: If timeout is invalid (<=0) or base_url is empty
        """
        self.base_url: str = base_url  # Store for potential reconnection
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.tool_choice: str = tool_choice
        self.seed = seed
        self.model_revision = model_revision
        self.cache = build_cache(
            content_cache_dir,
            "inference",
            force_recompute=force_recompute,
            read_only=read_only_cache,
            rank=cache_rank,
        )
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        if self.api_key == "EMPTY":
            log = logger.debug if is_local_endpoint(base_url) else logger.warning
            log("Using default 'EMPTY' API key.")

        # Initialize OpenAI client with validated configuration
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=httpx.Timeout(self.timeout),
            organization=organization,
        )
        masked_key = f"{self.api_key[:4]}***" if len(self.api_key) > 4 else "***"
        logger.info(
            f"Using API Key: {masked_key}, Timeout: {self.timeout}, Max Retries: {self.max_retries}, base_url: {self.base_url}"
        )

    def _prepare_messages(
        self, query: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        """Prepare messages for the API call by formatting them into the expected structure.

        This method constructs the message list in the format expected by the OpenAI API.
        It validates the input to ensure no chat template has been pre-applied and
        adds the system prompt if provided.

        Args:
            query: User's input query text
            system_prompt: Optional system prompt to set conversation context and behavior

        Returns:
            List[Dict[str, str]]: A list of message dictionaries in OpenAI chat format,
                                 each containing 'role' and 'content' keys

        Raises:
            ValueError: If chat template is already applied to the query, to prevent
                      double-application of templates
        """
        if is_chat_template_applied(query):
            logger.warning(
                "Chat template appears to be already applied to the query. "
                "Please use the raw prompt, as vLLM will apply the Hugging Face "
                "chat template automatically."
            )
            raise ValueError(
                "Your query has been applied with chat_template, please use the raw prompt, "
                "because the vLLM will apply the Hugging Face chat template automatically!"
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        return messages

    def get_content(
        self,
        query: str,
        system_prompt: str | None,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ) -> str:
        """Fetch content from the OpenAI API with comprehensive retry logic.

        This method handles the core interaction with the OpenAI API, including
        parameter validation, message preparation, and error handling. It supports
        various generation parameters and includes built-in retry logic for
        transient errors.

        Args:
            query: User's input query
            system_prompt: System prompt for the conversation (optional)
            model_name: The model to use for generation (e.g., 'gpt-3.5-turbo')
            max_tokens: Maximum tokens to generate (1 to model's context limit)
            temperature: The sampling temperature (0.0 to 2.0)
            top_p: The top-p value for nucleus sampling (0.0 to 1.0)
            top_k: The top-k value for sampling (positive integer)
            enable_thinking: Whether to enable the "thinking" feature

        Returns:
            The generated content string. May be empty ("") when the model
            produced no usable content (e.g. context length exceeded, or a
            reasoning model exhausted max_tokens during the thinking phase);
            callers treat an empty string as a failed sample.

        Raises:
            ClientError: If there's a non-retryable API issue or max retries exceeded
            ValueError: If input parameters are invalid or out of range

        Note:
            The method includes automatic retry logic for certain types of API
            errors (connection issues, rate limits) but will raise exceptions
            for non-recoverable errors.
        """
        call_args = self._build_call_args(
            query,
            system_prompt,
            model_name,
            max_tokens,
            temperature,
            top_p,
            top_k,
            enable_thinking,
        )
        cache = getattr(self, "cache", None)
        cache_key = (
            cache.key(self._cache_payload(call_args, "online_chat"))
            if cache is not None
            else None
        )
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None and isinstance(cached.get("content"), str):
                return cached["content"]
        completion = self._request_with_retry(call_args)
        if completion is None:
            return ""  # context length exceeded (logged in _request_with_retry)
        # Reasoning models may return content=None (thinking exhausted
        # max_tokens); normalize to "" so callers can treat it uniformly
        content = completion.choices[0].message.content or ""
        if cache is not None and cache_key is not None and content.strip():
            cache.set(cache_key, {"content": content})
        return content

    def get_contents(
        self,
        query: str,
        system_prompt: str | None,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
        n: int,
    ) -> list[str]:
        """Fetch n samples for the same prompt in ONE request (API `n` parameter).

        Sending n separate requests would re-run the prefill (system prompt +
        query) n times; with n>1 the server prefills once and samples n
        completions, which is significantly cheaper for pass@k-style evals.

        Args:
            query: User's input query
            system_prompt: System prompt for the conversation (optional)
            model_name: The model to use for generation
            max_tokens: Maximum tokens to generate per sample
            temperature: The sampling temperature (0.0 to 2.0)
            top_p: The top-p value for nucleus sampling (0.0 to 1.0)
            top_k: The top-k value for sampling (positive integer)
            enable_thinking: Whether to enable the "thinking" feature
            n: Number of samples to generate for this prompt (must be >= 1)

        Returns:
            List of generated content strings, one per returned choice
            (null contents normalized to ""). Empty list when the request
            could not produce samples (e.g. context length exceeded).

        Raises:
            ClientError: If there's a non-retryable API issue or max retries exceeded
            ValueError: If input parameters are invalid or out of range
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got: {n}")
        call_args = self._build_call_args(
            query,
            system_prompt,
            model_name,
            max_tokens,
            temperature,
            top_p,
            top_k,
            enable_thinking,
            n=n,
        )
        cache = getattr(self, "cache", None)
        cache_key = (
            cache.key(self._cache_payload(call_args, "online_chat_multi"))
            if cache is not None
            else None
        )
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None and isinstance(cached.get("contents"), list):
                return [str(content) for content in cached["contents"]]
        completion = self._request_with_retry(call_args)
        if completion is None:
            return []  # context length exceeded (logged in _request_with_retry)
        contents = [choice.message.content or "" for choice in completion.choices]
        if (
            cache is not None
            and cache_key is not None
            and contents
            and all(content.strip() for content in contents)
        ):
            cache.set(cache_key, {"contents": contents})
        return contents

    def _build_call_args(
        self,
        query: str,
        system_prompt: str | None,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
        n: int = 1,
    ) -> dict[str, Any]:
        """Validate inputs and assemble chat.completions call arguments.

        Args:
            query: User's input query (must be non-empty)
            system_prompt: Optional system prompt
            model_name: Served model name (must be non-empty)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter (sent via extra_body)
            enable_thinking: Whether to enable the "thinking" feature
            n: Number of samples per request; only sent when > 1

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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "top_k": top_k,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
            "timeout": self.timeout,
            "seed": getattr(self, "seed", 0),
        }
        # n: only sent for multi-sample requests (single-sample default is 1)
        if n > 1:
            call_args["n"] = n
        # tool_choice: only send when explicitly configured (vLLM 0.23+ supports it)
        if is_explicit_tool_choice(self.tool_choice):
            call_args["tool_choice"] = self.tool_choice
        return call_args

    def _cache_payload(self, call_args: dict[str, Any], kind: str) -> dict[str, Any]:
        """Describe all content-affecting inputs for an online request."""
        return {
            "backend": kind,
            "endpoint": self.base_url,
            "model_name": call_args.get("model"),
            "model_revision": getattr(self, "model_revision", None),
            "messages": call_args.get("messages", []),
            "generation_params": {
                key: value
                for key, value in call_args.items()
                if key not in {"model", "messages", "timeout"}
            },
            "sampling_seed": call_args.get("seed"),
            "postprocess_version": "online_chat_v1",
        }

    def _request_with_retry(self, call_args: dict[str, Any]) -> Any | None:
        """Execute a chat.completions call under the shared retry policy.

        Shared by get_content (single sample) and get_contents (n samples).
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
            _ = completion.choices[0].message
            return completion

        return call_with_retry(do_request, self.max_retries)


class InferenceRunner:
    """
    Main class to handle the inference process with concurrent execution.

    This class orchestrates the entire inference pipeline, including:
    - Data loading and validation
    - Resume functionality for interrupted runs
    - Concurrent processing with thread management
    - Progress tracking and reporting
    - Error handling and recovery
    - Result persistence

    Attributes:
        args (OnlineInferArguments): Configuration arguments for the inference process
        client (InferenceClient): The inference client instance for API interactions
        system_prompt (Optional[str]): System prompt template for conversation context
        _file_lock (threading.Lock): Thread lock for safe file writing operations
        _stats (Dict[str, int]): Runtime statistics for monitoring progress
    """

    def __init__(self, args: OnlineInferArguments) -> None:
        """Initialize the inference runner with comprehensive validation and setup.

        Sets up the inference pipeline with the provided configuration, including:
        - Client initialization
        - System prompt configuration
        - Thread safety mechanisms
        - Progress monitoring

        Args:
            args: Configuration arguments containing all necessary settings for
                 the inference pipeline, including API configuration, input/output
                 paths, and generation parameters

        Raises:
            ValueError: If arguments are invalid, inconsistent, or missing required values
            FileNotFoundError: If specified input file doesn't exist
            EnvironmentError: If required environment variables are not set

        Note:
            The runner is designed for thread-safe concurrent execution with
            proper resource management and progress tracking.
        """
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
                content_cache_dir=args.content_cache_dir,
                force_recompute=args.force_recompute,
                read_only_cache=args.read_only_cache,
                model_revision=getattr(args, "model_revision", None),
                cache_rank=getattr(args, "cache_rank", None),
                organization=args.organization,
            )
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to initialize inference client: {e}") from e

        # Set up system prompt with validation
        self.system_prompt: str | None = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type
        )
        if (
            args.system_prompt_type
            and args.system_prompt_type != "empty"
            and not self.system_prompt
        ):
            logger.warning(f"Unknown system_prompt_type: {args.system_prompt_type}")

        # Initialize thread safety and monitoring
        self._file_lock: threading.Lock = threading.Lock()
        self._stats: dict[str, int] = {"processed": 0, "failed": 0, "skipped": 0}
        self._stats_lock: threading.Lock = threading.Lock()  # Dedicated lock for stats

    def load_data(self) -> list[dict[str, Any]]:
        """Load and prepare the dataset, handling resume functionality.

        This method performs several key operations:
        1. Loads and validates the input data file
        2. Checks for previously completed samples
        3. Annotates each remaining prompt with an ``n_samples`` count
        4. Validates data structure and content

        Returns:
            List[Dict[str, Any]]: Prompt records to process, with ``n_samples``
                set to the remaining sample count after resume.

        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If input file contains invalid JSON
            ValueError: If input data structure is invalid
        """
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
        )

        if resume_state.completed_count > 0:
            logger.info(
                "Found %d completed samples from previous run.",
                resume_state.completed_count,
            )

        prepared_data = prepare_data_with_resume(
            raw_data,
            resume_state.completed_indices,
            resume_state.legacy_counts,
            self.args.input_key,
            self.args.n_samples,
        )
        total_remaining = sum(sample_count_for_item(item) for item in prepared_data)

        if not prepared_data:
            logger.warning("No data to process after preparation")

        logger.info(f"Total remaining prompts to process: {len(prepared_data)}")
        logger.info(f"Total remaining samples to process: {total_remaining}")
        return prepared_data

    def _write_result(self, result: dict[str, Any]) -> None:
        """Write result to output file in a thread-safe manner.

        Args:
            result: The result dictionary to write
        """
        with self._file_lock:
            try:
                with open(self.args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # Ensure data is immediately written
            except Exception as e:
                logger.error(f"Error writing batch results: {e}")
                raise OSError(f"Failed to write batch results: {e}") from e

    def _extract_query(self, item: Any) -> str | None:
        """Validate the item structure and extract the query text.

        Updates failure/skip statistics when the item is unusable.  Takes
        ``Any`` (not ``dict``) because malformed input lines can surface as
        non-dict items at runtime — that is exactly what this method rejects.

        Args:
            item: Raw data item (expected to be a dict with a prompt field)

        Returns:
            The query string, or None if the item is invalid (stats updated).
        """
        if not isinstance(item, dict):
            logger.error(f"Invalid item type: {type(item)}, expected dict")
            with self._stats_lock:
                self._stats["failed"] += 1
            return None

        query = item.get(self.args.input_key) or item.get("prompt")
        if not query:
            logger.warning(f"Missing required query field in item: {item}")
            with self._stats_lock:
                self._stats["skipped"] += 1
            return None
        return query

    def _build_result(
        self, item: dict[str, Any], response: str
    ) -> dict[str, Any] | None:
        """Validate the API response and append it to the item's gen list.

        Updates failure statistics when the response is unusable. The input
        item is never mutated; a shallow copy with a fresh gen list is returned.

        Args:
            item: The source data item
            response: Generated content; empty string means failure
                (get_content normalizes null content from reasoning models to "")

        Returns:
            The result dict, or None if the response is empty (stats updated).
        """
        if not response.strip():
            logger.warning("Empty response received")
            with self._stats_lock:
                self._stats["failed"] += 1
            return None

        result = item.copy()
        result.pop("n_samples", None)
        result.pop("_llmeval_sample_start", None)
        result.pop("_llmeval_requested_sample_indices", None)
        gen_list = list(result.get(self.args.response_key, []))
        gen_list.append(response)
        result[self.args.response_key] = gen_list
        return result

    def process_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single item through the complete inference pipeline.

        This method implements a robust processing pipeline for each input item:
        1. Input validation and query extraction
        2. API request preparation and execution
        3. Response validation and processing
        4. Result persistence with thread safety
        5. Comprehensive error handling and recovery

        Args:
            item: The data item to process containing query and metadata

        Returns:
            Optional[Dict[str, Any]]: Processed result or None if processing failed

        Note:
            - Thread-safe execution with proper resource management
            - Detailed logging of processing steps and errors
            - Automatic retry logic for transient failures
            - Progress tracking and statistics collection
        """
        # Step 1: Input Validation
        query = self._extract_query(item)
        if not query:
            return None

        # Step 2: API Request
        response = self.client.get_content(
            query=query,
            system_prompt=self.system_prompt,
            model_name=self.args.model_name,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            enable_thinking=self.args.enable_thinking,
        )

        # Step 3: Response Processing
        result = self._build_result(item, response)
        if not result:
            return None

        # Step 4: Result Persistence
        try:
            self._write_result(result)
            with self._stats_lock:
                self._stats["processed"] += 1
        except OSError as e:
            logger.error(f"Failed to write result: {e}")
            with self._stats_lock:
                self._stats["failed"] += 1
            return None

        return result

    def process_item_group(self, items: list[dict[str, Any]]) -> None:
        """Process one prompt group with a single batched request when possible.

        ``load_data`` records the remaining sample count in ``n_samples``.
        This method issues one request with the API's n parameter (one prefill,
        n samples) and writes one result line per sample, keeping the output
        format identical to per-sample processing.

        Args:
            items: Records sharing the same prompt (len >= 1)
        """
        sample_items = expand_group_for_sampling(items)
        request_n_samples = len(sample_items)

        if request_n_samples == 1:
            self.process_item(sample_items[0])
            return

        # Step 1: Input Validation (all copies share the same prompt)
        query = self._extract_query(sample_items[0])
        if not query:
            return

        # Step 2: ONE batched API request for all copies
        responses = self.client.get_contents(
            query=query,
            system_prompt=self.system_prompt,
            model_name=self.args.model_name,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            enable_thinking=self.args.enable_thinking,
            n=request_n_samples,
        )

        # Steps 3+4: Per-copy result building and persistence
        if not responses:
            # e.g. context length exceeded: every copy of this prompt failed
            with self._stats_lock:
                self._stats["failed"] += request_n_samples
            return
        for item, response in zip(sample_items, responses, strict=False):
            result = self._build_result(item, response)
            if not result:
                continue  # empty response already counted failed in _build_result
            try:
                self._write_result(result)
                with self._stats_lock:
                    self._stats["processed"] += 1
            except OSError as e:
                logger.error(f"Failed to write result: {e}")
                with self._stats_lock:
                    self._stats["failed"] += 1
        # Fewer choices than requested: count the missing copies as failed
        missing = request_n_samples - len(responses)
        if missing > 0:
            logger.warning(
                f"Batched request returned {len(responses)}/{request_n_samples} samples"
            )
            with self._stats_lock:
                self._stats["failed"] += missing

    def _group_sample_count(self, items: list[dict[str, Any]]) -> int:
        """Return the number of samples represented by a concurrent group."""
        return len(expand_group_for_sampling(items))

    def _process_concurrently(self, expanded_data: list[dict[str, Any]]) -> None:
        """Process items concurrently using thread pool with error handling and progress tracking.

        Records that share a prompt are grouped and processed with a single
        batched request (API n parameter, one prefill for n samples). Records
        with n_samples=1 go through the regular per-item path.

        Args:
            expanded_data: List of data items to process, where each item is a
                        dictionary containing the input data and metadata

        Note:
            - Uses ThreadPoolExecutor for concurrent processing
            - Implements proper error handling for each thread
            - Shows progress bar with tqdm
            - Maintains thread safety with class-level file lock
        """
        total_tasks = sum(
            sample_count_for_item(item) if isinstance(item, dict) else 1
            for item in expanded_data
        )
        failed_tasks: list[dict[str, Any]] = []

        # Group by stable document ID for batched (n-parameter) sampling.
        # Using the prompt as the key would merge distinct records that happen
        # to contain the same text and would corrupt their resume counts.
        groups: dict[Any, list[dict[str, Any]]] = {}
        for idx, item in enumerate(expanded_data):
            key: Any = None
            if isinstance(item, dict):
                candidate = item.get("doc_id")
                if candidate is not None and str(candidate).strip():
                    key = str(candidate)
                else:
                    # Direct callers and legacy in-memory inputs may not carry
                    # an ID yet; preserve the old prompt grouping in that case.
                    prompt = item.get(self.args.input_key) or item.get("prompt")
                    if isinstance(prompt, str) and prompt:
                        key = prompt
            if key is None:
                key = ("__invalid__", idx)
            groups.setdefault(key, []).append(item)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.max_workers, thread_name_prefix="inference_worker"
        ) as executor:
            futures = {
                executor.submit(self.process_item_group, group): group
                for group in groups.values()
            }

            with tqdm(
                total=total_tasks, desc="Processing samples", unit="sample"
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    group = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(
                            f"An unexpected error occurred in a thread: {e}",
                            exc_info=True,
                        )
                        group_samples = self._group_sample_count(group)
                        with self._stats_lock:
                            self._stats["failed"] += group_samples
                        sample = group[0]
                        prompt_val = (
                            sample.get(self.args.input_key, "") or sample.get("prompt")
                            if isinstance(sample, dict)
                            else None
                        )
                        failed_tasks.append(
                            {
                                self.args.input_key: (
                                    str(prompt_val)[:200]
                                    if prompt_val is not None
                                    else None
                                ),
                                "samples": group_samples,
                                "error": str(e),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                    finally:
                        pbar.update(self._group_sample_count(group))

        if failed_tasks:
            logger.warning(f"Total failed tasks: {len(failed_tasks)}")
            save_failed_items(self.args.output_file, failed_tasks)

    def run(self) -> None:
        """Execute the complete inference pipeline with monitoring and reporting.

        This method orchestrates the entire inference workflow:
        1. Configuration validation
        2. Data loading and preprocessing
        3. Concurrent execution management
        4. Progress monitoring and reporting
        5. Resource cleanup and final reporting

        The pipeline includes automatic resume capability and comprehensive
        error handling at each stage.

        Raises:
            FileNotFoundError: If input file is missing
            ValueError: If configuration is invalid
            RuntimeError: For unrecoverable execution errors
        """
        start_time = time.perf_counter()

        try:
            # Validate configuration
            if not self.args.input_file or not Path(self.args.input_file).exists():
                raise FileNotFoundError(f"Input file not found: {self.args.input_file}")
            if not self.args.output_file:
                raise ValueError("Output file path is required")

            # Initialize execution
            logger.info("🚀 Initializing inference pipeline")
            logger.info("Configuration: %s", _config_for_logging(self.args))

            # Load and prepare data
            eval_dataset: list[dict[str, Any]] = self.load_data()
            if not eval_dataset:
                logger.info("✅ All samples already processed")
                return

            # Set up output directory
            output_path = Path(self.args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Execute pipeline
            total_samples = sum(sample_count_for_item(item) for item in eval_dataset)
            logger.info(f"⏳ Processing {total_samples} samples")
            self._process_concurrently(eval_dataset)

            # Generate final report
            duration = time.perf_counter() - start_time
            success_rate = (self._stats["processed"] / max(total_samples, 1)) * 100

            logger.info("\n=== Execution Summary ===")
            logger.info(f"Total samples in dataset: {total_samples}")
            logger.info(f"Successfully processed: {self._stats['processed']}")
            logger.info(f"Failed: {self._stats['failed']}")
            logger.info(f"Skipped: {self._stats['skipped']}")
            logger.info(f"Success rate: {success_rate:.2f}%")
            logger.info(f"Total duration: {duration:.2f} seconds")
            logger.info(f"Output file: {self.args.output_file}")
            log_cache_stats(self.client.cache, logger, "Online inference")
            logger.info("✅ Inference pipeline completed successfully\n")

        except Exception as e:
            logger.critical(
                f"❌ Fatal error: {e!s}", exc_info=True, extra={"stats": self._stats}
            )
            raise RuntimeError(f"Pipeline execution failed: {e!s}") from e


def main() -> None:
    """
    Main entry point for the online inference server.

    This function serves as the primary entry point for the inference server,
    handling:
    1. Command line argument parsing using HfArgumentParser
    2. Initialization of the inference runner
    3. Execution of the inference process
    4. Comprehensive error handling and logging

    The function uses dataclasses for type-safe argument handling and provides
    detailed logging of the initialization process and any errors that occur.

    Returns:
        None

    Raises:
        SystemExit: If initialization fails or command line arguments are invalid
        Exception: For any unhandled errors during execution
    """
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
        logger.info(json.dumps(_config_for_logging(eval_args), indent=2))

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
