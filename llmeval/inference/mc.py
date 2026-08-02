"""Multiple-choice inference: loglikelihood and generation modes.

This module mirrors llmeval/inference/online_server.py in structure, naming, and
documentation style so the two can be reviewed side by side with a file diff:
- MCLoglikelihoodClient  ~  InferenceClient
- MCRunner               ~  InferenceRunner
- main() (HfArgumentParser) ~  main()

Shared utilities (ClientError, retry classification, backoff) live in
llmeval/utils/api_retry.py; the configuration dataclass lives in
llmeval/utils/config.py (MCInferConfig).
MC-specific pieces (kept deliberately): FewShotFormatter, batched choice
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

from llmeval.tasks.mc_eval.mc_score import argmax
from llmeval.utils.config import MCInferConfig
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY
from llmeval.utils.retry import should_retry

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
            items = self._load_items(source)
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

    @staticmethod
    def _load_items(filepath: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items


# ===========================================================================
# Loglikelihood Client
# ===========================================================================


class MCLoglikelihoodClient:
    """Client for computing choice log-probabilities via the completions API.

    Mirrors InferenceClient in online_server.py: same initialization, masked
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
        """Compute per-choice log-probabilities from first-token top_logprobs.

        Uses echo=False + max_tokens=1 + logprobs=20 to obtain the model's top
        predicted tokens after the prompt. For each target choice, we look up
        its logprob among the predictions. This directly measures
        P(target_token | prompt) without the token-alignment issues that arise
        from echo=True (where prompt tokenization can shift across continuations).

        The target choices (typically "A"/"B"/"C"/"D") are looked up in several
        common tokenizer forms: with/without leading space, upper/lower case.

        Args:
            prompt: The shared MC prompt (few-shot prefix + question)
            choice_texts: Candidate answer tokens (e.g. ["A","B","C","D"])

        Returns:
            One logprob per choice, aligned with choice_texts. Choices not found
            among the top predictions get float("-inf"). All -inf when the
            request fails after all retries.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
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
            except Exception as e:
                last_error = e
                action = should_retry(e, attempt, self.max_retries)
                if action is False:
                    break
                if action is None:
                    logger.warning(f"Unclassified error: {type(e).__name__}: {e!s}")
                    break
        logger.warning(
            f"Logprob request failed after {self.max_retries + 1} attempts: {last_error}"
        )
        return [float("-inf")] * len(choice_texts)


# ===========================================================================
# Runner
# ===========================================================================


class MCRunner:
    """Orchestrates MC inference with resume, threading, and stats.

    Mirrors InferenceRunner in online_server.py: same pipeline stages
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

    def get_completed_prompts(self) -> set[str]:
        """Get the set of completed prompts from existing output (for resume).

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
        raw_data = self._load_raw_data()
        logger.info(f"Loaded {len(raw_data)} items from input file")

        # Resume: skip items whose (few-shot prefixed) prompt is already written.
        # build_prompt is used on both sides so resume agrees with inference
        # even when n_shot > 0 changes the prompt that gets written.
        completed_prompts = self.get_completed_prompts()
        if completed_prompts:
            logger.info(f"Found {len(completed_prompts)} completed items.")
        remaining = [
            it for it in raw_data if self.build_prompt(it) not in completed_prompts
        ]

        logger.info(f"Total remaining samples to process: {len(remaining)}")
        return remaining

    def _load_raw_data(self) -> list[dict[str, Any]]:
        """Load raw data from the input file.

        Returns:
            List of raw data items.

        Raises:
            FileNotFoundError: If the input file does not exist
            json.JSONDecodeError: If an input line is not valid JSON
        """
        try:
            with open(self.config.input_file, encoding="utf-8") as f:
                data: list[dict[str, Any]] = [
                    json.loads(line) for line in f if line.strip()
                ]
        except FileNotFoundError as e:
            logger.critical(f"Input file not found: {self.config.input_file}, {e}")
            raise
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON in input file: {e}")
            raise
        return data

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
                            with self._stats_lock:
                                self._stats["processed"] += 1
                                if result.get("correct"):
                                    self._stats["correct"] += 1
                    pbar.update(1)

        if failed_items:
            logger.warning(f"Total failed tasks: {len(failed_items)}")
            # Save failed items next to the output file. NOTE: splitext, not
            # str.replace — a non-.jsonl output name must not collapse onto
            # the output file itself ("w" mode would truncate it).
            failed_file = os.path.splitext(self.config.output_file)[0] + "_failed.jsonl"
            try:
                with open(failed_file, "w", encoding="utf-8") as f:
                    for entry in failed_items:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logger.info(f"Failed items saved to: {failed_file}")
            except OSError as e:
                logger.error(f"Failed to save failed items to file: {e}")

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

        logprobs = self.client.get_choices_logprobs(prompt, choices)
        if all(lp == float("-inf") for lp in logprobs):
            raise RuntimeError("Logprob request failed for all choices")

        pred = argmax(logprobs) if logprobs else -1
        is_correct = pred == gold
        return {
            self.config.input_key: prompt,
            "choices": choices,  # mc_score 的 acc_norm 需要选项文本做长度归一
            "gold": gold,
            "logprobs": logprobs,
            "pred": pred,
            "correct": is_correct,
        }

    # ------------------------------------------------------------------
    # Generate mode
    # ------------------------------------------------------------------

    def run_generate(self, remaining: list[dict[str, Any]]) -> None:
        """Run the generate pipeline over the remaining items.

        Args:
            remaining: Items left after resume filtering (see load_data)
        """
        logger.info(f"⏳ Processing {len(remaining)} samples (generate mode)")
        gen_client: openai.OpenAI = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
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

        gen_text = ""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                call_args: dict[str, Any] = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "timeout": self.config.request_timeout,
                }
                # tool_choice: only send when explicitly configured
                if self.config.tool_choice:
                    call_args["tool_choice"] = self.config.tool_choice
                resp = client.chat.completions.create(**call_args)
                # Reasoning models may return content=None (thinking exhausted
                # max_tokens); normalize to "" — the empty result is a failure
                gen_text = resp.choices[0].message.content or ""
                break
            except Exception as e:
                last_error = e
                action = should_retry(e, attempt, self.config.max_retries)
                if action is False:
                    break
                if action is None:
                    logger.warning(
                        f"Generate aborted: unclassified {type(e).__name__}: {e!s}"
                    )
                    break
        if not gen_text:
            raise RuntimeError(f"Generate produced no usable text: {last_error}")

        return {
            self.config.input_key: prompt,
            self.config.label_key: gold,
            self.config.response_key: [gen_text],
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

    Mirrors main() in online_server.py: HfArgumentParser builds MCInferConfig
    directly from the dataclass (field names == CLI flags), then the runner
    executes with standardized exit-code handling.

    Raises:
        SystemExit: 130 on keyboard interrupt, 1 on any fatal error
    """
    start_time = time.perf_counter()
    try:
        # Parse command line arguments into a strongly typed dataclass
        parser = HfArgumentParser(MCInferConfig)
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
