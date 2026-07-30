"""Multiple-choice inference: loglikelihood and generation modes.

Architecture mirrors llmeval/vllm/online_server.py:
- MCLoglikelihoodClient: low-level API client with retry logic
- MCGenerateClient: chat completions client for generate mode
- MCRunner: orchestrates inference with resume, threading, stats
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai

from llmeval.utils.logger import init_logger
from llmeval.utils.template import SYSTEM_PROMPT_FACTORY

logger = init_logger("mc_infer")


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class MCInferConfig:
    """Configuration for MC inference."""

    input_file: str = ""
    output_file: str = ""
    base_url: str = "http://127.0.0.1:8200/v1"
    model_name: str = "longcat-flash"
    mode: str = "loglikelihood"  # "loglikelihood" | "generate"
    max_workers: int = 32
    request_timeout: int = 300
    max_retries: int = 3
    max_tokens: int = 2048
    temperature: float = 0.0
    system_prompt_type: str = "empty"
    tool_choice: str = "none"
    n_shot: int = 0  # few-shot examples count (0 = zero-shot)
    few_shot_file: str = (
        ""  # separate dev file for few-shot (uses input_file first N if empty)
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY")
    )


# ===========================================================================
# Few-shot formatter
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
        self._few_shot_pool: list[dict] = []
        self._all_formatted: list[str] = []

    def load(self, input_file: str) -> str:
        """Load few-shot examples and return the formatted prefix string."""
        if self.n_shot <= 0:
            return ""

        source = self.few_shot_file or input_file
        try:
            items = self._load_items(source)
        except Exception:
            logger.warning(f"Failed to load few-shot from {source}")
            return ""

        if len(items) < self.n_shot:
            logger.warning(f"Only {len(items)} examples available, need {self.n_shot}")
            return ""

        rng = random.Random(self.seed)
        # Sample n_shot+1: extra for dedup (lm-eval style)
        selected = rng.sample(items, min(self.n_shot + 1, len(items)))
        self._few_shot_pool = selected  # store for per-item dedup
        self._all_formatted = [self._format_demo(it) for it in selected]
        logger.info(f"Loaded {self.n_shot} few-shot examples (seed={self.seed})")
        return ""  # prefix built per-item for dedup

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
    def _format_demo(item: dict) -> str:
        """Format one few-shot demonstration."""
        prompt = item.get("prompt", "")
        answer = item.get("answer", "")
        # prompt already ends with "Answer:", append the answer
        return f"{prompt} {answer}"

    @staticmethod
    def _load_items(filepath: str) -> list[dict[str, Any]]:
        items = []
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
    """Client for computing choice log-probabilities via completions API.

    Mirrors InferenceClient in online_server.py.
    """

    def __init__(
        self, base_url: str, model_name: str, timeout: int = 300, max_retries: int = 3
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        if self.api_key == "EMPTY":
            logger.warning("Using default 'EMPTY' API key.")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
        logger.info(
            f"MC Client initialized: model={model_name}, timeout={timeout}, "
            f"max_retries={max_retries}, base_url={base_url}"
        )

    def get_choice_logprob(self, prompt: str, choice_text: str) -> float:
        """Compute log-probability of choice given prompt.

        Uses completions API with echo=True. Sums all token logprobs.
        Since prompt is identical across choices, logprob(prompt) cancels
        out in comparison: argmax Σ logprob(prompt+choice_i) = argmax logprob(choice_i|prompt).
        """
        full_text = f"{prompt} {choice_text}"
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.completions.create(
                    model=self.model_name,
                    prompt=full_text,
                    max_tokens=1,  # minimal: only need logprobs, not generation
                    temperature=0,  # deterministic for logprob computation
                    logprobs=1,  # only need top-1 token logprob per position
                    echo=True,  # return logprobs for all prompt tokens
                    timeout=self.timeout,
                )
                logprob_data = resp.choices[0].logprobs
                if logprob_data and logprob_data.token_logprobs:
                    return sum(
                        lp for lp in logprob_data.token_logprobs if lp is not None
                    )
                return float("-inf")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(2**attempt, 30)
                    logger.debug(
                        f"Retry {attempt + 1}/{self.max_retries} in {delay}s: {e}"
                    )
                    time.sleep(delay)
        logger.warning(
            f"Logprob request failed after {self.max_retries + 1} attempts: {last_error}"
        )
        return float("-inf")


# ===========================================================================
# Runner
# ===========================================================================


class MCRunner:
    """Orchestrates MC inference with resume, threading, and stats.

    Mirrors InferenceRunner in online_server.py.
    """

    def __init__(self, config: MCInferConfig) -> None:
        self.config = config
        self._file_lock = threading.Lock()
        self._stats: dict[str, int] = {"processed": 0, "failed": 0, "skipped": 0}
        self._stats_lock = threading.Lock()

        if config.mode == "loglikelihood":
            self.client = MCLoglikelihoodClient(
                base_url=config.base_url,
                model_name=config.model_name,
                timeout=config.request_timeout,
                max_retries=config.max_retries,
            )
        else:
            self.client = None  # generate mode uses direct API calls

        # System prompt for generate mode
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

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _get_completed_prompts(self) -> set[str]:
        """Get set of completed prompts from existing output (for resume)."""
        output_path = Path(self.config.output_file)
        if not output_path.exists() or output_path.stat().st_size == 0:
            return set()
        completed = set()
        try:
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        prompt = item.get("prompt", "")
                        if prompt:
                            completed.add(prompt)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return completed

    # ------------------------------------------------------------------
    # Loglikelihood mode
    # ------------------------------------------------------------------

    def _run_loglikelihood(self) -> None:
        items = self._load_items()
        if not items:
            return

        # Resume: skip items whose prompt already exists in output
        completed_prompts = self._get_completed_prompts()
        remaining = [
            it for it in items if it.get("prompt", "") not in completed_prompts
        ]
        if completed_prompts:
            logger.info(
                f"Found {len(completed_prompts)} completed items, {len(remaining)} remaining"
            )
        if not remaining:
            logger.info("✅ All items already processed")
            return

        logger.info(
            f"⏳ Processing {len(remaining)} items (~{len(remaining) * 4} loglikelihood requests)"
        )

        # Process with thread pool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            futures = {
                executor.submit(self._process_loglikelihood_item, item): i
                for i, item in enumerate(remaining)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self._write_result(result)
                        with self._stats_lock:
                            self._stats["processed"] += 1
                except Exception as e:
                    with self._stats_lock:
                        self._stats["failed"] += 1
                    logger.warning(f"Item failed: {e}")

        self._log_stats()

    def _process_loglikelihood_item(
        self, item: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process a single MC item via loglikelihood comparison."""
        fs_prefix = (
            self._few_shot_fmt.get_prefix(item.get("prompt", ""))
            if self._few_shot_fmt
            else ""
        )
        prompt = fs_prefix + item.get("prompt", "")
        choices = item.get("choices", [])
        gold = item.get("gold", -1)

        if not choices:
            return None

        logprobs = []
        for choice_text in choices:
            lp = self.client.get_choice_logprob(prompt, choice_text)
            logprobs.append(lp)

        return {"prompt": prompt, "gold": gold, "logprobs": logprobs}

    # ------------------------------------------------------------------
    # Generate mode
    # ------------------------------------------------------------------

    def _run_generate(self) -> None:
        items = self._load_items()
        if not items:
            return

        completed_prompts = self._get_completed_prompts()
        remaining = [
            it for it in items if it.get("prompt", "") not in completed_prompts
        ]
        if completed_prompts:
            logger.info(
                f"Found {len(completed_prompts)} completed items, {len(remaining)} remaining"
            )
        if not remaining:
            logger.info("✅ All items already processed")
            return

        logger.info(f"⏳ Processing {len(remaining)} samples (generate mode)")
        gen_client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

        base_messages: list[dict[str, str]] = []
        if self.system_prompt:
            base_messages.append({"role": "system", "content": self.system_prompt})

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._process_generate_item, item, gen_client, base_messages
                ): i
                for i, item in enumerate(remaining)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self._write_result(result)
                        with self._stats_lock:
                            self._stats["processed"] += 1
                except Exception as e:
                    with self._stats_lock:
                        self._stats["failed"] += 1
                    logger.warning(f"Item failed: {e}")

        self._log_stats()

    def _process_generate_item(
        self,
        item: dict[str, Any],
        client: openai.OpenAI,
        base_messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Process a single MC item via text generation."""
        fs_prefix = (
            self._few_shot_fmt.get_prefix(item.get("prompt", ""))
            if self._few_shot_fmt
            else ""
        )
        prompt = fs_prefix + item.get("prompt", "")
        gold = item.get("answer", "")
        messages = [*base_messages, {"role": "user", "content": prompt}]

        gen_text = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "timeout": self.config.request_timeout,
                }
                if self.config.tool_choice:
                    kwargs["tool_choice"] = self.config.tool_choice
                resp = client.chat.completions.create(**kwargs)
                gen_text = resp.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.warning(f"Generate failed after retries: {e}")

        return {"prompt": prompt, "answer": gold, "gen": [gen_text]}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_items(self) -> list[dict[str, Any]]:
        items = []
        with open(self.config.input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        logger.info(f"Loaded {len(items)} items from {self.config.input_file}")
        return items

    def _write_result(self, result: dict[str, Any]) -> None:
        """Thread-safe write to output file."""
        with self._file_lock:
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _log_stats(self) -> None:
        logger.info(
            f"Stats: {self._stats['processed']} processed, "
            f"{self._stats['failed']} failed, "
            f"{self._stats['skipped']} skipped"
        )
        # Quick accuracy summary for loglikelihood mode
        if self.config.mode == "loglikelihood":
            self._print_loglikelihood_summary()

    def _print_loglikelihood_summary(self) -> None:
        """Print accuracy summary from output file."""
        try:
            correct = 0
            total = 0
            with open(self.config.output_file, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line.strip())
                    logprobs = r.get("logprobs", [])
                    gold = r.get("gold", -1)
                    if logprobs and gold >= 0:
                        total += 1
                        if max(range(len(logprobs)), key=lambda i: logprobs[i]) == gold:
                            correct += 1
            if total:
                logger.info(
                    f"Accuracy (loglikelihood): {correct}/{total} = {correct / total:.2%}"
                )
        except Exception:
            pass

    def run(self) -> None:
        """Main entry point."""
        start_time = time.perf_counter()

        logger.info(
            "Initializing MCInferArguments with parsed command line arguments..."
        )
        logger.info("\n--- Parsed Arguments ---")
        log_data = {
            "input_file": self.config.input_file,
            "output_file": self.config.output_file,
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "mode": self.config.mode,
            "max_workers": self.config.max_workers,
            "request_timeout": self.config.request_timeout,
            "max_retries": self.config.max_retries,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system_prompt_type": self.config.system_prompt_type,
            "tool_choice": self.config.tool_choice,
            "n_shot": self.config.n_shot,
        }
        logger.info(json.dumps(log_data, indent=2))

        logger.info(f"🚀 Initializing MC inference pipeline ({self.config.mode} mode)")

        if self.config.mode == "loglikelihood":
            self._run_loglikelihood()
        elif self.config.mode == "generate":
            self._run_generate()
        else:
            logger.error(f"Unknown mode: {self.config.mode}")
            sys.exit(1)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"✅ MC inference completed in {elapsed:.2f} seconds. Results: {self.config.output_file}"
        )


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MC inference (loglikelihood or generate)"
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--model_name", default="longcat-flash")
    parser.add_argument(
        "--mode", default="loglikelihood", choices=["loglikelihood", "generate"]
    )
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system_prompt_type", default="empty")
    parser.add_argument("--tool_choice", default="none")
    parser.add_argument(
        "--n_shot", type=int, default=0, help="Few-shot examples count (0=zero-shot)"
    )
    parser.add_argument(
        "--few_shot_file",
        default="",
        help="Dev file for few-shot (uses input_file first N if empty)",
    )
    args = parser.parse_args()

    config = MCInferConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        base_url=args.base_url,
        model_name=args.model_name,
        mode=args.mode,
        max_workers=args.max_workers,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt_type=args.system_prompt_type,
        tool_choice=args.tool_choice,
        n_shot=args.n_shot,
        few_shot_file=args.few_shot_file,
    )
    runner = MCRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
