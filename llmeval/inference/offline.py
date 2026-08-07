"""Offline vLLM inference runner.

This module provides a small, documented wrapper around vLLM to:
- Load a line-delimited JSON dataset
- Resume generation per unique prompt up to a requested sample count
- Convert records into vLLM chat message format
- Run batched chat inference
- Persist unified results incrementally for robustness

The output schema stores one generation per JSONL row.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from llmeval.inference.common import (
    build_vllm_llm_kwargs,
    expand_data_with_resume,
    get_request_seed,
    load_jsonl,
    load_resume_state,
    process_batches_with_policy,
    redact_config_for_logging,
    save_failed_items,
)
from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY, is_chat_template_applied

# Initialize logger
logger = init_logger("offline_vllm_infer", logging.INFO)


def _sample_failure(
    item: dict[str, Any], category: str, error: object
) -> dict[str, Any]:
    """Build a compact failed-sample audit record."""
    return {
        "item": {"doc_id": item["doc_id"]} if "doc_id" in item else {},
        "error_category": category,
        "error_type": type(error).__name__,
        "error": str(error),
    }


class OfflineInferenceRunner:
    """Main class to handle offline inference with the vLLM engine.

    This runner:
    - Loads input data and expands per-record sampling counts with resume support.
    - Converts input records into vLLM chat message format.
    - Runs batched inference using vLLM.
    - Writes results to a line-delimited JSON output file in a unified schema.

    Attributes:
        args: Configuration arguments for the inference process
        _file_lock: Thread lock for safe file writing operations
        llm: vLLM engine instance (initialized during setup)
        system_prompt: System prompt text (resolved from system_prompt_type)
    """

    def __init__(self, args: OfflineInferArguments) -> None:
        """Initialize the runner with parsed CLI arguments.

        Args:
            args: Parsed `OfflineInferArguments` used to configure vLLM and IO.

        Raises:
            ValueError: If arguments are invalid or missing required fields.
        """
        self.args: OfflineInferArguments = args
        self._file_lock: threading.Lock = threading.Lock()
        self.llm: LLM | None = None
        self.system_prompt: str | None = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type
        )

    def setup_vllm_engine(self) -> LLM:
        """Initialize and return the configured vLLM engine."""
        logger.info("=" * 60)
        logger.info("🚀 Initializing vLLM Engine")
        logger.info(f"Model: {self.args.model_name_or_path}")
        logger.info(f"Max Model Length: {self.args.max_model_len}")
        logger.info(f"Max tokens: {self.args.max_tokens}")
        logger.info(f"RoPE Scaling: {self.args.rope_scaling}")
        logger.info(f"Tensor Parallel Size: {self.args.tensor_parallel_size}")
        logger.info(f"Pipeline Parallel Size: {self.args.pipeline_parallel_size}")
        logger.info(f"GPU Memory Utilization: {self.args.gpu_memory_utilization}")
        logger.info(f"Batch Size: {self.args.batch_size}")
        logger.info("=" * 60)

        # Prepare HuggingFace overrides
        hf_overrides: dict[str, Any] = self._prepare_hf_overrides()

        # Initialize vLLM engine. Let the CLI boundary render a traceback once.
        logger.info("Loading vLLM engine...")
        llm_kwargs = build_vllm_llm_kwargs(self.args)
        llm_kwargs["hf_overrides"] = hf_overrides
        llm: LLM = LLM(**llm_kwargs)
        logger.info("✅ vLLM engine loaded successfully")

        logger.info("✅ vLLM engine initialization completed")
        return llm

    def _build_sampling_params(self, seed: int) -> SamplingParams:
        """Build generation parameters for one deterministic sample stream."""
        temperature = (
            self.args.temperature if getattr(self.args, "do_sample", True) else 0.0
        )
        return SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
            seed=seed,
            skip_special_tokens=getattr(self.args, "skip_special_tokens", True),
        )

    def _sampling_params_for_items(
        self, items: Sequence[dict[str, Any]]
    ) -> list[SamplingParams]:
        """Return independent, resume-stable sampling parameters per item."""
        return [
            self._build_sampling_params(get_request_seed(item))
            for item in items
        ]

    def _prepare_hf_overrides(self) -> dict[str, Any]:
        """Prepare HuggingFace model overrides from arguments.

        This method processes the configuration arguments and creates a dictionary
        of overrides that will be passed to the HuggingFace model loading process.

        Returns:
            Dictionary of overrides for HuggingFace model loading.
        """
        hf_overrides: dict[str, Any] = {}

        # Use the parsed rope_scaling_dict instead of the raw string
        if hasattr(self.args, "rope_scaling_dict") and self.args.rope_scaling_dict:
            hf_overrides["rope_scaling"] = self.args.rope_scaling_dict

        return hf_overrides

    def convert_to_messages_format(
        self, item: dict[str, Any]
    ) -> list[dict[str, str]] | None:
        """Convert an input record to the vLLM chat messages format.

        This method handles the conversion of input records to the chat message format
        required by vLLM, including:
        - Field key resolution with fallbacks
        - Input validation and sanitization
        - Chat template validation to prevent double-application
        - System prompt integration

        Expected item keys:
            - Prefer `self.args.input_key`; fallback to 'prompt'.
            - Prefer `self.args.label_key`; fallback to 'answer'.

        Args:
            item: Input record dictionary containing prompt and label data.

        Returns:
            The messages list if conversion succeeds, otherwise None.

        Raises:
            ValueError: If required fields are missing, invalid, or chat template is already applied.
        """
        input_key: str = self.args.input_key

        # Only input_key is required for inference; prompt is the canonical fallback.
        prompt: Any = item.get(input_key) or item.get("prompt")
        if prompt is None:
            logger.warning(
                f"Missing required key '{input_key}' (or 'prompt') in item: {list(item.keys())}"
            )
            return None

        # Validate required fields
        prompt_str: str = str(prompt).strip()
        if not prompt_str:
            logger.warning("Empty prompt field in item")
            return None

        # Check if chat template is already applied
        if is_chat_template_applied(prompt_str):
            logger.warning(
                "Chat template appears to be already applied to the query. "
                "Please use the raw prompt, as vLLM will apply the Hugging Face "
                "chat template automatically."
            )
            raise ValueError(
                "Your query has been applied with chat_template, please use the raw prompt, "
                "because the vLLM will apply the Hugging Face chat template automatically!"
            )

        # Build messages list
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt_str})

        logger.debug(f"Converted to messages format: {len(messages)} messages")
        return messages

    def _write_response_results(
        self, original_items: Sequence[dict[str, Any]], responses: Sequence[str]
    ) -> None:
        """Persist non-empty generated responses."""
        if len(original_items) != len(responses):
            raise ValueError("original_items and responses must have equal length")
        valid_pairs: list[tuple[dict[str, Any], str]] = []
        failures: list[dict[str, Any]] = []
        for item, response in zip(original_items, responses, strict=True):
            if response and response.strip():
                valid_pairs.append((item, response))
            else:
                failures.append(
                    _sample_failure(item, "inference", "empty model response")
                )
        self._handle_sample_failures(failures)
        if not valid_pairs:
            return
        with self._file_lock:
            try:
                with open(self.args.output_file, "a", encoding="utf-8") as f:
                    for original_item, model_response in valid_pairs:
                        result = dict(original_item)
                        result.pop("_request_seed", None)
                        result[self.args.response_key] = [model_response]
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
            except Exception as e:
                raise OSError(f"Failed to write batch results: {e}") from e

    def _extract_model_response(self, output: RequestOutput) -> str:
        """Extract text response from vLLM output object.

        This method safely extracts the generated text from vLLM's RequestOutput
        object, handling various edge cases and potential errors.

        Args:
            output: vLLM RequestOutput object.

        Returns:
            Extracted text response, empty string if extraction fails.
        """
        if output is None:
            return ""

        try:
            # vLLM chat returns RequestOutput objects with `.outputs`
            # and each contains `.text`.
            if output.outputs and len(output.outputs) > 0:
                return output.outputs[0].text
            return ""
        except (AttributeError, IndexError) as e:
            logger.warning(f"Failed to extract response from output: {e}")
            return ""

    def load_data(self) -> list[dict[str, Any]]:
        """Load and expand the dataset, handling resume functionality per prompt.

        This method orchestrates the complete data loading process including:
        - Raw data loading from input file
        - Resume functionality by checking completed samples
        - Data expansion based on remaining samples needed
        - Comprehensive validation and error handling

        Returns:
            Expanded dataset where each record appears as many times as its
            remaining required generations.

        Raises:
            FileNotFoundError: If the input file does not exist.
            json.JSONDecodeError: If an input line is not valid JSON.
            ValueError: If the dataset is empty or invalid.
        """
        logger.info(f"Loading data from: {self.args.input_file}")

        # Load raw data
        raw_data: list[dict[str, Any]] = load_jsonl(self.args.input_file)
        logger.info(f"Loaded {len(raw_data)} items from input file")

        # Check for completed samples. Track the explicit completed index set
        # (not just a count) so a partially-failed sample in the middle of a
        # run is regenerated instead of duplicating the highest contiguous count.
        resume_state = load_resume_state(
            self.args.output_file,
            self.args.input_key,
            self.args.response_key,
            repair_truncated_last_line=getattr(self.args, "repair_resume", False),
        )

        if resume_state.completed_count > 0:
            logger.info(
                "Found %d completed samples from previous run",
                resume_state.completed_count,
            )

        expanded_data = expand_data_with_resume(
            raw_data,
            resume_state,
            self.args.input_key,
            self.args.n_samples,
            base_seed=self.args.seed,
        )

        if not expanded_data:
            logger.warning("No data to process after expansion")

        logger.info(f"Total remaining samples to process: {len(expanded_data)}")
        return expanded_data

    def process_and_write_batch(self, batch_data: Sequence[dict[str, Any]]) -> None:
        """Process a single batch of data and write results to file.

        This method handles the complete processing of a batch including:
        - Conversion of items to messages format with validation
        - Filtering out invalid items safely
        - Running vLLM chat inference
        - Persisting outputs for valid items
        - Comprehensive error handling and logging

        Args:
            batch_data: Batch of input items to process.

        Raises:
            RuntimeError: If the vLLM engine is not initialized or processing fails.
        """
        if not batch_data:
            logger.warning("Empty batch data provided")
            return

        if self.llm is None:
            raise RuntimeError(
                "vLLM engine is not initialized. Call setup_vllm_engine() first."
            )

        valid_items: list[dict[str, Any]] = []
        valid_messages: list[list[dict[str, str]]] = []
        invalid_items: list[dict[str, Any]] = []
        for item in batch_data:
            try:
                messages = self.convert_to_messages_format(item)
            except ValueError as exc:
                invalid_items.append(_sample_failure(item, "input_validation", exc))
                continue
            if messages is None:
                invalid_items.append(
                    _sample_failure(item, "input_validation", "invalid or empty prompt")
                )
                continue
            valid_items.append(item)
            valid_messages.append(messages)
        self._handle_sample_failures(invalid_items)
        if not valid_messages:
            return

        outputs: list[RequestOutput] = self.llm.chat(
            valid_messages,
            self._sampling_params_for_items(valid_items),
            use_tqdm=False,
            chat_template_kwargs={
                "enable_thinking": getattr(self.args, "enable_thinking", False)
            },
        )
        responses = [self._extract_model_response(output) for output in outputs]
        self._write_response_results(valid_items, responses)

    def _handle_sample_failures(self, failures: list[dict[str, Any]]) -> None:
        """Apply the configured strict or auditing policy to sample failures."""
        if not failures:
            return
        if self.args.fail_fast:
            first = failures[0]
            raise ValueError(
                f"Sample {first['item']} failed {first['error_category']}: "
                f"{first['error']}"
            )
        save_failed_items(self.args.output_file, failures)

    def run(self) -> None:
        """Run the main inference process end-to-end.

        This method orchestrates the complete inference process including:
        - File path validation
        - Data loading with resume functionality
        - Output directory creation
        - vLLM engine initialization
        - Batch processing with progress tracking
        - Comprehensive error handling

        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If output file path is not provided.
            RuntimeError: If inference process fails.
        """
        # Validate file paths
        if not self.args.input_file or not Path(self.args.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {self.args.input_file}")
        if not self.args.output_file:
            raise ValueError("Output file path is required")

        # Load data (including resume functionality)
        eval_dataset: list[dict[str, Any]] = self.load_data()
        if not eval_dataset:
            logger.info("All samples have already been processed, skipping inference")
            return

        # Create output directory if it doesn't exist
        output_path: Path = Path(self.args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"⏳ Starting to process {len(eval_dataset)} entries")

        # Initialize vLLM engine
        self.llm = self.setup_vllm_engine()

        # Process data in batches
        self._process_batches(eval_dataset)

        logger.info(
            f"✨ Final data processing completed. Results saved to {self.args.output_file}"
        )

    def _process_batches(self, eval_dataset: list[dict[str, Any]]) -> None:
        """Process the evaluation dataset in batches.

        This method handles the batch processing of the evaluation dataset with
        progress tracking and comprehensive error handling.

        Args:
            eval_dataset: Dataset to process.
        """
        total_batches: int = (
            len(eval_dataset) + self.args.batch_size - 1
        ) // self.args.batch_size
        logger.info(
            f"Processing {total_batches} batches with batch size {self.args.batch_size}"
        )

        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
            failures = process_batches_with_policy(
                eval_dataset,
                self.args.batch_size,
                self.process_and_write_batch,
                fail_fast=getattr(self.args, "fail_fast", True),
                on_batch_complete=lambda: pbar.update(1),
            )
        if failures:
            save_failed_items(self.args.output_file, failures)
            logger.warning(
                "Continued after %d failed batch(es); details saved next to output",
                len(failures),
            )


def main(args: OfflineInferArguments) -> None:
    """Main function to run the vLLM offline inference process.

    This function serves as the main entry point for the offline inference
    process, handling initialization and execution with comprehensive error handling.

    Args:
        args: Configuration arguments for the inference process

    Backend, schema, and persistence errors propagate to the CLI boundary.
    """
    runner = OfflineInferenceRunner(args)
    runner.run()


if __name__ == "__main__":
    """Command-line interface for vLLM offline inference."""
    try:
        # Parse command line arguments
        parser = HfArgumentParser(OfflineInferArguments)  # type: ignore[arg-type]
        (eval_args,) = parser.parse_args_into_dataclasses()

        # Log configuration
        logger.info(
            "Initializing OfflineInferArguments with parsed command line arguments..."
        )
        logger.info("\n--- Parsed Arguments ---")
        logger.info(
            json.dumps(
                redact_config_for_logging(asdict(eval_args)), indent=2, default=str
            )
        )

        # Run main inference process
        main(eval_args)

    except ImportError as e:
        logger.error(f"❌ A required library is missing: {e}. Please install it.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"❌ An unrecoverable error occurred during execution: {e}")
        sys.exit(1)
