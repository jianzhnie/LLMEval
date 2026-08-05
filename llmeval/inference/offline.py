"""Offline vLLM inference runner.

This module provides a small, documented wrapper around vLLM to:
- Load a line-delimited JSON dataset
- Resume generation per unique prompt up to a requested sample count
- Convert records into vLLM chat message format
- Run batched chat inference
- Persist unified results incrementally for robustness

The output schema appends generations into a `gen` list for each input record.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from llmeval.cache import ContentAddressedCache
from llmeval.inference.common import (
    count_completed_samples,
    count_completed_samples_by_identity,
    expand_data_with_resume,
    load_jsonl,
)
from llmeval.tasks.provenance import (
    get_git_hash,
    hash_evaluation_inputs,
    hash_json,
)
from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.log import init_logger
from llmeval.utils.prompts import SYSTEM_PROMPT_FACTORY, is_chat_template_applied
from llmeval.utils.reproducibility import seed_everything, seed_provenance

# Initialize logger
logger = init_logger("offline_vllm_infer", logging.INFO)


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
        sampling_params: Sampling parameters for text generation
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
        self.reproducibility = seed_provenance(seed_everything(args.seed))
        self._file_lock: threading.Lock = threading.Lock()
        self.llm: LLM | None = None
        self.sampling_params: SamplingParams | None = None
        content_cache_dir = getattr(args, "content_cache_dir", "")
        self.cache: ContentAddressedCache | None = (
            ContentAddressedCache(
                content_cache_dir,
                "inference",
                force_recompute=getattr(args, "force_recompute", False),
                read_only=getattr(args, "read_only_cache", False),
                rank=getattr(args, "cache_rank", None),
            )
            if content_cache_dir
            else None
        )
        self._git_hash = get_git_hash()
        self.system_prompt: str | None = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type
        )

    def setup_vllm_engine(self) -> tuple[LLM, SamplingParams]:
        """Initialize the vLLM engine and sampling parameters.

        This method handles the complete setup of the vLLM engine including:
        - Model loading with specified parameters
        - HuggingFace overrides configuration
        - Sampling parameters setup
        - Comprehensive error handling and logging

        Returns:
            A tuple containing the LLM instance and SamplingParams instance.

        Raises:
            RuntimeError: If engine initialization fails.
        """
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

        try:
            # Initialize vLLM engine
            logger.info("Loading vLLM engine...")
            llm_kwargs: dict[str, Any] = {
                "model": self.args.model_name_or_path,
                "tensor_parallel_size": self.args.tensor_parallel_size,
                "pipeline_parallel_size": self.args.pipeline_parallel_size,
                "gpu_memory_utilization": self.args.gpu_memory_utilization,
                "enable_chunked_prefill": self.args.enable_chunked_prefill,
                "enable_prefix_caching": self.args.enable_prefix_caching,
                "enforce_eager": self.args.enforce_eager,
                "max_num_seqs": self.args.max_num_seqs,
                "max_model_len": self.args.max_model_len,
                "hf_overrides": hf_overrides,
                "seed": self.args.seed,
                "trust_remote_code": self.args.trust_remote_code,
                "dtype": self.args.dtype,
                "device": self.args.device,
            }
            if self.args.max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = self.args.max_num_batched_tokens
            if self.args.quantization is not None:
                llm_kwargs["quantization"] = self.args.quantization
            model_revision = getattr(self.args, "model_revision", None)
            if model_revision is not None:
                llm_kwargs["revision"] = model_revision
            llm: LLM = LLM(**llm_kwargs)
            logger.info("✅ vLLM engine loaded successfully")

        except Exception as e:
            # Include traceback for easier debugging
            logger.exception(f"❌ Failed to initialize vLLM engine: {e}")
            raise RuntimeError(f"Engine initialization failed: {e}") from e

        # Configure sampling parameters
        sampling_params: SamplingParams = SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
            seed=self.args.seed,
        )

        logger.info("✅ vLLM engine initialization completed")
        return llm, sampling_params

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

        if self.args.max_model_len:
            hf_overrides["max_model_len"] = self.args.max_model_len

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

    def _write_batch_results(
        self, original_items: Sequence[dict[str, Any]], outputs: Sequence[RequestOutput]
    ) -> None:
        """Write batch results to file in unified schema with a 'gen' field.

        This method handles the writing of batch results to the output file with:
        - Thread-safe file operations using locks
        - Unified schema with 'gen' field for generated responses
        - Incremental writing for robustness
        - Comprehensive error handling

        The output schema appends a generated string into `gen` list for each item.

        Args:
            original_items: Original input items that were processed.
            outputs: vLLM output objects containing generated text.

        Raises:
            IOError: If file writing fails.
        """
        responses = [self._extract_model_response(output) for output in outputs]
        self._write_response_results(original_items, responses)

    def _write_response_results(
        self, original_items: Sequence[dict[str, Any]], responses: Sequence[str]
    ) -> None:
        """Persist non-empty responses, whether generated or read from cache."""
        if len(original_items) != len(responses):
            raise ValueError("original_items and responses must have equal length")
        with self._file_lock:
            try:
                with open(self.args.output_file, "a", encoding="utf-8") as f:
                    for idx, (original_item, model_response) in enumerate(
                        zip(original_items, responses, strict=True)
                    ):
                        if model_response and model_response.strip():
                            result: dict[str, Any] = original_item.copy()
                            gen_list: list[str] = list(
                                result.get(self.args.response_key, [])
                            )
                            gen_list.append(model_response)
                            result[self.args.response_key] = gen_list
                            provenance = getattr(self, "reproducibility", None)
                            if provenance is not None:
                                result["inference_provenance"] = provenance
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                        else:
                            logger.warning(
                                "Empty response for item %d, skipping write", idx
                            )
            except Exception as e:
                logger.error(f"Error writing batch results: {e}")
                raise OSError(f"Failed to write batch results: {e}") from e

    def _cache_key(
        self, item: dict[str, Any], messages: list[dict[str, str]]
    ) -> str | None:
        """Build a generation key from stable inputs and all output parameters."""
        cache = getattr(self, "cache", None)
        if cache is None:
            return None
        payload = {
            "backend": "offline_chat",
            "model_name": self.args.model_name_or_path,
            "model_revision": getattr(self.args, "model_revision", None),
            "task_name": item.get("task", getattr(self.args, "task", None)),
            "task_version": item.get("task_version"),
            "dataset_hash": hash_evaluation_inputs(
                [item], getattr(self.args, "response_key", "gen")
            ),
            "prompt_hash": hash_json(messages),
            "generation_params": {
                "max_tokens": self.args.max_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "repetition_penalty": self.args.repetition_penalty,
            },
            "sampling_seed": self.args.seed,
            "postprocess_version": "offline_chat_v1",
            "git_commit": getattr(self, "_git_hash", None),
            "doc_id": item.get("doc_id"),
            "sample_index": item.get("_llmeval_sample_index"),
        }
        return cache.key(payload)

    def _log_cache_stats(self) -> None:
        """Log cache counters at the end of a run when caching is enabled."""
        cache = getattr(self, "cache", None)
        if cache is not None:
            logger.info(
                "Offline inference cache statistics: %s", cache.stats().to_dict()
            )

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

        # Check for completed samples
        completed_counts = cast(
            dict[object, int],
            count_completed_samples_by_identity(
                self.args.output_file,
                self.args.input_key,
                self.args.response_key,
            ),
        )
        legacy_counts = count_completed_samples(
            self.args.output_file,
            self.args.input_key,
            self.args.response_key,
            legacy_only=True,
        )
        if legacy_counts:
            # Preserve prompt-based resume for legacy records while allowing
            # stable-ID records in the same file to remain independently keyed.
            completed_counts.update(legacy_counts)
        total_completed: int = sum(completed_counts.values())

        if total_completed > 0:
            logger.info(f"Found {total_completed} completed samples from previous run")

        # Expand data according to n_samples and resume functionality
        expanded_data: list[dict[str, Any]] = expand_data_with_resume(
            raw_data,
            completed_counts,
            self.args.input_key,
            self.args.n_samples,
            stable_ids=True,
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

        if self.llm is None or self.sampling_params is None:
            raise RuntimeError(
                "vLLM engine is not initialized. Call setup_vllm_engine() first."
            )

        # Keep only items that successfully convert to message format
        valid_items, valid_messages = self._filter_valid_items(batch_data)

        if not valid_messages:
            logger.warning(
                "All items in this batch failed message conversion; skipping."
            )
            return

        cache = getattr(self, "cache", None)
        responses: list[str] = [""] * len(valid_items)
        missing_indices: list[int] = []
        missing_messages: list[list[dict[str, str]]] = []
        cache_keys: dict[int, str] = {}
        for index, (item, messages) in enumerate(
            zip(valid_items, valid_messages, strict=True)
        ):
            key = self._cache_key(item, messages)
            if cache is not None and key is not None:
                cache_keys[index] = key
                cached = cache.get(key)
                cached_response = cached.get("response") if cached else None
                if isinstance(cached_response, str) and cached_response.strip():
                    responses[index] = cached_response
                    continue
            missing_indices.append(index)
            missing_messages.append(messages)

        try:
            if missing_messages:
                logger.debug("Processing %d uncached prompts", len(missing_messages))
                outputs: list[RequestOutput] = self.llm.chat(
                    missing_messages, self.sampling_params, use_tqdm=False
                )
                for index, output in zip(missing_indices, outputs, strict=False):
                    response = self._extract_model_response(output)
                    responses[index] = response
                    key = cache_keys.get(index)
                    if cache is not None and key is not None and response.strip():
                        cache.set(key, {"response": response})
            else:
                logger.debug("All prompts in batch were served from cache")
            self._write_response_results(valid_items, responses)
        except Exception as e:
            logger.error(f"❌ Error during vLLM processing for this batch: {e}")
            raise RuntimeError(f"Batch processing failed: {e}") from e

    def _filter_valid_items(
        self, batch_data: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, str]]]]:
        """Filter batch data to keep only valid items that can be converted to messages.

        This method processes each item in the batch and attempts to convert it
        to the messages format, keeping only those that succeed.

        Args:
            batch_data: Input batch data to filter.

        Returns:
            Tuple of (valid_items, valid_messages).
        """
        valid_items: list[dict[str, Any]] = []
        valid_messages: list[list[dict[str, str]]] = []

        for item in batch_data:
            try:
                messages: list[dict[str, str]] | None = self.convert_to_messages_format(
                    item
                )
                if messages is not None:
                    valid_items.append(item)
                    valid_messages.append(messages)
            except ValueError as e:
                # Log the error but continue processing other items
                logger.warning(f"Failed to convert item to messages format: {e}")
                continue

        return valid_items, valid_messages

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

        try:
            # Load data (including resume functionality)
            eval_dataset: list[dict[str, Any]] = self.load_data()
            if not eval_dataset:
                logger.info(
                    "All samples have already been processed, skipping inference"
                )
                return

            # Create output directory if it doesn't exist
            output_path: Path = Path(self.args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"⏳ Starting to process {len(eval_dataset)} entries")

            # Initialize vLLM engine
            self.llm, self.sampling_params = self.setup_vllm_engine()

            # Process data in batches
            self._process_batches(eval_dataset)
            self._log_cache_stats()

            logger.info(
                f"✨ Final data processing completed. Results saved to {self.args.output_file}"
            )

        except Exception as e:
            logger.critical(f"❌ Fatal error during inference: {e}")
            raise

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
            for i in range(0, len(eval_dataset), self.args.batch_size):
                batch: list[dict[str, Any]] = eval_dataset[i : i + self.args.batch_size]
                self.process_and_write_batch(batch)
                pbar.update(1)


def main(args: OfflineInferArguments) -> None:
    """Main function to run the vLLM offline inference process.

    This function serves as the main entry point for the offline inference
    process, handling initialization and execution with comprehensive error handling.

    Args:
        args: Configuration arguments for the inference process

    Raises:
        RuntimeError: If inference process fails
    """
    try:
        runner = OfflineInferenceRunner(args)
        runner.run()
    except Exception as e:
        logger.critical(f"❌ Inference process failed: {e}")
        raise RuntimeError(f"Inference failed: {e}") from e


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
        logger.info(json.dumps(asdict(eval_args), indent=2, default=str))

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
