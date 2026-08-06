"""
Verifier Offline Inference Module

This module provides functionality for running offline inference using vLLM engine
for the Verifier evaluation system. It supports batch processing, resume
functionality, and robust error handling.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from llmeval.inference.common import (
    iter_resume_records,
    load_jsonl,
    missing_sample_indices,
    process_batches_with_policy,
    redact_config_for_logging,
    sample_seed_for_item,
    save_failed_items,
    to_public_result_schema,
)
from llmeval.utils.config import VerifierInferArguments
from llmeval.utils.log import init_logger
from llmeval.utils.verifier_prompts import VERIFY_PROMPT_FACTORY

# Initialize logger
logger = init_logger("compass_verifier_infer", logging.INFO)


# Precompiled extraction patterns (compiled once at import, not per call).
_ANSWER_TAG_RE: re.Pattern[str] = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_END_RE: re.Pattern[str] = re.compile(r"</think\s*>", re.IGNORECASE)
_BOXED_CONTENT_RE: re.Pattern[str] = re.compile(
    r"\\boxed\s*\{([^}]*)\}", re.DOTALL | re.IGNORECASE
)
_LETTER_IN_BOX_RE: re.Pattern[str] = re.compile(r"([A-D])", re.IGNORECASE)
_PAREN_LETTER_RE: re.Pattern[str] = re.compile(r"\(([A-D])\)", re.IGNORECASE)
_STANDALONE_LETTER_RE: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z])([A-D])(?![A-Za-z])", re.IGNORECASE
)
_VERIFIER_RESUME_KEY = "llmeval_verifier_id"


def _last_n_strs(text: str, n: int) -> str:
    """Return the last n whitespace-separated tokens as a string."""
    tokens = text.split()
    return " ".join(tokens[-n:]) if tokens else ""


def extract_tagged_answer(response_string: str, fallback_tokens: int = 200) -> str:
    """
    Extract content from <answer> tags in model response.

    This function searches for <answer> tags in the response string and extracts
    the content within them. If no tags are found, it attempts sensible fallbacks.

    Args:
        response_string: Complete string containing <answer> tags.
        fallback_tokens: Number of trailing tokens to return as a last fallback.

    Returns:
        Extracted content from <answer> tags, or a fallback string if no tags found.

    Example:
        >>> extract_tagged_answer("Some text <answer>42</answer> more text")
        "42"
        >>> extract_tagged_answer("No tags here")
        "<last 200 tokens of the string, if any>"
    """
    if not response_string or not isinstance(response_string, str):
        return ""

    # (.*?) 是一个非贪婪捕获组，用于匹配并提取标签内的所有内容。
    # re.DOTALL 标志确保 . 也能匹配换行符，以防 answer 内容有多行。
    match = _ANSWER_TAG_RE.search(response_string)
    # 如果找到匹配项，返回第一个捕获组（括号内的内容），并去除首尾空格
    if match:
        content = match.group(1).strip()
        if content:
            return content

    # Fallback 1: content after </think>
    match = _THINK_END_RE.search(response_string)
    if match:
        tail = response_string[match.end() :].strip()
        if tail:
            return tail

    # Fallback 2: last N tokens
    last_n_str = _last_n_strs(response_string, fallback_tokens).strip()
    return last_n_str if last_n_str else ""


def process_judgment(judgment_str: str) -> str:
    """Extract judgment letter — delegates to :func:`process_judgment_cursor`."""
    return process_judgment_cursor(judgment_str)


def process_judgment_cursor(judgment_str: str) -> str:
    """
    Extract and normalize the final judgment from model output.

    Uses a five-strategy cascade (first match wins):
    1. Direct single-letter match (``A``, ``B``, ``C``, ``D``).
    2. Last ``\\boxed{...}`` content, then the last A-D letter within it.
    3. "Final Judgment:" section — parenthesized letter within that section.
    4. Any parenthesized letter ``(A)`` through ``(D)`` anywhere in the text.
    5. Any standalone A-D letter (word-boundary delimited) as a last resort.
    All strategies are case-insensitive.  Returns uppercase or ``""``.

    Examples:
        >>> process_judgment("\\boxed{A}")
        'A'
        >>> process_judgment("some text \\boxed{  c }")
        'C'
        >>> process_judgment("Final: (D)")
        'D'
        >>> process_judgment("noise only")
        ''
    """
    if not isinstance(judgment_str, str) or not judgment_str:
        return ""

    s = judgment_str.strip()

    # Strategy 1: direct single-letter match (no further processing needed).
    if s in {"A", "B", "C", "D"}:
        return s

    # Strategy 2: extract the last boxed content, then find a valid letter within it.
    boxed_contents = _BOXED_CONTENT_RE.findall(s)
    if boxed_contents:
        candidates = _LETTER_IN_BOX_RE.findall(boxed_contents[-1])
        if candidates:
            return candidates[-1].upper()

    # Strategy 3: "Final Judgment:" section extraction (common in English verifier prompts).
    if "Final Judgment:" in s:
        final_section = s.split("Final Judgment:")[-1]
        paren_matches = _PAREN_LETTER_RE.findall(final_section)
        if paren_matches:
            return paren_matches[-1].upper()

    # Strategy 4: explicit parenthesized letter like (A), (b), etc.
    paren_matches = _PAREN_LETTER_RE.findall(s)
    if paren_matches:
        return paren_matches[-1].upper()

    # Strategy 5: any standalone A-D letter (avoid letters embedded in words).
    all_matches = _STANDALONE_LETTER_RE.findall(s)
    if all_matches:
        return all_matches[-1].upper()

    return ""


# Map verifier prompt types to their judgment extraction functions.
# When adding a new prompt type to VERIFY_PROMPT_FACTORY, add its entry here too.
JUDGMENT_EXTRACTOR: dict[str, Callable[[str], str]] = {
    "compassverify_prompt": process_judgment,
    "compassverify_prompt_zh": process_judgment,
    "compassverify_cot_prompt": process_judgment,
    "compassverify_cot_prompt_zh": process_judgment,
}

# Fail fast at import time if a prompt type was added to VERIFY_PROMPT_FACTORY
# but no matching judgment extractor was registered here.
_missing_extractors = set(VERIFY_PROMPT_FACTORY) - set(JUDGMENT_EXTRACTOR)
if _missing_extractors:
    raise RuntimeError(
        f"VERIFY_PROMPT_FACTORY keys have no entry in JUDGMENT_EXTRACTOR: "
        f"{sorted(_missing_extractors)}. "
        "Add the missing key(s) to JUDGMENT_EXTRACTOR in "
        "llmeval/inference/verifier.py."
    )


class VerifierOfflineInferenceRunner:
    """
    Main class for handling offline inference with vLLM engine for Verifier.

    This class provides a comprehensive solution for running Verifier inference
    with support for batch processing, resume functionality, and robust error handling.

    Attributes:
        args: Configuration arguments for the inference process.
        _file_lock: Thread lock for safe file writing operations.
        llm: vLLM engine instance for model inference.
        tokenizer: HuggingFace tokenizer instance for text processing.
        sampling_params: Sampling parameters for generation control.
        verifier_prompt: String template used to format verifier prompts.
    """

    def __init__(self, args: VerifierInferArguments) -> None:
        """
        Initialize the Verifier inference runner.

        Args:
            args: Configuration arguments containing model settings, file paths, etc.

        Raises:
            ValueError: If required arguments are invalid.
        """
        self.args: VerifierInferArguments = args
        self._file_lock: threading.Lock = threading.Lock()
        self.llm: LLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.sampling_params: SamplingParams | None = None
        self.verifier_prompt: str | None = VERIFY_PROMPT_FACTORY.get(
            args.verifier_prompt_type
        )

    def setup_vllm_engine(self) -> tuple[LLM, AutoTokenizer, SamplingParams]:
        """
        Initialize the vLLM engine, tokenizer, and sampling parameters.

        This method sets up the complete inference pipeline including model loading,
        tokenizer initialization, and sampling parameter configuration.

        Returns:
            A tuple containing:
                - LLM instance for inference
                - AutoTokenizer for text processing
                - SamplingParams for generation control

        Raises:
            RuntimeError: If engine initialization fails.
            ImportError: If required dependencies are missing.
        """
        logger.info("=" * 60)
        logger.info("🚀 Initializing Verifier vLLM Engine")
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
        hf_overrides = self._prepare_hf_overrides()

        # Initialize engine components. The CLI boundary owns traceback logging.
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=self.args.trust_remote_code,
            cache_dir=self.args.cache_dir,
        )
        logger.info("✅ Tokenizer loaded successfully")

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
        llm = LLM(**llm_kwargs)
        logger.info("✅ vLLM engine loaded successfully")

        # Configure sampling parameters
        sampling_params = self._build_sampling_params(self.args.seed)

        logger.info("✅ Verifier vLLM engine initialization completed")
        return llm, tokenizer, sampling_params

    def _build_sampling_params(self, seed: int) -> SamplingParams:
        """Build generation parameters for one deterministic verifier sample."""
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
        self, items: list[dict[str, Any]]
    ) -> list[SamplingParams]:
        """Return independent, resume-stable sampling parameters per item."""
        return [
            self._build_sampling_params(sample_seed_for_item(self.args.seed, item))
            for item in items
        ]

    def _prepare_hf_overrides(self) -> dict[str, Any]:
        """Prepare HuggingFace model overrides from arguments.

        Returns:
            Dictionary of overrides for HuggingFace model loading.
        """
        hf_overrides: dict[str, Any] = {}

        # Use the parsed rope_scaling_dict instead of the raw string
        if hasattr(self.args, "rope_scaling_dict") and self.args.rope_scaling_dict:
            hf_overrides["rope_scaling"] = self.args.rope_scaling_dict

        return hf_overrides

    def _effective_keys(self) -> tuple[str, str, str]:
        """Resolve the effective input/label/response keys with fallbacks."""
        input_key = self.args.input_key
        label_key = self.args.label_key
        response_key = self.args.response_key
        return input_key, label_key, response_key

    def _resume_id(self, item: dict[str, Any]) -> str:
        """Resolve the prepared dataset ID for verifier resume.

        New benchmark files carry ``doc_id`` from the preparation stage. The
        hash fallback is retained only for legacy verifier inputs that predate
        the dataset ID field.
        """
        document_id = item.get("doc_id")
        if document_id:
            return str(document_id)
        existing = item.get(_VERIFIER_RESUME_KEY)
        if existing:
            return str(existing)

        input_key, label_key, response_key = self._effective_keys()
        payload = {
            "prompt": item.get(input_key) or item.get("prompt"),
            "gold": item.get(label_key),
            "response": item.get(response_key),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()
        return f"verifier:{digest}"

    def convert_to_compass_verifier_format(self, item: dict[str, Any]) -> str | None:
        """
        Convert input data item to Verifier prompt format.

        This method extracts the required fields from the input data and formats
        them according to the Verifier template.

        Args:
            item: Input data item containing question, gold_answer, and llm_response.

        Returns:
            Formatted prompt string for Verifier or None if conversion fails.

        Raises:
            KeyError: If required keys are missing from the input item.
            ValueError: If required fields are empty or invalid.
        """
        # Determine field keys with fallbacks
        input_key, label_key, response_key = self._effective_keys()

        # Check for required keys
        required_keys = [input_key, label_key, response_key]
        missing_keys = [key for key in required_keys if key not in item]

        if missing_keys:
            logger.warning(
                f"Missing required keys {missing_keys} in item: {list(item.keys())}"
            )
            return None

        # Extract required fields
        prompt = item.get(input_key)
        ground_truth = item.get(label_key)
        llm_response_raw = item.get(response_key)

        # Handle different response formats
        llm_response = self._extract_llm_response(llm_response_raw)
        if llm_response is None:
            return None

        # Validate required fields. Explicit None/empty-string checks (rather
        # than truthiness) so a legitimate gold answer of 0/0.0/False or a
        # numeric prompt is not silently dropped.
        def _field_empty(value: Any) -> bool:
            return value is None or (isinstance(value, str) and not value.strip())

        if (
            _field_empty(prompt)
            or _field_empty(ground_truth)
            or _field_empty(llm_response)
        ):
            logger.warning(
                f"Empty required field in item - question: {bool(prompt)}, "
                f"ground_truth: {bool(ground_truth)}, llm_response: {bool(llm_response)}"
            )
            return None

        # Extract answer from response if it contains <answer> tags
        llm_response = extract_tagged_answer(llm_response)

        # Ensure we have a verifier prompt template
        if not self.verifier_prompt:
            logger.error("Verifier prompt template is not configured.")
            return None

        # Format the prompt using Verifier template
        try:
            formatted_prompt = self.verifier_prompt.format(
                question=prompt, gold_answer=ground_truth, llm_response=llm_response
            )
            return formatted_prompt
        except Exception as e:
            logger.error(f"Error formatting Verifier prompt: {e}")
            return None

    def _extract_llm_response(self, llm_response_raw: Any) -> str | None:
        """
        Extract LLM response from various input formats.

        Args:
            llm_response_raw: Raw LLM response in various formats.

        Returns:
            Extracted response string or None if extraction fails.
        """
        if isinstance(llm_response_raw, list) and llm_response_raw:
            first = llm_response_raw[0]
            if isinstance(first, str):
                return first
            logger.warning(f"Invalid response element type: {type(first)}")
            return None
        elif isinstance(llm_response_raw, str):
            return llm_response_raw
        elif llm_response_raw is None:
            logger.warning("Invalid response format: None")
            return None
        else:
            logger.warning(f"Invalid response format: {type(llm_response_raw)}")
            return None

    def _write_response_results(
        self, original_items: list[dict[str, Any]], responses: list[str]
    ) -> None:
        """Write generated responses after current judgment extraction."""
        if len(original_items) != len(responses):
            raise ValueError("original_items and responses must have equal length")
        with self._file_lock:
            try:
                with open(self.args.output_file, "a", encoding="utf-8") as f:
                    for idx, (original_item, model_response) in enumerate(
                        zip(original_items, responses, strict=True)
                    ):
                        if model_response and model_response.strip():
                            result = self._prepare_result_item(
                                original_item, model_response
                            )
                            f.write(
                                json.dumps(
                                    to_public_result_schema(result), ensure_ascii=False
                                )
                                + "\n"
                            )
                            f.flush()
                        else:
                            logger.warning(
                                "Empty response for item %d, skipping write", idx
                            )
            except Exception as e:
                raise OSError(f"Failed to write batch results: {e}") from e

    def _extract_model_response(self, output: RequestOutput) -> str:
        """Extract text response from vLLM output object.

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

    def _prepare_result_item(
        self, original_item: dict[str, Any], model_response: str
    ) -> dict[str, Any]:
        """
        Prepare result item for writing to output file.

        Args:
            original_item: Original data item.
            model_response: Model's response text.

        Returns:
            Processed result item ready for JSON serialization.
        """
        result = original_item.copy()
        result[_VERIFIER_RESUME_KEY] = self._resume_id(original_item)
        result["Verifier_response"] = model_response

        # Optionally strip original large fields to reduce output size
        if not self.args.keep_origin_data:
            input_key, _, response_key = self._effective_keys()
            # Clear original input/response fields to reduce output size.
            # Verifier_response and Verifier_judgment are preserved for auditability.
            if input_key in result:
                result[input_key] = ""
            if response_key in result:
                result[response_key] = ""

        # Extract judgment using the mapped extractor for the prompt type
        extractor = JUDGMENT_EXTRACTOR.get(self.args.verifier_prompt_type)
        if extractor is None:
            raise NotImplementedError(
                f"Unknown verifier_prompt_type: {self.args.verifier_prompt_type}"
            )
        result["Verifier_judgment"] = extractor(model_response)

        return result

    def get_completed_sample_indices(self) -> dict[str, set[int]]:
        """
        Count completed samples for resume functionality.

        This method scans the output file to determine how many samples have
        already been processed for each unique question, enabling resume
        functionality for interrupted runs.

        Returns:
            Mapping of verifier identity to completed sample indices.
        """
        completed_indices: dict[str, set[int]] = collections.defaultdict(set)

        if not os.path.exists(self.args.output_file):
            return {}

        if os.path.getsize(self.args.output_file) == 0:
            return {}

        for _, item in iter_resume_records(
            self.args.output_file,
            repair_truncated_last_line=getattr(self.args, "repair_resume", False),
        ):
            prompt_key = (
                item.get(_VERIFIER_RESUME_KEY)
                or item.get(self.args.input_key)
                or item.get("prompt")
            )

            # Inference completion is independent from whether the judgment
            # parser could classify the model response.
            if prompt_key is not None and item.get("Verifier_response"):
                key = str(prompt_key)
                raw_index = item.get("sample_index")
                if type(raw_index) is int and raw_index >= 0:
                    sample_index = raw_index
                else:
                    sample_index = 0
                    while sample_index in completed_indices[key]:
                        sample_index += 1
                completed_indices[key].add(sample_index)

        return completed_indices

    def load_data(self) -> list[dict[str, Any]]:
        """
        Load and prepare dataset with resume functionality.

        This method loads the input data, checks for previously completed samples,
        and expands the dataset according to the n_samples parameter while
        respecting the resume functionality.

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
        raw_data = load_jsonl(self.args.input_file)
        logger.info(f"Loaded {len(raw_data)} items from input file")

        # Check for completed samples
        completed_indices = self.get_completed_sample_indices()
        total_completed = sum(len(indices) for indices in completed_indices.values())

        if total_completed > 0:
            logger.info(f"Found {total_completed} completed samples from previous run")

        expanded_data: list[dict[str, Any]] = []
        skipped_items = 0
        seen_resume_ids: set[str] = set()
        for item in raw_data:
            if not isinstance(item, dict):
                skipped_items += 1
                continue
            resume_id = self._resume_id(item)
            if resume_id in seen_resume_ids:
                raise ValueError(
                    f"Duplicate verifier resume id {resume_id!r}; provide a unique "
                    "doc_id for every input row"
                )
            seen_resume_ids.add(resume_id)
            used_indices = completed_indices.get(resume_id, set())
            for sample_index in missing_sample_indices(
                self.args.n_samples, used_indices
            ):
                expanded_item = copy.deepcopy(item)
                expanded_item[_VERIFIER_RESUME_KEY] = resume_id
                expanded_item["sample_index"] = sample_index
                expanded_data.append(expanded_item)
        if skipped_items > 0:
            logger.warning(f"Skipped {skipped_items} non-dict item(s)")
        if not expanded_data:
            logger.warning("No data to process after expansion")

        logger.info(f"Total remaining samples to process: {len(expanded_data)}")
        return expanded_data

    def process_and_write_batch(self, batch_data: list[dict[str, Any]]) -> None:
        """
        Process a single batch of data and write results.

        This method handles the complete batch processing pipeline including
        data conversion, model inference, and result writing.

        Args:
            batch_data: List of data dictionaries for the current batch.

        Raises:
            RuntimeError: If batch processing fails.
        """
        if not batch_data:
            logger.warning("Empty batch data provided")
            return

        if self.llm is None or self.tokenizer is None or self.sampling_params is None:
            raise RuntimeError(
                "Engine is not initialized. Call setup_vllm_engine() before processing."
            )

        original_items = batch_data
        batch_prompts: list[str | None] = []

        # Convert data format and filter invalid items
        for item in batch_data:
            prompt = self.convert_to_compass_verifier_format(item)
            if prompt is not None:
                batch_prompts.append(prompt)
            else:
                logger.warning("Failed to convert item to Verifier format")
                batch_prompts.append("")

        # Filter out empty prompts and corresponding original items
        valid_prompts, valid_original_items = self._filter_valid_prompts(
            batch_prompts, original_items
        )

        if not valid_prompts:
            logger.warning("No valid prompts in this batch, skipping")
            return

        # Convert prompts to messages format for vLLM
        batch_messages: list[str] = []
        tokenizer = cast(Any, self.tokenizer)
        for prompt in valid_prompts:
            messages = [{"role": "user", "content": prompt}]
            model_inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            batch_messages.append(model_inputs)

        outputs: list[RequestOutput] = self.llm.generate(
            batch_messages,
            self._sampling_params_for_items(valid_original_items),
            use_tqdm=False,  # Avoid progress bar conflicts
        )
        responses = [self._extract_model_response(output) for output in outputs]

        self._write_response_results(valid_original_items, responses)
        logger.debug(
            f"Successfully processed batch of {len(valid_original_items)} items"
        )

    def _filter_valid_prompts(
        self, batch_prompts: Iterable[str | None], original_items: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Filter out empty prompts and corresponding original items.

        Args:
            batch_prompts: Iterable of prompt strings (some may be None/empty).
            original_items: List of original data items.

        Returns:
            Tuple of (valid_prompts, valid_original_items).
        """
        valid_prompts: list[str] = []
        valid_original_items: list[dict[str, Any]] = []

        for i, prompt in enumerate(batch_prompts):
            if prompt:  # Only include non-empty prompts
                valid_prompts.append(prompt)
                valid_original_items.append(original_items[i])

        return valid_prompts, valid_original_items

    def run(self) -> None:
        """
        Run the main inference process.

        This method orchestrates the complete inference pipeline including
        data loading, engine initialization, batch processing, and result writing.

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
        eval_dataset = self.load_data()
        if not eval_dataset:
            logger.info("All samples have already been processed, skipping inference")
            return

        # Create output directory if it doesn't exist
        output_path = Path(self.args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"⏳ Starting to process {len(eval_dataset)} entries")

        # Initialize vLLM engine
        self.llm, self.tokenizer, self.sampling_params = self.setup_vllm_engine()

        # Process data in batches
        self._process_batches(eval_dataset)

        logger.info(
            f"✨ Final data processing completed. Results saved to {self.args.output_file}"
        )

    def _process_batches(self, eval_dataset: list[dict[str, Any]]) -> None:
        """Process the evaluation dataset in batches.

        Args:
            eval_dataset: Dataset to process.
        """
        total_batches = (
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


def main(args: VerifierInferArguments) -> None:
    """
    Main function to run the Verifier vLLM inference process.

    Args:
        args: Configuration arguments for the inference process.

    Backend, schema, and persistence errors propagate to the CLI boundary.
    """
    runner = VerifierOfflineInferenceRunner(args)
    runner.run()


if __name__ == "__main__":
    """Command-line interface for Verifier offline inference."""
    try:
        # Parse command line arguments
        parser = HfArgumentParser(VerifierInferArguments)  # type: ignore[arg-type]
        (eval_args,) = parser.parse_args_into_dataclasses()

        # Log configuration
        logger.info(
            "Initializing Verifier VerifierInferArguments with parsed command line arguments..."
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
