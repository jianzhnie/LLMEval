import collections
import copy
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.compass_template import CompassVerifier_PROMPT
from llmeval.utils.config import OfflineInferArguments
from llmeval.utils.logger import init_logger

# Initialize logger
logger = init_logger('compass_verifier_infer', logging.INFO)


class CompassVerifierOfflineInferenceRunner:
    """Main class to handle offline inference with vLLM engine for CompassVerifier."""

    def __init__(self, args: OfflineInferArguments):
        self.args: OfflineInferArguments = args
        self._file_lock = threading.Lock()
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.sampling_params: Optional[SamplingParams] = None

    def setup_vllm_engine(self) -> Tuple[LLM, AutoTokenizer, SamplingParams]:
        """
        Initialize the vLLM engine, tokenizer, and sampling parameters with improved error handling.

        Returns:
            A tuple containing the LLM instance, tokenizer, and SamplingParams instance.
        """
        # Print engine initialization information
        logger.info('=' * 50)
        logger.info('🚀 Initializing CompassVerifier vLLM Engine')
        logger.info(f'Model: {self.args.model_name_or_path}')
        logger.info(f'Max Model Length: {self.args.max_model_len}')
        logger.info(f'Max tokens: {self.args.max_tokens}')
        logger.info(f'RoPE Scaling: {self.args.rope_scaling}')
        logger.info(f'Tensor Parallel Size: {self.args.tensor_parallel_size}')
        logger.info(
            f'Pipeline Parallel Size: {self.args.pipeline_parallel_size}')
        logger.info(
            f'GPU Memory Utilization: {self.args.gpu_memory_utilization}')
        logger.info(f'Batch Size: {self.args.batch_size}')
        logger.info('=' * 50)

        # Prepare hf_overrides from arguments
        hf_overrides = {}
        if self.args.rope_scaling:
            hf_overrides['rope_scaling'] = self.args.rope_scaling
        if self.args.max_model_len:
            hf_overrides['max_model_len'] = self.args.max_model_len

        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path,
                trust_remote_code=self.args.trust_remote_code,
                cache_dir=self.args.cache_dir)

            # Initialize vLLM engine
            llm = LLM(
                model=self.args.model_name_or_path,
                tensor_parallel_size=self.args.tensor_parallel_size,
                pipeline_parallel_size=self.args.pipeline_parallel_size,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                enable_chunked_prefill=self.args.enable_chunked_prefill,
                enable_prefix_caching=self.args.enable_prefix_caching,
                enforce_eager=self.args.enforce_eager,
                max_num_seqs=self.args.max_num_seqs,
                max_model_len=self.args.max_model_len,
                hf_overrides=hf_overrides,
                seed=self.args.seed,
                trust_remote_code=self.args.trust_remote_code,
                dtype=self.args.dtype,
            )
        except Exception as e:
            logger.error(f'❌ Failed to initialize vLLM engine: {e}')
            raise

        sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            repetition_penalty=self.args.repetition_penalty,
        )

        logger.info('✅ CompassVerifier vLLM engine initialization completed.')
        return llm, tokenizer, sampling_params

    def convert_to_compass_verifier_format(
            self, item: Dict[str, Any]) -> Optional[str]:
        """
        Convert input data to CompassVerifier prompt format.

        Args:
            item: Input data item containing question, gold_answer, and llm_response

        Returns:
            Formatted prompt string for CompassVerifier or None if conversion fails
        """
        required_keys = ['question', 'gold_answer', 'llm_response']

        # Check if all required keys are present
        for key in required_keys:
            if key not in item:
                logger.warning(f'Missing required key "{key}" in item: {item}')
                return None

        # Extract the required fields
        question = item['question']
        gold_answer = item['gold_answer']
        llm_response = item['llm_response']

        if not question or not gold_answer or not llm_response:
            logger.warning(f'Empty required field in item: {item}')
            return None

        # Format the prompt using CompassVerifier template
        try:
            formatted_prompt = CompassVerifier_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                llm_response=llm_response)
            return formatted_prompt
        except Exception as e:
            logger.error(f'Error formatting CompassVerifier prompt: {e}')
            return None

    def _write_batch_results(self, original_items: List[Dict],
                             outputs: List) -> None:
        """Write batch results to file in unified schema with 'judgment' field."""
        with self._file_lock:
            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                for idx, original_item in enumerate(original_items):
                    output = outputs[idx]
                    model_response = output.outputs[
                        0].text if output.outputs else ''

                    # Only write if we got a valid response
                    if model_response and model_response.strip():
                        result = copy.deepcopy(original_item)
                        result['judgment'] = model_response.strip()
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                    else:
                        logger.warning(
                            f'Empty response for item {idx}, skipping write')

    def count_completed_samples(self) -> Dict[str, int]:
        """
        Counts the number of completed samples for each question in the output file.
        Uses question content to identify unique samples.
        """
        completed_counts = collections.defaultdict(int)
        if os.path.exists(self.args.output_file) and os.path.getsize(
                self.args.output_file) > 0:
            with open(self.args.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        question = item.get('question')
                        if question is not None and 'judgment' in item:
                            completed_counts[question] += 1
                    except json.JSONDecodeError:
                        continue
        return completed_counts

    def load_data(self) -> List[Dict[str, Any]]:
        """Loads and prepares the dataset, handling resume functionality per question."""
        try:
            with open(self.args.input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(
                f"Error loading input file '{self.args.input_file}': {e}")
            raise

        completed_counts = self.count_completed_samples()
        total_completed = sum(completed_counts.values())
        if total_completed > 0:
            logger.info(
                f'Found a total of {total_completed} samples from a previous run.'
            )

        expanded_data = []
        for item in data:
            question = item.get('question')
            if not question:
                logger.warning(f'No "question" found in item: {item}')
                continue
            completed = completed_counts.get(question, 0)
            remaining = max(0, self.args.n_samples - completed)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))

        logger.info(
            f'Total remaining samples to process: {len(expanded_data)}')
        return expanded_data

    def process_and_write_batch(
        self,
        batch_data: List[Dict],
    ) -> None:
        """
        Process a single batch of data and write results to file with improved error handling.

        Args:
            batch_data: A list of data dictionaries for the current batch.
        """
        original_items = copy.deepcopy(batch_data)
        batch_prompts = []

        # Convert data format and filter invalid items
        for item in batch_data:
            prompt = self.convert_to_compass_verifier_format(item)
            if prompt is not None:
                batch_prompts.append(prompt)
            else:
                logger.warning(
                    f'Failed to convert item to CompassVerifier format: {item}'
                )
                # Add empty string to maintain batch alignment
                batch_prompts.append('')

        # Filter out empty prompts and corresponding original items
        valid_prompts = []
        valid_original_items = []
        for i, prompt in enumerate(batch_prompts):
            if prompt:  # Only include non-empty prompts
                valid_prompts.append(prompt)
                valid_original_items.append(original_items[i])

        if not valid_prompts:
            logger.warning('No valid prompts in this batch, skipping')
            return

        try:
            # Convert prompts to messages format for vLLM
            batch_messages = []
            for prompt in valid_prompts:
                messages = [{'role': 'user', 'content': prompt}]
                model_inputs = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                batch_messages.append(model_inputs)

            # Use vLLM for inference
            outputs = self.llm.generate(
                batch_messages,
                self.sampling_params,
                use_tqdm=False  # Avoid progress bar conflicts
            )

            # Write results (unified schema)
            self._write_batch_results(valid_original_items, outputs)

        except Exception as e:
            logger.error(f'❌ Error during vLLM processing for this batch: {e}')

    def run(self) -> None:
        """Run the main inference process."""
        # Load data (including resume functionality)
        eval_dataset = self.load_data()
        if not eval_dataset:
            logger.info(
                'All samples have already been processed, skipping inference. Exiting.'
            )
            return

        logger.info(f'⏳ Starting to process {len(eval_dataset)} entries.')

        # Initialize vLLM engine
        self.llm, self.tokenizer, self.sampling_params = self.setup_vllm_engine(
        )

        # Process data in batches
        total_batches = (len(eval_dataset) + self.args.batch_size -
                         1) // self.args.batch_size
        with tqdm(total=total_batches, desc='Processing batches') as pbar:
            for i in range(0, len(eval_dataset), self.args.batch_size):
                batch = eval_dataset[i:i + self.args.batch_size]
                self.process_and_write_batch(batch)
                pbar.update(1)

        logger.info(
            f'✨ Final data processing completed. Results saved to {self.args.output_file}.'
        )


def main(args: OfflineInferArguments) -> None:
    """Main function to run the CompassVerifier vLLM inference and evaluation process."""
    runner = CompassVerifierOfflineInferenceRunner(args)
    runner.run()


if __name__ == '__main__':
    try:
        parser = HfArgumentParser(OfflineInferArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing CompassVerifier OfflineInferArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        import dataclasses
        logger.info(json.dumps(dataclasses.asdict(eval_args), indent=2))

        main(eval_args)

    except ImportError as e:
        logger.error(
            f'❌ A required library is missing: {e}. Please install it.')
        exit(1)
    except Exception as e:
        logger.critical(
            f'❌ An unrecoverable error occurred during execution: {e}')
        exit(1)
