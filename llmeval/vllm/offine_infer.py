import json
import logging
import os
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.config import EvaluationArguments
from llmeval.utils.logger import init_logger

# Initialize logger
logger = init_logger('vllm_infer', logging.INFO)


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from a JSONL file.

    Args:
        dataset_path: The path to the JSONL dataset file.

    Returns:
        A list of dictionaries, where each dictionary is a data entry.
    """
    data_list = []
    logger.info(f'🔄 Loading dataset from: {dataset_path}')
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in tqdm(enumerate(f, 1),
                                       desc='Loading data',
                                       unit=' lines'):
                line = line.strip()
                if not line:
                    continue
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        f'⚠️ Unable to parse JSON on line {line_num}. Skipping. Snippet: "{line[:50]}..."'
                    )
    except FileNotFoundError:
        logger.error(f'❌ Dataset file not found at: {dataset_path}')
        raise
    except Exception as e:
        logger.error(
            f'❌ An unexpected error occurred while loading the dataset: {e}')
        raise

    logger.info(f'✅ Successfully loaded {len(data_list)} entries.')
    return data_list


def setup_vllm_engine(args: EvaluationArguments) -> Tuple[LLM, SamplingParams]:
    """
    Initialize the vLLM engine and sampling parameters based on arguments.

    Args:
        args: An instance of EvaluationArguments.

    Returns:
        A tuple containing the LLM instance and SamplingParams instance.
    """
    # Print engine initialization information
    logger.info('=' * 50)
    logger.info('🚀 Initializing vLLM Engine')
    logger.info(f'Model: {args.model_name_or_path}')
    logger.info(f'Max Model Length: {args.max_model_len}')
    logger.info(f'RoPE Scaling: {args.rope_scaling}')
    logger.info(f'Tensor Parallel Size: {args.tensor_parallel_size}')
    logger.info(f'GPU Memory Utilization: {args.gpu_memory_utilization}')
    logger.info(f'Batch Size: {args.batch_size}')
    logger.info('=' * 50)

    # Set environment variable for vLLM sampler
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'

    # Prepare hf_overrides from arguments
    hf_overrides = {
        'rope_scaling': args.rope_scaling,
        'max_model_len': args.max_model_len,
    }

    try:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            max_num_seqs=args.max_num_seqs,
            hf_overrides=hf_overrides,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f'❌ Failed to initialize vLLM engine: {e}')
        raise

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    logger.info('✅ vLLM engine initialization completed.')
    return llm, sampling_params


def process_and_write_batch(
    llm: LLM,
    batch_data: List[Dict],
    sampling_params: SamplingParams,
    results_file: str,
    start_index: int,
):
    """
    Processes a single batch of data and writes the results to a file.

    Args:
        llm: The vLLM instance.
        batch_data: A list of data dictionaries for the current batch.
        sampling_params: The sampling parameters for generation.
        results_file: The path to the output results file.
        start_index: The global starting index of this batch.
    """
    batch_messages = []
    # Use a list to store original items in case of data issues
    original_items = []

    for item in batch_data:
        original_items.append(item)
        if 'messages' in item and isinstance(item['messages'], list):
            batch_messages.append(item['messages'])
        else:
            logger.warning(
                "⚠️ Invalid data item found (missing 'messages' or not a list). Skipping this entry."
            )
            batch_messages.append(None)  # Add a placeholder for a failed item

    # Filter out invalid entries before passing to LLM
    valid_batch_messages = [msg for msg in batch_messages if msg is not None]
    if not valid_batch_messages:
        logger.warning('No valid messages in this batch. Skipping.')
        return

    try:
        outputs = llm.chat(valid_batch_messages,
                           sampling_params,
                           use_tqdm=True)

        with open(results_file, 'a', encoding='utf-8') as f:
            valid_output_index = 0
            for i, original_item in enumerate(original_items):
                # Check if this item was valid and has an output
                if batch_messages[i] is None:
                    # Log and write an error entry for the invalid item
                    error_result = {
                        'id': start_index + i,
                        'meta': original_item,
                        'messages': original_item.get('messages', []),
                        'error':
                        'Invalid data format (missing "messages" field).'
                    }
                    f.write(
                        json.dumps(error_result, ensure_ascii=False) + '\n')
                    continue

                # Process the valid item and its corresponding output
                output = outputs[valid_output_index]
                ai_response = output.outputs[0].text
                complete_messages = valid_batch_messages[
                    valid_output_index].copy()
                complete_messages.append({
                    'role': 'assistant',
                    'content': ai_response
                })

                result = {
                    'id': start_index + i,
                    'meta': original_item,
                    'messages': complete_messages
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                valid_output_index += 1
            f.flush()

    except Exception as e:
        logger.error(
            f'❌ Error during vLLM processing for this batch: {e}. Writing error entries.'
        )
        # If the whole batch fails, write error entries for all items in the batch
        with open(results_file, 'a', encoding='utf-8') as f:
            for i, original_item in enumerate(original_items):
                error_result = {
                    'id': start_index + i,
                    'meta': original_item,
                    'messages': original_item.get('messages', []),
                    'error': f'Batch processing error: {str(e)}'
                }
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
            f.flush()


def main(args: EvaluationArguments):
    """
    Main function to run the vLLM inference and evaluation process.
    """
    # Load and prepare dataset
    eval_dataset = load_dataset(
        os.path.join(args.dataset_dir, args.dataset_name))
    original_len = len(eval_dataset)
    if args.n_sampling > 1:
        eval_dataset = eval_dataset * args.n_sampling
        logger.info(
            f'🔁 Dataset repeated {args.n_sampling} times, expanded from {original_len} entries to {len(eval_dataset)}.'
        )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'inference_results.jsonl')

    # Resume from previous progress if file exists
    processed_count = 0
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
        logger.info(
            f'📦 Found existing results file. Resuming from {processed_count} entries.'
        )

    remaining_data = eval_dataset[processed_count:]
    if not remaining_data:
        logger.info('✅ All data processing completed. Exiting.')
        return

    logger.info(
        f'⏳ Starting to process remaining {len(remaining_data)} entries.')

    # Setup vLLM engine and sampling parameters
    llm, sampling_params = setup_vllm_engine(args)

    # Process data in batches
    for i in tqdm(range(0, len(remaining_data), args.batch_size),
                  desc='Processing batches'):
        batch_start = i
        batch_end = min(i + args.batch_size, len(remaining_data))
        batch = remaining_data[batch_start:batch_end]

        # Calculate global start index
        global_start_index = processed_count + batch_start

        process_and_write_batch(llm=llm,
                                batch_data=batch,
                                sampling_params=sampling_params,
                                results_file=results_file,
                                start_index=global_start_index)

    logger.info(
        f'✨ Final data processing completed. Results saved to {results_file}.')


if __name__ == '__main__':
    try:
        parser = HfArgumentParser(EvaluationArguments)
        eval_args, = parser.parse_args_into_dataclasses()

        logger.info(
            'Initializing EvaluationArguments with parsed command line arguments...'
        )
        logger.info('\n--- Parsed Arguments ---')
        # Use dataclasses.asdict to print arguments cleanly
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
