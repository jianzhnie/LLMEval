import json
import logging
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams

from llmeval.utils.config import EvaluationArguments
from llmeval.utils.logger import init_logger

logger = init_logger('vllm_infer', logging.INFO, None)


def load_dataset(dataset_path: str) -> list:
    """Load dataset"""
    data_list = []
    logger.info(f'Loading dataset: {dataset_path}')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading data'):
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f'Unable to parse JSON line: {line}')

    logger.info(f'Loaded {len(data_list)} entries')
    return data_list


def main(args: EvaluationArguments):
    # Load dataset
    eval_dataset = load_dataset(args.dataset_dir, args.dataset_name)
    # Repeat dataset n times
    original_len = len(eval_dataset)
    eval_dataset = eval_dataset * args.n_sampling
    logger.info(
        f'Dataset repeated {args.n_sampling} times, expanded from {original_len} entries to {len(eval_dataset)} entries'
    )

    # Ensure output and cache directories exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse rope_scaling from string to dict if needed
    rope_scaling = args.get('rope_scaling')
    if isinstance(rope_scaling, str):
        try:
            rope_scaling = json.loads(rope_scaling)
        except json.JSONDecodeError as e:
            logger.error(f'Error parsing rope_scaling JSON: {e}')
            rope_scaling = {
                'rope_type': 'yarn',
                'factor': 2.5,
                'original_max_position_embeddings': 32768
            }

    # Print engine information
    logger.info('=' * 50)
    logger.info('Initializing vLLM Engine')
    logger.info(f"Model: {args.get('model_name_or_path', 'unknown')}")
    logger.info(f"Max Model Length: {args.get('max_model_len', 'default')}")
    logger.info(f'RoPE Scaling: {rope_scaling}')
    logger.info(f"Tensor Parallel Size: {args.get('tensor_parallel_size', 1)}")
    logger.info(
        f"GPU Memory Utilization: {args.get('gpu_memory_utilization', 0.9)}")
    logger.info(f'Batch Size: {args.batch_size}')
    logger.info('=' * 50)

    # Create LLM instance using parsed arguments
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'

    # Prepare hf_overrides with rope_scaling and max_model_len
    hf_overrides = {
        'rope_scaling': rope_scaling,
        'max_model_len': args.get('max_model_len', 81920)
    }

    llm = LLM(
        model=args.get('model_name_or_path', './KlearReasoner-8B'),
        tensor_parallel_size=args.get('tensor_parallel_size', 8),
        gpu_memory_utilization=args.get('gpu_memory_utilization', 0.9),
        enable_prefix_caching=args.get('enable_prefix_caching', False),
        max_num_seqs=args.get('max_num_seqs', 128),
        hf_overrides=hf_overrides,
        enforce_eager=args.get('enforce_eager', False),
        seed=args.get('seed', 0),
    )

    logger.info('vLLM engine initialization completed')

    # Create sampling parameters
    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     repetition_penalty=1.05)
    print(sampling_params)

    # Process data and write results in real-time
    process_data_batch(llm, eval_dataset, sampling_params)


def process_data_batch(llm, eval_dataset, sampling_params,
                       args: EvaluationArguments):
    """Process data using batch processing and write results in real-time"""
    results_file = os.path.join(args.output_dir, 'inference_results.jsonl')
    # Track number of processed items
    processed_count = 0

    # Check for partial results
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
        logger.info(
            f'Found existing results file, {processed_count} entries already processed'
        )

    # Continue from last processed position
    remaining_data = eval_dataset[processed_count:]
    if not remaining_data:
        logger.info('All data processing completed')
        return

    logger.info(f'Starting to process remaining {len(remaining_data)} entries')

    # Process in batches
    with open(results_file, 'a', encoding='utf-8') as f:
        # Split data into batches
        for batch_start in range(0, len(remaining_data), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(remaining_data))
            batch = remaining_data[batch_start:batch_end]

            try:
                # Prepare batch messages list and original data
                batch_messages = []
                original_items = []

                for item in batch:
                    # Save original data item
                    original_items.append(item)

                    # Directly use messages field from data
                    if 'messages' in item:
                        batch_messages.append(item['messages'])
                    else:
                        logger.warning(
                            f"Data item missing 'messages' field: {item}")
                        batch_messages.append([{
                            'role': 'user',
                            'content': 'Invalid data'
                        }])

                # Batch call chat API
                logger.debug(
                    f'Starting batch processing {batch_start} to {batch_end-1}'
                )
                outputs = llm.chat(batch_messages,
                                   sampling_params,
                                   use_tqdm=True)

                # Process each output and save results
                for i, output in enumerate(outputs):
                    item_idx = processed_count + batch_start + i
                    original_item = original_items[i]

                    # Get AI response
                    ai_response = output.outputs[0].text

                    # Build complete messages list including model's response
                    complete_messages = batch_messages[i].copy()
                    complete_messages.append({
                        'role': 'assistant',
                        'content': ai_response
                    })

                    # Build metadata
                    meta = original_item

                    # Build complete result
                    result = {
                        'id': item_idx,
                        'meta': meta,
                        'messages': complete_messages
                    }

                    # Write result and flush immediately
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                # Update progress
                logger.info(
                    f'Processed {processed_count + batch_end}/{len(eval_dataset)} entries'
                )

            except Exception as e:
                logger.error(
                    f'Error processing batch {batch_start}-{batch_end-1}: {str(e)}'
                )
                # Log error but continue with next batch
                for i in range(len(batch)):
                    item_idx = processed_count + batch_start + i
                    original_item = batch[i]

                    # Build metadata
                    meta = original_item

                    error_result = {
                        'id': item_idx,
                        'meta': meta,
                        'messages': original_item.get('messages', []),
                        'error': str(e)
                    }
                    f.write(
                        json.dumps(error_result, ensure_ascii=False) + '\n')
                    f.flush()

    logger.info(f'Data processing completed, results saved to {results_file}')


if __name__ == '__main__':
    try:
        from transformers import HfArgumentParser
    except ImportError:
        raise ImportError(
            'Please install the transformers library to use HfArgumentParser: pip install transformers'
        )

    # Create an HfArgumentParser instance for the EvaluationArguments class.
    # It automatically reads all fields and metadata from the dataclass.
    parser = HfArgumentParser(EvaluationArguments)

    # Parse the command-line arguments and get an instance of EvaluationArguments.
    # The return value is a tuple, we only need the first element.
    eval_args, = parser.parse_args_into_dataclasses()
    logger.info(
        'Initializing EvaluationArguments with parsed command line arguments...'
    )
    logger.info('\n--- Parsed Arguments ---')
    logger.info(eval_args)
    main(eval_args)
