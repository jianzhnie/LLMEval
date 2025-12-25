import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Final

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Defaults (used only if CLI values are not provided)
AMTHINKING_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant. To answer the user's question, you first think "
    'about the reasoning process and then provide the user with the answer. '
    'The reasoning process and answer are enclosed within <think> </think> and '
    '<answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think> <answer> answer here </answer>.')


def safe_get_text(field: Any) -> str:
    """Safely extract text from various field types."""
    if isinstance(field, str):
        return field
    elif isinstance(field, list) and len(field) > 0:
        return str(field[0])
    return ''


def compute_token_lengths(example: Dict[str, Any], tokenizer: AutoTokenizer,
                          system_prompt: str) -> Dict[str, Any]:
    """Compute token lengths for prompt and generation text."""
    prompt_text = safe_get_text(example.get('prompt', ''))
    prompt_with_system = system_prompt + prompt_text
    gen_text = safe_get_text(example.get('gen', ''))

    # Tokenize once and cache results
    prompt_tokens = tokenizer(prompt_with_system,
                              truncation=False,
                              padding=False)
    gen_tokens = tokenizer(gen_text, truncation=False, padding=False)

    prompt_length = len(prompt_tokens['input_ids'])
    gen_length = len(gen_tokens['input_ids'])

    # Check for boxed{} in the last 1000 characters
    tail_text = gen_text[-1000:]
    has_boxed = 'boxed{' in tail_text

    return {
        'prompt_length': prompt_length,
        'gen_length': gen_length,
        'max_token_length': prompt_length + gen_length,
        'has_boxed': has_boxed
    }


def should_keep_example(example: Dict[str, Any],
                        max_token_length: int) -> bool:
    """Determine if example should be kept based on filtering criteria."""
    # Keep examples that have boxed{} and are within token limit
    exceeded = example['max_token_length'] > max_token_length
    return example['has_boxed'] or exceeded


def load_and_validate_args() -> argparse.Namespace:
    """Load and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description=
        "Filter dataset by 'gen' field token length and remove the field.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_path',
        required=True,
        help="Path to a local .jsonl or glob pattern (e.g., '/data/*.jsonl')")
    parser.add_argument(
        '--output_file',
        required=True,
        help="Output JSONL file path (e.g., './filtered.jsonl')")
    parser.add_argument(
        '--tokenizer_name_or_path',
        required=True,
        help="Tokenizer name or path (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument('--max_token_length',
                        type=int,
                        default=128,
                        help="Maximum allowed token length for 'gen' field")
    parser.add_argument('--num_proc',
                        type=int,
                        default=16,
                        help='Number of processes for filtering')

    return parser.parse_args()


def validate_paths(input_path: str, output_file: str) -> None:
    """Validate input and output paths."""
    input_files = list(Path().glob(input_path))
    if not input_files:
        raise ValueError(f'No files found matching pattern: {input_path}')

    output_path = Path(output_file)
    if output_path.exists() and not output_path.is_file():
        raise ValueError(
            f'Output path exists but is not a file: {output_file}')


def process_dataset(dataset: Dataset, tokenizer: AutoTokenizer,
                    max_token_length: int, num_proc: int) -> Dataset:
    """Process dataset with token length computation and filtering."""
    logger.info('Computing token lengths...')

    # Add token length information
    dataset = dataset.map(lambda x: compute_token_lengths(
        x, tokenizer, AMTHINKING_SYSTEM_PROMPT),
                          num_proc=num_proc,
                          desc='Computing token lengths')

    initial_count = len(dataset)
    logger.info(f'Initial dataset size: {initial_count}')

    # Apply filtering
    dataset = dataset.filter(
        lambda x: should_keep_example(x, max_token_length),
        num_proc=num_proc,
        desc='Filtering by criteria')

    filtered_count = len(dataset)
    logger.info(f'Filtered dataset size: {filtered_count}')
    logger.info(
        f'Removed {initial_count - filtered_count} examples '
        f'({(initial_count - filtered_count) / initial_count * 100:.1f}%)')

    # Remove 'gen' field if exists
    if 'gen' in dataset.column_names:
        dataset = dataset.remove_columns(['gen'])
        logger.info("Removed 'gen' field")

    return dataset


def main():
    """Main processing function."""
    try:
        # Load and validate arguments
        args = load_and_validate_args()

        # Validate paths
        validate_paths(args.input_path, args.output_file)

        # Setup output directory
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        logger.info(f'Loading tokenizer from: {args.tokenizer_name_or_path}')
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name_or_path)
        except Exception as e:
            logger.error(f'Failed to load tokenizer: {e}')
            raise

        # Load dataset
        logger.info(f'Loading dataset from: {args.input_path}')
        try:
            dataset = load_dataset('json',
                                   data_files=str(args.input_path),
                                   split='train')
        except Exception as e:
            logger.error(f'Failed to load dataset: {e}')
            raise

        logger.info(f'Loaded {len(dataset)} examples')

        # Process dataset
        processed_dataset = process_dataset(dataset, tokenizer,
                                            args.max_token_length,
                                            args.num_proc)

        # Save results
        logger.info(f'Saving filtered dataset to: {output_file}')
        processed_dataset.to_json(str(output_file),
                                  lines=True,
                                  force_ascii=False)
        logger.info('Processing completed successfully')

    except Exception as e:
        logger.error(f'Processing failed: {e}')
        raise


if __name__ == '__main__':
    main()
