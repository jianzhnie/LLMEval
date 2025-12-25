import argparse
import logging
from pathlib import Path
from typing import Final

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults (used only if CLI values are not provided)
amthinking_system_prompt: Final[str] = (
    "You are a helpful assistant. To answer the user's question, you first think "
    'about the reasoning process and then provide the user with the answer. '
    'The reasoning process and answer are enclosed within <think> </think> and '
    '<answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think> <answer> answer here </answer>.')


def main():
    parser = argparse.ArgumentParser(
        description=
        "Filter dataset by 'gen' field token length and remove the field.")
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
    args = parser.parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f'Loading tokenizer from: {args.tokenizer_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Load dataset
    logger.info(f'Loading dataset from: {args.input_path}')
    dataset = load_dataset('json',
                           data_files=str(args.input_path),
                           split='train')
    logger.info(f'Loaded {len(dataset)} examples')

    # Safely extract gen text
    def get_text(field):
        if isinstance(field, str):
            return field
        elif isinstance(field, list) and len(field) > 0:
            return str(field[0])
        else:
            return ''  # fallback to empty string

    # Step 1: Add token_length column
    def add_token_length(example):
        prompt_text = get_text(example.get('prompt', ''))
        prompt_text = amthinking_system_prompt + prompt_text
        gen_text = get_text(example.get('gen', ''))
        prompt_tokens = tokenizer(prompt_text,
                                  truncation=False,
                                  padding=False,
                                  return_tensors=None)
        gen_tokens = tokenizer(gen_text,
                               truncation=False,
                               padding=False,
                               return_tensors=None)
        prompt_length = len(prompt_tokens['input_ids'])
        gen_length = len(gen_tokens['input_ids'])
        max_token_length = prompt_length + gen_length
        example['prompt_length'] = prompt_length
        example['gen_length'] = gen_length
        example['max_token_length'] = max_token_length
        return example

    # Pure filter function (no side effects)
    def filter_example(example):
        gen_text = get_text(example.get('gen', ''))
        tail_text = gen_text[:-1000]
        have_box = 'boxed{}' in tail_text
        execeded = example['max_token_length'] >= args.max_token_length
        if not have_box or execeded:
            return True
        return False

    # Add token length
    dataset = dataset.map(add_token_length,
                          num_proc=args.num_proc,
                          desc='Computing token lengths')
    # Apply filter
    dataset = dataset.filter(filter_example,
                             num_proc=args.num_proc,
                             desc='Filtering by max token length')
    logger.info(f'Filtered down to {len(dataset)} examples')

    # Remove 'gen' field if exists
    if 'gen' in dataset.column_names:
        dataset = dataset.remove_columns(['gen'])
        logger.info("Removed 'gen' field")

    # Save to JSONL file (not directory!)
    dataset.to_json(str(output_file), lines=True, force_ascii=False)
    logger.info(f'Saved filtered dataset to: {output_file}')


if __name__ == '__main__':
    main()
