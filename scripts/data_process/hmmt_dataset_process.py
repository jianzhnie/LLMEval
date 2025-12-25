from typing import Final

from datasets import load_dataset

qwen_math_cot_prompt: Final[str] = (
    'Please reason step by step, and put your final answer within \\boxed{}.')


def format_example(example):
    question = example['problem']
    answer = example['answer']
    prompt = f'{question}\n{qwen_math_cot_prompt}'
    return {
        'prompt': prompt,
        'answer': answer,
        'problem_type': example['problem_type']
    }


def main() -> None:

    data_path = '/home/jianzhnie/llmtuner/hfhub/datasets/MathArena/hmmt_feb_2025'
    output_path = '/home/jianzhnie/llmtuner/llm/LLMEval/data/hmmt_feb_2025.jsonl'
    # Load the dataset from JSONL file
    dataset = load_dataset(data_path, split='train')
    # Apply format using the format function
    formatted_dataset = dataset.map(format_example,
                                    remove_columns=dataset.column_names)
    # Save the formatted dataset back to JSONL
    formatted_dataset.to_json(output_path, lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
