from datasets import load_dataset

def filter_data(example):
    """
    Filter function to exclude data based on specific criteria:
    1. level should be >= 4
    2. Exclude entries where contest contains 'AIME' and year contains '2024'
    """
    # Filter out if level < 4
    if example["level"] < 4:
        return False
    
    # Filter out if contest contains 'AIME' and year contains '2024'
    if 'AIME' in example["contest"] and '2024' in example["year"]:
        return False
    
    return True

def main() -> None:

    data_path = "/home/jianzhnie/llmtuner/llm/LLMEval/data/HARP/HARP.jsonl"
    output_path = "/home/jianzhnie/llmtuner/llm/LLMEval/data/HARP/HARP_filter.jsonl"
    # Load the dataset from JSONL file
    dataset = load_dataset('json', data_files=data_path, split="train")

    # Apply filter using the filter function
    filtered_dataset = dataset.filter(filter_data)

    # Save the filtered dataset back to JSONL
    filtered_dataset.to_json(output_path, lines=True, force_ascii=False)


if __name__ == '__main__':
    main()