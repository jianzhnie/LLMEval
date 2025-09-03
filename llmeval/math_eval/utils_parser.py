"""
This module provides a utility function to parse ground truth data
from various datasets, extracting the chain-of-thought (CoT) and
the final answer.
"""
from typing import Any, Dict, List, Optional, Tuple

# A constant mapping dataset names to their answer field names.
# This makes the code more scalable and easier to update.
DATASET_ANSWER_FIELDS: Dict[str, str] = {
    'gsm8k': 'answer',
    'olympiadbench': 'final_answer',
    'aime24': 'answer',
    'amc23': 'answer',
    'cmath': 'answer',
    'imo2024': 'answer',
    'aimo12': 'answer',
    'cnmo24': 'answer',
    'aime25': 'answer',
    'math500': 'answer',
}


def parse_ground_truth(example: Dict[str, Any],
                       data_name: str) -> Tuple[Optional[str], str]:
    """
    Parses the ground truth data from an example dictionary based on the dataset name.

    This function extracts the chain-of-thought (CoT) and the final answer. The CoT is
    optional and will be returned as None if not present.

    Args:
        example (Dict[str, Any]): A dictionary containing the raw data for a single example.
        data_name (str): The name of the dataset (e.g., 'gsm8k', 'olympiadbench').

    Returns:
        Tuple[Optional[str], str]: A tuple containing two strings:
                                   - The chain-of-thought (CoT), or None if not available.
                                   - The final answer.

    Raises:
        ValueError: If the required fields for the specified dataset are missing or
                    in an unexpected format.
        NotImplementedError: If the provided `data_name` is not supported.
    """
    # Check if the dataset is supported by looking up the constant dictionary
    if data_name not in DATASET_ANSWER_FIELDS:
        raise NotImplementedError(f'Unsupported dataset: `{data_name}`')

    # Get the name of the field that contains the answer
    answer_field_name = DATASET_ANSWER_FIELDS[data_name]
    answer_field = example.get(answer_field_name)

    # Handle cases where the required answer field is missing
    if answer_field is None:
        raise ValueError(
            f"Required field '{answer_field_name}' not found for dataset '{data_name}'."
        )

    # --- Dataset-specific parsing logic ---
    if data_name == 'gsm8k':
        # GSM8K's answer field contains both CoT and the final answer separated by '####'
        if '####' not in answer_field:
            raise ValueError(
                f"GSM8K example is missing the '####' separator: {example}")

        parts = answer_field.split('####', 1)  # Use maxsplit for robustness
        gt_cot = parts[0].strip()
        gt_ans = parts[1].strip()
        return gt_cot, gt_ans

    elif data_name == 'olympiadbench':
        # OlympiadBench's final answer is in a list and may contain '$'
        if not isinstance(answer_field, List) or not answer_field:
            raise ValueError(
                f"OlympiadBench 'final_answer' is missing or has an invalid format: {example}"
            )

        # Access the first element of the list and strip any leading/trailing '$'
        gt_ans = str(answer_field[0]).strip('$').strip()
        return None, gt_ans

    # For all other datasets, the answer is a simple string in the 'answer' field
    else:
        # These datasets do not have a separate CoT field
        gt_ans = str(answer_field).strip()
        return None, gt_ans
