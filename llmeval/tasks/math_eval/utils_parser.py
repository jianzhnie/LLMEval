"""
This module provides a utility function to parse ground truth data
from various datasets, extracting the chain-of-thought (CoT) and
the final answer.
"""
from typing import Any, Dict, List, Optional, Tuple


def parse_ground_truth(example: Dict[str, Any],
                       data_name: str,
                       label_key: str = None) -> Tuple[Optional[str], str]:
    """
    Parses the ground truth data from an example dictionary based on the dataset name.

    This function extracts the chain-of-thought (CoT) and the final answer. The CoT is
    optional and will be returned as None if not present.

    Args:
        example (Dict[str, Any]): A dictionary containing the raw data for a single example.
        data_name (str): The name of the dataset (e.g., 'gsm8k', 'olympiadbench').
        label_key (str, optional): Custom key for accessing the answer field. If provided,
                                  this will override the default dataset-specific key.

    Returns:
        Tuple[Optional[str], str]: A tuple containing two strings:
                                   - The chain-of-thought (CoT), or None if not available.
                                   - The final answer.

    Raises:
        ValueError: If the required fields for the specified dataset are missing or
                    in an unexpected format.
        NotImplementedError: If the provided `data_name` is not supported and no label_key is provided.
    """
    # Get the name of the field that contains the answer
    if label_key not in example:
        raise ValueError(
            f"The ground truth key '{label_key}' not found in example.")
    ground_truth = str(example[label_key]).strip()

    # Handle cases where the required answer field is missing
    if ground_truth is None:
        raise ValueError(
            f"Required field '{label_key}' not found for dataset '{data_name}'."
        )

    # --- Dataset-specific parsing logic ---
    if data_name == 'gsm8k':
        # GSM8K's answer field contains both CoT and the final answer separated by '####'
        if '####' not in ground_truth:
            raise ValueError(
                f"GSM8K example is missing the '####' separator: {example}")

        parts = ground_truth.split('####', 1)  # Use maxsplit for robustness
        gt_cot = parts[0].strip()
        ground_truth = parts[1].strip()
        return gt_cot, ground_truth

    elif data_name == 'olympiadbench':
        # OlympiadBench's final answer is in a list and may contain '$'
        if not isinstance(ground_truth, List) or not ground_truth:
            raise ValueError(
                f"OlympiadBench 'final_answer' is missing or has an invalid format: {example}"
            )

        # Access the first element of the list and strip any leading/trailing '$'
        ground_truth = str(ground_truth[0]).strip('$').strip()
        return None, ground_truth

    # For all other datasets, the answer is a simple string in the 'answer' field
    else:
        # These datasets do not have a separate CoT field
        return None, ground_truth
