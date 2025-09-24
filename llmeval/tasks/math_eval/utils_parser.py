"""
This module provides a utility function to parse ground truth data
from various datasets, extracting the chain-of-thought (CoT) and
the final answer.
"""
from typing import Any, Dict, Optional, Tuple


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
        TypeError: If the input parameters are of incorrect type.
    """
    # Input validation
    if not isinstance(example, dict):
        raise TypeError("'example' must be a dictionary")
    if not isinstance(data_name, str):
        raise TypeError("'data_name' must be a string")
    if label_key is not None and not isinstance(label_key, str):
        raise TypeError("'label_key' must be a string or None")

    # Get the ground truth value
    if label_key not in example:
        raise ValueError(
            f"The ground truth key '{label_key}' not found in example. Available keys: {list(example.keys())}"
        )

    ground_truth = example[label_key]

    # Handle empty or None values
    if ground_truth is None or (isinstance(ground_truth, str)
                                and not ground_truth.strip()):
        raise ValueError(f"Empty or None value found for key '{label_key}'")

    # --- Dataset-specific parsing logic ---
    data_name = data_name.lower()  # Case-insensitive comparison

    if data_name == 'gsm8k':
        return _parse_gsm8k(ground_truth)
    elif data_name == 'olympiadbench':
        return _parse_olympiadbench(ground_truth)
    else:
        # Generic parsing for other datasets
        return _parse_generic(ground_truth)


def _parse_gsm8k(ground_truth: str) -> Tuple[str, str]:
    """Helper function to parse GSM8K format answers."""
    ground_truth = str(ground_truth).strip()
    if '####' not in ground_truth:
        raise ValueError("GSM8K answer must contain '####' separator")

    parts = ground_truth.split('####', 1)
    gt_cot = parts[0].strip()
    final_answer = parts[1].strip()

    if not final_answer:
        raise ValueError('GSM8K final answer is empty')

    return gt_cot, final_answer


def _parse_olympiadbench(ground_truth: Any) -> Tuple[None, str]:
    """Helper function to parse OlympiadBench format answers."""
    if not isinstance(ground_truth, (list, str)):
        raise ValueError('OlympiadBench answer must be a list or string')

    if isinstance(ground_truth, list):
        if not ground_truth:
            raise ValueError('OlympiadBench answer list is empty')
        answer = str(ground_truth[0])
    else:
        answer = str(ground_truth)

    # Clean up the answer
    answer = answer.strip('$').strip()
    if not answer:
        raise ValueError('OlympiadBench final answer is empty')

    return None, answer


def _parse_generic(ground_truth: Any) -> Tuple[None, str]:
    """Helper function to parse generic format answers."""
    answer = str(ground_truth).strip()
    if not answer:
        raise ValueError('Final answer is empty')
    return None, answer
