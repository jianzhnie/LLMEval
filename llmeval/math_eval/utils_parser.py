from typing import Any, Dict, Optional, Tuple


def parse_ground_truth(example: Dict[str, Any],
                       data_name: str) -> Tuple[Optional[str], str]:
    """
    从示例中解析出思维链 (CoT) 和最终答案 (Answer)。

    Args:
        example (Dict[str, Any]): 包含原始数据的字典。
        data_name (str): 数据集名称，用于决定解析方式。

    Returns:
        Tuple[Optional[str], str]: 返回 (CoT, Answer)，若无 CoT 则为 (None, Answer)。

    Raises:
        ValueError: 如果无法提取到答案。
        NotImplementedError: 如果不支持该数据集。
    """
    if data_name == 'gsm8k':
        answer_field = example.get('answer', '')
        if '####' not in answer_field:
            raise ValueError(f"GSM8K 示例缺失 '####' 分隔符: {example}")
        parts = answer_field.split('####')
        gt_cot = parts[0].strip()
        gt_ans = parts[1].strip()
        return gt_cot, gt_ans

    elif data_name == 'olympiadbench':
        final_answer = example.get('final_answer', [])
        if not final_answer or not isinstance(final_answer, list):
            raise ValueError(f'olympiadbench 缺失或格式错误的 final_answer: {example}')
        return None, final_answer[0].strip('$')

    elif data_name in [
            'aime24',
            'amc23',
            'cmath',
            'imo2024',
            'aimo12',
            'cnmo24',
            'aime25',
            'math500box',
    ]:
        answer = example.get('answer', '')
        if not answer:
            raise ValueError(f"{data_name} 缺失 'answer' 字段: {example}")
        return None, answer.strip()

    else:
        raise NotImplementedError(f'不支持的数据集: `{data_name}`')
