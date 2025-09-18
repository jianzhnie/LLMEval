from typing import Dict, Final, Optional

# Defaults (used only if CLI values are not provided)
amthinking_system_prompt: Final[str] = (
    "You are a helpful assistant. To answer the user's question, you first think "
    'about the reasoning process and then provide the user with the answer. '
    'The reasoning process and answer are enclosed within <think> </think> and '
    '<answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think> <answer> answer here </answer>.')

deepseek_r1_system_prompt: Final[str] = (
    'A conversation between User and Assistant. The User asks a question, '
    'and the Assistant solves it. The Assistant first thinks about the '
    'reasoning process in the mind and then provides the User with the '
    'answer. The reasoning process is enclosed within <think> </think> '
    'and the answer is enclosed within <answer> </answer>.')

openr1_system_prompt: Final[str] = (
    'You are a helpful AI Assistant that provides well-reasoned and detailed responses. '
    'You first think about the reasoning process as an internal monologue and then '
    'provide the user with the answer. Respond in the following format: '
    '<think>\n...\n</think>\n<answer>\n...\n</answer>')

qwen_math_cot_prompt: Final[str] = (
    'Please reason step by step, and put your final answer within \\boxed{}.')

default_system_prompt: Final[str] = 'You are a helpful AI assistant.'

# A factory for different types of system prompts.
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'deepseek_r1': deepseek_r1_system_prompt,
    'amthinking': amthinking_system_prompt,
    'openr1': openr1_system_prompt,
    'default': default_system_prompt,
    'empty': None
}


def is_chat_template_applied(query: str) -> bool:
    """Check if the query has already been processed with a chat template.

    Args:
        query: The input query string

    Returns:
        True if chat template appears to be applied, False otherwise
    """
    # Common chat template markers that indicate the query is already formatted
    template_markers = [
        '<|im_start|>',
        '<|im_end|>',  # ChatML format
        '<s>',
        '</s>',  # Some models use these
        '[INST]',
        '[/INST]',  # Llama format
        '<|user|>',
        '<|assistant|>',  # Other formats
        '### Human:',
        '### Assistant:',  # Alpaca format
        'Human:',
        'Assistant:',  # Simple format
    ]

    # Check if any template markers are present
    return any(marker in query for marker in template_markers)
