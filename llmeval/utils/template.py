from typing import Dict, Optional

# A factory for different types of system prompts.
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'deepseek_r1':
    ('A conversation between User and Assistant. The User asks a question, '
     'and the Assistant solves it. The Assistant first thinks about the '
     'reasoning process in the mind and then provides the User with the '
     'answer. The reasoning process is enclosed within <think> </think> '
     'and the answer is enclosed within <answer> </answer>.'),
    'openr1_prompt':
    ('You are a helpful AI Assistant that provides well-reasoned and detailed responses. '
     'You first think about the reasoning process as an internal monologue and then '
     'provide the user with the answer. Respond in the following format: '
     '<think>\n...\n</think>\n<answer>\n...\n</answer>'),
    'default':
    'You are a helpful assistant.',
    'none':
    None,
}
