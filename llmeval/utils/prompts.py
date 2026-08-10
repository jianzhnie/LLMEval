from __future__ import annotations

import re
from typing import Final

# Defaults (used only if CLI values are not provided)
amthinking_system_prompt: Final[str] = (
    "You are a helpful assistant. To answer the user's question, you first think "
    "about the reasoning process and then provide the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>."
)

deepseek_r1_system_prompt: Final[str] = (
    "A conversation between User and Assistant. The User asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the "
    "reasoning process in the mind and then provides the User with the "
    "answer. The reasoning process is enclosed within <think> </think> "
    "and the answer is enclosed within <answer> </answer>."
)

openr1_system_prompt: Final[str] = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then "
    "provide the user with the answer. Respond in the following format: "
    "<think>\n...\n</think>\n<answer>\n...\n</answer>"
)

default_system_prompt: Final[str] = "You are a helpful assistant."

# A factory for different types of system prompts.
SYSTEM_PROMPT_FACTORY: dict[str, str | None] = {
    "deepseek_r1": deepseek_r1_system_prompt,
    "amthinking": amthinking_system_prompt,
    "openr1": openr1_system_prompt,
    "default": default_system_prompt,
    "empty": None,
}

_SPECIAL_TOKEN_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|user|>",
    "<|assistant|>",
)

# Token boundaries keep HTML tags and ordinary text such as ``x[INST]y``
# from being mistaken for Llama control tokens.
_LLAMA_CONTROL_TOKEN_RE = re.compile(
    r"(?m)(?:^|(?<=\s))</?s>(?=\s|$)|(?<!\w)\[/?INST\](?!\w)"
)
_CHAT_ROLE_LINE_RE = re.compile(r"(?m)^\s*(?:(###)\s*)?(Human|Assistant)\s*:")


def is_chat_template_applied(query: str) -> bool:
    """Return whether ``query`` contains a serialized chat template."""
    if not query:
        return False

    if any(marker in query for marker in _SPECIAL_TOKEN_MARKERS):
        return True

    if _LLAMA_CONTROL_TOKEN_RE.search(query):
        return True

    plain_roles: set[str] = set()
    for hashed, role in _CHAT_ROLE_LINE_RE.findall(query):
        if hashed:
            return True
        plain_roles.add(role)

    return plain_roles == {"Human", "Assistant"}
