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

_CHAT_ROLE_LINE_RE: re.Pattern[str] = re.compile(
    r"(?m)^\s*(?:###\s*)?(?:Human|Assistant)\s*:",
)

# Unambiguous special tokens — plain substring matching is safe for these.
_SPECIAL_TOKEN_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",  # ChatML format
    "<|user|>",
    "<|assistant|>",  # Other formats
)

# Llama-style control tokens need context to avoid false positives: bare
# <s>/</s> collide with HTML tags (<sub>, <sup>, <span>, <strong>, ...) and
# [INST] can appear glued to ordinary text, so <s>/</s> must be a complete
# tag sitting at the start of the string or right after whitespace/a line
# start, followed by whitespace or the end of the string, and [INST] must
# not touch word characters.
_S_TAG_RE: re.Pattern[str] = re.compile(r"(?m)(?:^|(?<=\s))</?s>(?=\s|$)")
_INST_TAG_RE: re.Pattern[str] = re.compile(r"(?<!\w)\[/?INST\](?!\w)")


def is_chat_template_applied(query: str) -> bool:
    """Check if the query has already been processed with a chat template.

    Args:
        query: The input query string

    Returns:
        True if chat template appears to be applied, False otherwise
    """
    if not query:
        return False

    if any(marker in query for marker in _SPECIAL_TOKEN_MARKERS):
        return True

    if _S_TAG_RE.search(query) or _INST_TAG_RE.search(query):
        return True

    # Human:/Assistant: is too common as normal problem text to detect via
    # substring; treat it as a template only when it appears as a role line.
    return bool(_CHAT_ROLE_LINE_RE.search(query))
