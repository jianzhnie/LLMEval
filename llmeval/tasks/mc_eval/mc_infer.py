"""Multiple-choice inference: loglikelihood and generation modes.

loglikelihood mode (aligned with lm-eval loglikelihood approach):
- For each question, compute logprob of each choice continuation
- Pick the choice with the highest total logprob
- Uses /v1/completions endpoint with logprobs

generate mode:
- Standard chat completions, extract answer letter from generated text
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from tqdm import tqdm

from llmeval.utils.logger import init_logger

logger = init_logger("mc_infer")


@dataclass
class MCInferConfig:
    """Configuration for MC inference."""
    input_file: str = ""
    output_file: str = ""
    base_url: str = "http://127.0.0.1:8200/v1"
    model_name: str = "longcat-flash"
    mode: str = "loglikelihood"  # "loglikelihood" | "generate"
    max_workers: int = 32
    request_timeout: int = 300
    max_retries: int = 3
    max_tokens: int = 2048        # for generate mode
    temperature: float = 0.0      # for generate mode
    system_prompt_type: str = "empty"
    tool_choice: str = "none"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"))


def run_mc_infer(config: MCInferConfig) -> None:
    """Run MC inference in the specified mode."""
    if config.mode == "loglikelihood":
        _run_loglikelihood(config)
    elif config.mode == "generate":
        _run_generate(config)
    else:
        logger.error(f"Unknown mode: {config.mode}. Use 'loglikelihood' or 'generate'.")
        sys.exit(1)


# ===========================================================================
# Loglikelihood mode
# ===========================================================================

def _run_loglikelihood(config: MCInferConfig) -> None:
    """Compute logprob of each choice for each question via completions API."""
    items = _load_items(config.input_file)
    if not items:
        logger.error("No items loaded.")
        return

    client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)

    results = []
    for item in tqdm(items, desc="loglikelihood"):
        prompt = item["prompt"]
        choices = item.get("choices", [])
        gold = item.get("gold", -1)

        if not choices:
            continue

        logprobs = []
        for choice_text in choices:
            # Continuation format: prompt + space + choice
            full_text = f"{prompt} {choice_text}"
            lp = _get_continuation_logprob(client, config.model_name, full_text, config)
            logprobs.append(lp)

        results.append({
            "prompt": prompt,
            "gold": gold,
            "logprobs": logprobs,
        })

    # Write results
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick accuracy summary
    correct = sum(
        1 for r in results
        if r["logprobs"] and r["gold"] >= 0
        and max(range(len(r["logprobs"])), key=lambda i: r["logprobs"][i]) == r["gold"]
    )
    total = len([r for r in results if r["logprobs"] and r["gold"] >= 0])
    logger.info(f"Accuracy (loglikelihood): {correct}/{total} = {correct/total:.2%}" if total else "N/A")
    logger.info(f"Results saved to {output_path}")


def _get_continuation_logprob(
    client: openai.OpenAI, model: str, text: str, config: MCInferConfig,
) -> float:
    """Get log-probability of the continuation part of `text`."""
    for attempt in range(config.max_retries + 1):
        try:
            resp = client.completions.create(
                model=model,
                prompt=text,
                max_tokens=1,
                temperature=0,
                logprobs=1,
                echo=True,
                timeout=config.request_timeout,
            )
            # Sum logprobs of all tokens (prompt + choice).
            # Since prompt is identical for all choices, logprob(prompt) cancels
            # out in comparison: argmax_i Σ logprob(prompt+choice_i) = argmax_i logprob(choice_i|prompt)
            logprob_data = resp.choices[0].logprobs
            if logprob_data and logprob_data.token_logprobs:
                return sum(lp for lp in logprob_data.token_logprobs if lp is not None)
            return float("-inf")
        except Exception as e:
            if attempt == config.max_retries:
                logger.warning(f"Logprob request failed: {e}")
                return float("-inf")
    return float("-inf")


# ===========================================================================
# Generate mode
# ===========================================================================

def _run_generate(config: MCInferConfig) -> None:
    """Generate text responses and extract answer letters."""
    items = _load_items(config.input_file)
    if not items:
        return

    client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)

    # Build messages (optionally with system prompt)
    messages: list[dict[str, str]] = []
    if config.system_prompt_type != "empty":
        from llmeval.utils.template import SYSTEM_PROMPT_FACTORY
        sp = SYSTEM_PROMPT_FACTORY.get(config.system_prompt_type)
        if sp:
            messages.append({"role": "system", "content": sp})

    results = []
    for item in tqdm(items, desc="generate"):
        prompt = item["prompt"]
        gold = item.get("answer", "")
        item_messages = messages + [{"role": "user", "content": prompt}]

        for attempt in range(config.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": config.model_name,
                    "messages": item_messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "timeout": config.request_timeout,
                }
                if config.tool_choice:
                    kwargs["tool_choice"] = config.tool_choice
                resp = client.chat.completions.create(**kwargs)
                gen_text = resp.choices[0].message.content or ""
                break
            except Exception:
                if attempt == config.max_retries:
                    gen_text = ""

        results.append({
            "prompt": prompt,
            "answer": gold,
            "gen": [gen_text],
        })

    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Generate results saved to {output_path}")


# ===========================================================================
# Helpers
# ===========================================================================

def _load_items(input_file: str) -> list[dict[str, Any]]:
    items = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} items from {input_file}")
    return items


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MC inference (loglikelihood or generate)")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--model_name", default="longcat-flash")
    parser.add_argument("--mode", default="loglikelihood", choices=["loglikelihood", "generate"])
    parser.add_argument("--max_workers", type=int, default=32)  # ignored, single-threaded for now
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system_prompt_type", default="empty")
    parser.add_argument("--tool_choice", default="none")
    args = parser.parse_args()

    config = MCInferConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        base_url=args.base_url,
        model_name=args.model_name,
        mode=args.mode,
        max_workers=args.max_workers,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt_type=args.system_prompt_type,
        tool_choice=args.tool_choice,
    )
    run_mc_infer(config)


if __name__ == "__main__":
    main()
