"""
LLMEval — A comprehensive evaluation system for Large Language Models.

Package structure::

    llmeval/
    ├── evaluator.py       # Main orchestrator (dispatches to task scorers)
    ├── inference/         # Inference backends (online API / offline vLLM / MC)
    ├── tasks/             # Task-specific scoring
    │   ├── code_eval/     #   Code generation (HumanEval / MBPP / pass@k)
    │   ├── math_eval/     #   Math reasoning (math-verify)
    │   └── mc_eval/       #   Multiple choice (accuracy)
    └── utils/             # Shared config, logging, retry, prompts

Quick-start::

    from llmeval.evaluator import evaluate_task

    accuracy = evaluate_task(data, "code_opensource/humaneval",
                             label_key="answer", response_key="gen",
                             cache_path="./cache", max_workers=8)
"""
