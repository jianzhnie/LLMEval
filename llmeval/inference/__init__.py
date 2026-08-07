"""Inference backends for LLM evaluation.

Provides online (OpenAI-compatible API), offline (vLLM), and MC inference.

Usage::

    from llmeval.inference.online import InferenceClient
    from llmeval.inference.offline import OfflineInferenceRunner
    from llmeval.inference.mc import MCRunner
"""
