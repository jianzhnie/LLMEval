"""Inference backends for LLM evaluation.

Provides both online (OpenAI-compatible API) and offline (vLLM engine)
inference, plus MC-specific and verifier variants.

Usage::

    from llmeval.inference.online import InferenceClient
    from llmeval.inference.offline import OfflineInferenceRunner
    from llmeval.inference.verifier import VerifierInferenceRunner
    from llmeval.inference.mc import MCRunner
"""
