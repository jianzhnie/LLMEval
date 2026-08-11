"""Code evaluation — sandbox execution and pass@k scoring.

Public API::

    from llmeval.tasks.code_eval import (
        check_correctness,                  # execution guard
        extract_code, estimate_pass_at_k,   # code utilities
        score_code_result,                  # structured scoring
        CodeScoreResult, TimeoutException,  # types
    )
"""

from llmeval.tasks.code_eval.code_score import (
    CodeScoreResult,
    estimate_pass_at_k,
    extract_code,
    score_code_result,
)
from llmeval.tasks.code_eval.execute import (
    TimeoutException,
    check_correctness,
)

__all__ = [
    "CodeScoreResult",
    "TimeoutException",
    "check_correctness",
    "estimate_pass_at_k",
    "extract_code",
    "score_code_result",
]
