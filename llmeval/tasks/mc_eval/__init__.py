"""Multiple-choice evaluation: loglikelihood comparison and generation-based scoring.

Public API::

    from llmeval.tasks.mc_eval import (
        score_loglikelihood_result, score_generate_result,
        MCScoreResult,
    )
"""

from llmeval.tasks.mc_eval.mc_score import (
    MCScoreResult,
    score_generate_result,
    score_loglikelihood_result,
)

__all__ = [
    "MCScoreResult",
    "score_generate_result",
    "score_loglikelihood_result",
]
