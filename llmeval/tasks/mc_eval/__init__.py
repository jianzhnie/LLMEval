"""Multiple-choice evaluation: loglikelihood comparison and generation-based scoring.

Public API::

    from llmeval.tasks.mc_eval import (
        score_loglikelihood, score_generate,  # scoring entry points
        MCScoreResult,                        # metrics container
    )
"""

from llmeval.tasks.mc_eval.mc_score import (
    MCScoreResult,
    score_generate,
    score_loglikelihood,
)

__all__ = [
    "MCScoreResult",
    "score_generate",
    "score_loglikelihood",
]
