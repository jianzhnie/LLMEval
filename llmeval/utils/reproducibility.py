"""Best-effort, dependency-optional random seed management."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

__all__ = ["SeedState", "seed_everything", "seed_provenance"]


@dataclass(frozen=True)
class SeedState:
    """Seeds applied to the available random number generators."""

    seed: int
    python_seed: int
    numpy_seed: int | None
    torch_seed: int | None


def seed_everything(seed: int) -> SeedState:
    """Seed Python and optionally installed NumPy/PyTorch random sources."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    numpy_seed: int | None = None
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seed = seed
    except (ImportError, ValueError):
        pass

    torch_seed: int | None = None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seed = seed
    except (ImportError, RuntimeError):
        pass
    return SeedState(seed, seed, numpy_seed, torch_seed)


def seed_provenance(state: SeedState) -> dict[str, Any]:
    """Serialize seed state for run provenance and cache payloads."""
    return {
        "seed": state.seed,
        "python_seed": state.python_seed,
        "numpy_seed": state.numpy_seed,
        "torch_seed": state.torch_seed,
        "fewshot_seed": state.seed,
        "generation_seed": state.seed,
    }
