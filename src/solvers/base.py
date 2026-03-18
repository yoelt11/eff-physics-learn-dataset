from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SolverConfig:
    """Minimal configuration for a dataset-generating solver."""

    out_dir: Path
    n_samples: int
    seed: int = 0


class Solver(Protocol):
    """Protocol for PDE dataset solvers.

    Implementations should generate a dataset (including the main *_dataset.pkl
    and test_indices.pkl) under `config.out_dir` and return the path to the main
    dataset pickle.
    """

    def __call__(self, config: SolverConfig) -> Path:  # pragma: no cover - structural
        ...

