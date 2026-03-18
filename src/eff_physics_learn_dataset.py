"""Thin convenience module exposing the flat dataset API.

This keeps `import eff_physics_learn_dataset` working while the actual
implementation lives in flat modules under `src/`.
"""

from __future__ import annotations

from dataset import PDEDataset, load_pde_dataset
from download import download_dataset, load_dataset_links

__version__ = "0.1.0"

__all__ = [
    "download_dataset",
    "load_dataset_links",
    "load_pde_dataset",
    "PDEDataset",
    "__version__",
]

