"""Thin convenience module exposing the flat dataset API.

This keeps `import eff_physics_learn_dataset` working while the actual
implementation lives in flat modules under `src/`.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.dataset import PDEDataset, load_pde_dataset as _base_load_pde_dataset
from src.download import download_dataset, load_dataset_links

__version__ = "0.1.0"


def load_pde_dataset(
    equation: str,
    data_dir: str | os.PathLike[str] | None = None,
    *,
    cache: bool = True,
    auto_download: bool = True,
    tmp_root: str | os.PathLike[str] | None = None,
) -> PDEDataset:
    """Load a PDE dataset, optionally auto-downloading if missing.

    Default behavior:
    - First, try the base loader (which expects ./datasets/{equation}/ground_truth).
    - If the ground-truth directory is missing and ``auto_download`` is True,
      download the dataset into a known temporary location and reload from there.

    Args:
        equation: Name of the equation/dataset (e.g. \"burgers\").
        data_dir: Optional root directory containing ``{equation}/ground_truth``.
        cache: Whether to use the dataset-level LRU cache (unchanged behavior).
        auto_download: If True, automatically call ``download_dataset`` when the
            ground-truth directory is missing.
        tmp_root: Optional override for the temporary datasets root. If not
            provided, defaults to ``/tmp/eff-physics-learn-dataset`` or the value
            of the ``EFF_PHYSICS_LEARN_DATA_DIR`` environment variable.
    """
    try:
        return _base_load_pde_dataset(equation, data_dir=data_dir, cache=cache)
    except FileNotFoundError:
        if not auto_download:
            raise

        root = Path(
            tmp_root
            or os.environ.get("EFF_PHYSICS_LEARN_DATA_DIR", "/tmp/eff-physics-learn-dataset")
        ).expanduser()

        # Ensure parent directory exists before download.
        root.mkdir(parents=True, exist_ok=True)

        download_dataset(equation, data_dir=root)
        return _base_load_pde_dataset(equation, data_dir=root, cache=cache)


__all__ = [
    "download_dataset",
    "load_dataset_links",
    "load_pde_dataset",
    "PDEDataset",
    "__version__",
]

