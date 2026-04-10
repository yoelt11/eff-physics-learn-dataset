"""Efficient Physics Learning Datasets - utilities for loading and working with PDE datasets."""

from __future__ import annotations

import os
from pathlib import Path

from .dataset import PDEDataset
from .dataset import load_pde_dataset as _base_load_pde_dataset
from .download import download_dataset, load_dataset_links
from .field_layout import FieldLayout, infer_field_layout

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
        equation: Name of the equation/dataset (e.g. "burgers").
        data_dir: Optional root directory containing ``{equation}/ground_truth``.
        cache: Whether to use the dataset-level LRU cache (unchanged behavior).
        auto_download: If True, automatically call ``download_dataset`` when the
            ground-truth directory is missing.
        tmp_root: Optional override for the temporary datasets root. If not
            provided, defaults to ``/tmp/eff-physics-learn-dataset`` or the value
            of the ``EFF_PHYSICS_LEARN_DATA_DIR`` environment variable.
    """
    if data_dir is None:
        data_dir = "datasets"

    try:
        return _base_load_pde_dataset(equation, data_dir=data_dir, cache=cache)
    except FileNotFoundError:
        if not auto_download:
            raise

        root = Path(
            tmp_root
            or os.environ.get("EFF_PHYSICS_LEARN_DATA_DIR", "/tmp/eff-physics-learn-dataset")
        ).expanduser()

        root.mkdir(parents=True, exist_ok=True)

        links = load_dataset_links()
        file_id = links.get(equation) or links.get(equation.lower()) or links.get(equation.upper())
        if not file_id:
            raise KeyError(
                f"Unknown dataset '{equation}'. Available: {sorted(list(links.keys()))[:20]}..."
            )

        ok = download_dataset(name=equation, file_id=file_id, output_dir=root, extract=True)
        if not ok:
            raise RuntimeError(f"Failed to download dataset '{equation}' into: {root}")
        return _base_load_pde_dataset(equation, data_dir=root, cache=cache)


__all__ = [
    "download_dataset",
    "FieldLayout",
    "infer_field_layout",
    "load_dataset_links",
    "load_pde_dataset",
    "PDEDataset",
    "__version__",
]
