"""Efficient Physics Learning Datasets - JAX-based dataset utilities for PDE learning."""

__version__ = "0.1.0"

from .download import download_dataset, load_dataset_links
from .datasets import PDEDataset, load_pde_dataset

__all__ = [
    "download_dataset",
    "load_dataset_links",
    "load_pde_dataset",
    "PDEDataset",
    "__version__",
]

