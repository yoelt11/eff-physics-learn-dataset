"""Efficient Physics Learning Datasets - JAX-based dataset utilities for PDE learning."""

__version__ = "0.1.0"

from .download import download_dataset, load_dataset_links

__all__ = ["download_dataset", "load_dataset_links", "__version__"]

