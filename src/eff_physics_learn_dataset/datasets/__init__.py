"""HuggingFace-like dataset API for PDE datasets in this repo."""

from .pde_dataset import PDEDataset, load_pde_dataset

__all__ = ["PDEDataset", "load_pde_dataset"]


