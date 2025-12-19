from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class InferredPDEData:
    u: np.ndarray
    params: np.ndarray
    param_names: tuple[str, ...]
    grids: Dict[str, np.ndarray]
    metadata: Mapping[str, Any]
    grid_info: Mapping[str, Any]


def infer_pde_pickle(
    raw: Any,
    *,
    param_order: str = "file",
) -> InferredPDEData:
    """Infer a consistent in-memory representation from a dataset pickle.

    Supports the common structure used by this repo's datasets:
      - solutions: (N, H, W) float array
      - pde_params: dict of parameter arrays length N
      - X_grid/Y_grid/T_grid: 2D grids
      - metadata/grid_info: dicts
    """

    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict pickle, got {type(raw)}")

    u = _infer_solutions(raw)
    params, param_names = _infer_params(raw, n_samples=int(u.shape[0]), param_order=param_order)

    grids: Dict[str, np.ndarray] = {}
    for k in ("X_grid", "Y_grid", "Z_grid", "T_grid"):
        if k in raw:
            grids[k] = np.asarray(raw[k])
    # Support alternative grid naming in some datasets (e.g. helmholtz3D: X/Y/Z)
    if "X" in raw and "X_grid" not in grids:
        grids["X_grid"] = np.asarray(raw["X"])
    if "Y" in raw and "Y_grid" not in grids:
        grids["Y_grid"] = np.asarray(raw["Y"])
    if "Z" in raw and "Z_grid" not in grids:
        grids["Z_grid"] = np.asarray(raw["Z"])
    # 1D coordinate arrays (optional, useful for volumetric tasks)
    for k in ("x", "y", "z", "t"):
        if k in raw and k not in grids:
            arr = np.asarray(raw[k])
            if arr.ndim == 1:
                grids[k] = arr

    metadata = raw.get("metadata", {})
    grid_info = raw.get("grid_info", {})

    return InferredPDEData(
        u=u,
        params=params,
        param_names=param_names,
        grids=grids,
        metadata=metadata if isinstance(metadata, dict) else {},
        grid_info=grid_info if isinstance(grid_info, dict) else {},
    )


def _infer_solutions(raw: Mapping[str, Any]) -> np.ndarray:
    # Primary key used across current datasets
    if "solutions" in raw:
        u = np.asarray(raw["solutions"])
        if u.ndim < 2:
            raise ValueError(f"Unexpected solutions array shape: {u.shape}")
        return u
    # Alternative key used by some datasets (e.g. helmholtz3D)
    if "u" in raw:
        u = np.asarray(raw["u"])
        if u.ndim < 2:
            raise ValueError(f"Unexpected u array shape: {u.shape}")
        return u

    # Fallback: choose the largest array whose first dim looks like N
    arrays = []
    for k, v in raw.items():
        if hasattr(v, "shape"):
            a = np.asarray(v)
            if a.ndim >= 2:
                arrays.append((k, a))
    if not arrays:
        raise KeyError("Could not infer solutions array from pickle (no array-like values).")
    arrays.sort(key=lambda kv: (kv[1].size, kv[0]), reverse=True)
    return arrays[0][1]


def _infer_params(
    raw: Mapping[str, Any],
    *,
    n_samples: int,
    param_order: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    # Common format: pde_params dict
    if "pde_params" in raw and isinstance(raw["pde_params"], dict):
        pde_params = raw["pde_params"]
        names: Sequence[str]
        if param_order == "sorted":
            names = sorted(pde_params.keys())
        elif param_order == "file":
            # pickle preserves insertion order
            names = list(pde_params.keys())
        else:
            raise ValueError("param_order must be 'file' or 'sorted'")

        cols = []
        for k in names:
            col = np.asarray(pde_params[k]).reshape(-1)
            if col.shape[0] != n_samples:
                raise ValueError(
                    f"Param {k} has length {col.shape[0]} but expected n_samples={n_samples}"
                )
            cols.append(col.astype(np.float64))
        params = np.stack(cols, axis=1)
        return params, tuple(names)

    # Fallback format: parameters stored as top-level 1D arrays of length N.
    # (e.g. helmholtz3D_dataset.pkl has a1/a2/a3/k at top level)
    ignore = {
        "solutions",
        "u",
        "X_grid",
        "Y_grid",
        "Z_grid",
        "T_grid",
        "X",
        "Y",
        "Z",
        "x",
        "y",
        "z",
        "t",
        "metadata",
        "grid_info",
        "domain",
        "grid_shape",
        "grid_size",
        "n_samples",
        "valid_solutions",
        "failed_solutions",
        "jax_accelerated",
    }
    candidates: list[str] = []
    for k, v in raw.items():
        if k in ignore:
            continue
        if hasattr(v, "shape"):
            a = np.asarray(v)
            if a.ndim == 1 and a.shape[0] == n_samples and np.issubdtype(a.dtype, np.number):
                candidates.append(k)

    if not candidates:
        raise KeyError("Expected `pde_params` dict in dataset pickle (or top-level 1D param arrays).")

    if param_order == "sorted":
        candidates = sorted(candidates)
    elif param_order == "file":
        # preserve pickle insertion order by iterating original dict keys
        ordered = [k for k in raw.keys() if k in set(candidates)]
        candidates = ordered
    else:
        raise ValueError("param_order must be 'file' or 'sorted'")

    cols = [np.asarray(raw[k]).reshape(-1).astype(np.float64) for k in candidates]
    params = np.stack(cols, axis=1)
    return params, tuple(candidates)


