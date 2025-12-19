from __future__ import annotations

from typing import Tuple

import numpy as np


def split_interpolation_extrapolation(
    *,
    train_params: np.ndarray,
    candidate_params: np.ndarray,
    method: str = "convex_hull",
    bounds_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (is_interp, is_extrap) over candidates.

    - **convex_hull**: candidate is interpolation if inside convex hull of train params.
    - fallback: axis-aligned bounds in parameter space.
    """

    train_params = np.asarray(train_params, dtype=np.float64)
    candidate_params = np.asarray(candidate_params, dtype=np.float64)

    if method != "convex_hull":
        raise ValueError("Only method='convex_hull' is supported by plan.")

    if train_params.ndim != 2 or candidate_params.ndim != 2:
        raise ValueError("train_params and candidate_params must be 2D arrays.")
    if train_params.shape[1] != candidate_params.shape[1]:
        raise ValueError("train_params and candidate_params must have same feature dimension.")

    is_interp = None
    try:
        from scipy.spatial import Delaunay  # type: ignore
        from scipy.spatial import QhullError  # type: ignore

        try:
            tri = Delaunay(train_params)
            is_interp = tri.find_simplex(candidate_params) >= 0
        except QhullError:
            is_interp = None
    except Exception:
        is_interp = None

    if is_interp is None:
        is_interp = _axis_aligned_in_bounds(train_params, candidate_params, margin=bounds_margin)

    is_extrap = ~is_interp
    return is_interp, is_extrap


def _axis_aligned_in_bounds(
    train_params: np.ndarray, candidate_params: np.ndarray, *, margin: float
) -> np.ndarray:
    lo = train_params.min(axis=0) - float(margin)
    hi = train_params.max(axis=0) + float(margin)
    return np.all((candidate_params >= lo) & (candidate_params <= hi), axis=1)


