from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal

import numpy as np

from .similarity import (
    farthest_point_subset,
    fit_solution_pca,
    pairwise_min_distances,
    vectorize_solutions,
)


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
        from scipy.spatial import (
            Delaunay,  # type: ignore
            QhullError,  # type: ignore
        )

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


@dataclass(frozen=True)
class ParametricSplitConfig:
    seed: int
    n_train: int
    method: Literal["convex_hull", "solution_percentile"] = "solution_percentile"
    bounds_margin: float = 0.0
    percentile: float = 50.0
    n_interp: int | None = None
    n_extrap: int | None = None
    balance: bool = False
    n_each: int | None = None
    balance_strategy: Literal["random", "solution_nn"] = "random"
    replace: bool = False
    on_insufficient: Literal["cap", "error"] = "cap"
    solution_n_components: int = 5
    solution_slice_axis: int = 1
    solution_slice_index: int | None = None
    diversify: bool = False
    diversify_pool_mult: int = 5


def _embed_in_solution_space(
    u: np.ndarray,
    index_sets: list[np.ndarray],
    config: ParametricSplitConfig,
) -> tuple[np.ndarray, dict[int, int]]:
    """Fit PCA over the union of index_sets and return (Z, row_map).

    Z is the full embedding matrix (one row per element in the union).
    row_map maps original dataset index -> row in Z.
    """
    union_idx = np.unique(np.concatenate([idx.astype(np.int64) for idx in index_sets]))
    X = vectorize_solutions(
        u[union_idx],
        slice_axis=config.solution_slice_axis,
        slice_index=config.solution_slice_index,
    )
    _, Z = fit_solution_pca(X, n_components=int(config.solution_n_components))
    row_map = {int(i): j for j, i in enumerate(union_idx.tolist())}
    return Z, row_map


def _select_train_few(
    train_pool: np.ndarray,
    n_train: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Validate pool size and draw train_few indices."""
    if n_train > len(train_pool):
        raise ValueError(
            f"Requested n_train={n_train} but train_pool has {len(train_pool)} samples."
        )
    return rng.choice(train_pool, size=int(n_train), replace=False)


def _split_by_percentile(
    u: np.ndarray,
    train_few_idx: np.ndarray,
    candidates_idx: np.ndarray,
    config: ParametricSplitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Split candidates into interp/extrap by solution-space distance percentile."""
    Z, row_map = _embed_in_solution_space(u, [train_few_idx, candidates_idx], config)
    Z_train = Z[np.asarray([row_map[int(i)] for i in train_few_idx], dtype=np.int64)]
    Z_candidates = Z[np.asarray([row_map[int(i)] for i in candidates_idx], dtype=np.int64)]
    nn_distances = pairwise_min_distances(Z_candidates, Z_train)
    threshold = np.percentile(nn_distances, config.percentile)
    is_interp = nn_distances <= threshold
    is_extrap = nn_distances > threshold
    return candidates_idx[is_interp], candidates_idx[is_extrap]


def _balance_splits(
    interp_idx: np.ndarray,
    extrap_idx: np.ndarray,
    train_few_idx: np.ndarray,
    u: np.ndarray,
    config: ParametricSplitConfig,
    subsample_fn: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Balance interp/extrap splits to equal size using the configured strategy."""
    target = min(len(interp_idx), len(extrap_idx))
    if config.n_each is not None:
        target = min(target, int(config.n_each))

    if config.balance_strategy == "random":
        interp_idx_sampled = subsample_fn(interp_idx, target)
        extrap_idx_sampled = subsample_fn(extrap_idx, target)

        Z, row_map = _embed_in_solution_space(
            u, [train_few_idx, interp_idx, extrap_idx], config
        )
        Z_train = Z[np.asarray([row_map[int(i)] for i in train_few_idx], dtype=np.int64)]
        Z_interp_sampled = Z[np.asarray([row_map[int(i)] for i in interp_idx_sampled], dtype=np.int64)]
        Z_extrap_sampled = Z[np.asarray([row_map[int(i)] for i in extrap_idx_sampled], dtype=np.int64)]

        d_interp_sampled = pairwise_min_distances(Z_interp_sampled, Z_train)
        d_extrap_sampled = pairwise_min_distances(Z_extrap_sampled, Z_train)

        if d_interp_sampled.mean() > d_extrap_sampled.mean():
            Z_interp_all = Z[np.asarray([row_map[int(i)] for i in interp_idx], dtype=np.int64)]
            Z_extrap_all = Z[np.asarray([row_map[int(i)] for i in extrap_idx], dtype=np.int64)]

            d_interp_all = pairwise_min_distances(Z_interp_all, Z_train)
            d_extrap_all = pairwise_min_distances(Z_extrap_all, Z_train)

            interp_order = np.argsort(d_interp_all)[:target]
            extrap_order = np.argsort(d_extrap_all)[::-1][:target]

            return interp_idx[interp_order], extrap_idx[extrap_order]
        else:
            return interp_idx_sampled, extrap_idx_sampled

    elif config.balance_strategy == "solution_nn":
        Z, row_map = _embed_in_solution_space(
            u, [train_few_idx, interp_idx, extrap_idx], config
        )
        Z_train = Z[np.asarray([row_map[int(i)] for i in train_few_idx], dtype=np.int64)]
        Z_interp = Z[np.asarray([row_map[int(i)] for i in interp_idx], dtype=np.int64)]
        Z_extrap = Z[np.asarray([row_map[int(i)] for i in extrap_idx], dtype=np.int64)]

        d_interp = pairwise_min_distances(Z_interp, Z_train)
        d_extrap = pairwise_min_distances(Z_extrap, Z_train)

        interp_order = np.argsort(d_interp)
        extrap_order = np.argsort(d_extrap)[::-1]

        if not config.diversify:
            return interp_idx[interp_order[:target]], extrap_idx[extrap_order[:target]]
        else:
            pool = int(max(target, int(config.diversify_pool_mult) * target))
            pool_i = min(pool, len(interp_idx))
            pool_e = min(pool, len(extrap_idx))

            interp_pool_idx = interp_order[:pool_i]
            extrap_pool_idx = extrap_order[:pool_e]

            sel_i = farthest_point_subset(Z_interp[interp_pool_idx], target, seed=int(config.seed) + 7)
            sel_e = farthest_point_subset(Z_extrap[extrap_pool_idx], target, seed=int(config.seed) + 11)

            return interp_idx[interp_pool_idx[sel_i]], extrap_idx[extrap_pool_idx[sel_e]]
    else:
        raise ValueError("balance_strategy must be 'random' or 'solution_nn'")


def compute_parametric_splits(
    *,
    u: np.ndarray,
    params: np.ndarray,
    test_indices: np.ndarray,
    config: ParametricSplitConfig,
) -> Dict[str, np.ndarray]:
    """Core implementation for parametric few-shot / interp / extrap splits.

    This is a direct extraction of the logic from `PDEDataset.parametric_splits`,
    operating only on arrays and indices. It returns index arrays for each split.
    """

    all_idx = np.arange(u.shape[0], dtype=np.int64)
    is_test = np.zeros(u.shape[0], dtype=bool)
    is_test[test_indices] = True
    train_pool = all_idx[~is_test]

    rng = np.random.default_rng(int(config.seed))
    train_few_idx = _select_train_few(train_pool, config.n_train, rng)

    candidates_idx = np.setdiff1d(all_idx, train_few_idx, assume_unique=False)

    if config.method == "solution_percentile":
        interp_idx, extrap_idx = _split_by_percentile(u, train_few_idx, candidates_idx, config)
    else:
        train_params = params[train_few_idx]
        cand_params = params[candidates_idx]

        is_interp, is_extrap = split_interpolation_extrapolation(
            train_params=train_params,
            candidate_params=cand_params,
            method=config.method,
            bounds_margin=config.bounds_margin,
        )

        interp_idx = candidates_idx[is_interp]
        extrap_idx = candidates_idx[is_extrap]

    sub_rng = np.random.default_rng(int(config.seed) + 1_000_003)

    def subsample(idx: np.ndarray, n: int) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be >= 0")
        if len(idx) < n and not config.replace:
            if config.on_insufficient == "error":
                raise ValueError(f"Requested n={n} but only {len(idx)} available (set replace=True or cap).")
            n = len(idx)
        if n == len(idx):
            return idx
        return sub_rng.choice(idx, size=int(n), replace=bool(config.replace))

    if config.balance:
        interp_idx, extrap_idx = _balance_splits(
            interp_idx, extrap_idx, train_few_idx, u, config, subsample
        )
    else:
        if config.n_interp is not None:
            interp_idx = subsample(interp_idx, int(config.n_interp))
        if config.n_extrap is not None:
            extrap_idx = subsample(extrap_idx, int(config.n_extrap))

    return {
        "train_few": np.asarray(train_few_idx, dtype=np.int64),
        "interp": np.asarray(interp_idx, dtype=np.int64),
        "extrap": np.asarray(extrap_idx, dtype=np.int64),
    }
