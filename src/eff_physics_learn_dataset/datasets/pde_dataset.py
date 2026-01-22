from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence

import numpy as np

from .pkl_schema import infer_pde_pickle
from .parametric_split import split_interpolation_extrapolation
from .plotting import (
    plot_param_points,
    plot_solution_grid,
    plot_solution_rows,
    plot_solution_similarity_hist,
)
from .solution_similarity import (
    farthest_point_subset,
    fit_solution_pca,
    pairwise_min_distances,
    solution_similarity_report,
    vectorize_solutions,
)

BudgetName = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class PDEDataset:
    """A lightweight, HF-like dataset wrapper around on-disk pickle artifacts.

    The dataset is assumed to live at:
      datasets/{equation}/ground_truth/
    and contain one in-distribution `*_dataset.pkl` plus `test_indices.pkl`.
    """

    equation: str
    root_dir: Path
    ground_truth_dir: Path

    # Core arrays
    u: np.ndarray  # (N, H, W) or (N, ...); solution snapshots/fields
    params: np.ndarray  # (N, P) float
    param_names: tuple[str, ...]

    # Optional grids/metadata
    grids: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    grid_info: Mapping[str, Any]

    # Index mapping (view semantics)
    _indices: np.ndarray | None = None

    def __len__(self) -> int:
        if self._indices is None:
            return int(self.u.shape[0])
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = int(idx)
        if self._indices is not None:
            i = int(self._indices[i])

        p = self.params[i]
        return {
            "equation": self.equation,
            "index": i,
            "u": self.u[i],
            "params": p,
            "param_dict": {k: float(v) for k, v in zip(self.param_names, p)},
            **{k: v for k, v in self.grids.items()},
        }

    @property
    def indices(self) -> np.ndarray:
        if self._indices is None:
            return np.arange(len(self.u), dtype=np.int64)
        return self._indices

    def select(self, indices: Sequence[int] | np.ndarray) -> "PDEDataset":
        ind = np.asarray(indices, dtype=np.int64)
        return PDEDataset(
            equation=self.equation,
            root_dir=self.root_dir,
            ground_truth_dir=self.ground_truth_dir,
            u=self.u,
            params=self.params,
            param_names=self.param_names,
            grids=self.grids,
            metadata=self.metadata,
            grid_info=self.grid_info,
            _indices=ind,
        )

    def _load_test_indices(self) -> np.ndarray:
        p = self.ground_truth_dir / "test_indices.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Missing test indices file: {p}")
        arr = _load_pickle(p)
        return np.asarray(arr, dtype=np.int64)

    def train_test_splits(
        self,
        *,
        seed: int,
        n_train: int,
        test_indices: Literal["from_file"] = "from_file",
        return_train_rest: bool = True,
    ) -> Dict[str, "PDEDataset"]:
        """Return seedable train budget split + fixed test split.

        - Test split is authoritative and loaded from `test_indices.pkl`.
        - Train pool is all remaining indices.
        - The `train` subset is a seeded subsample of train pool.
        """

        if test_indices != "from_file":
            raise ValueError("Only test_indices='from_file' is supported in this repo.")

        test_idx = self._load_test_indices()
        all_idx = np.arange(len(self.u), dtype=np.int64)
        is_test = np.zeros(len(self.u), dtype=bool)
        is_test[test_idx] = True
        train_pool = all_idx[~is_test]

        if n_train > len(train_pool):
            raise ValueError(f"Requested n_train={n_train} but train_pool has {len(train_pool)} samples.")

        rng = np.random.default_rng(int(seed))
        train_idx = rng.choice(train_pool, size=int(n_train), replace=False)

        out: Dict[str, PDEDataset] = {
            "train": self.select(train_idx),
            "test": self.select(test_idx),
        }

        if return_train_rest:
            train_rest = np.setdiff1d(train_pool, train_idx, assume_unique=False)
            out["train_rest"] = self.select(train_rest)

        return out

    def budget_split(self, budget: int | BudgetName, *, seed: int) -> Dict[str, "PDEDataset"]:
        """Convenience wrapper around `train_test_splits`.

        Returns a dict with keys `train`, `test`, and `train_rest`.
        """

        n_train = _budget_to_n_train(budget)
        return self.train_test_splits(seed=seed, n_train=n_train)

    def parametric_splits(
        self,
        *,
        seed: int,
        n_train: int,
        method: Literal["convex_hull", "solution_percentile"] = "solution_percentile",
        bounds_margin: float = 0.0,
        percentile: float = 50.0,
        n_interp: int | None = None,
        n_extrap: int | None = None,
        balance: bool = False,
        n_each: int | None = None,
        balance_strategy: Literal["random", "solution_nn"] = "random",
        replace: bool = False,
        on_insufficient: Literal["cap", "error"] = "cap",
        # For solution-space ranking when balance_strategy='solution_nn'
        solution_n_components: int = 5,
        solution_slice_axis: int = 1,
        solution_slice_index: int | None = None,
        diversify: bool = False,
        diversify_pool_mult: int = 5,
    ) -> Dict[str, "PDEDataset"]:
        """Few-shot split into train_few + interpolation + extrapolation.

        Methods:
        - "solution_percentile" (default): Split by distance in solution space.
          Samples closer than the percentile threshold are interp, farther are extrap.
          This directly measures generalization difficulty and works robustly across all parameter spaces.
        - "convex_hull": Split by convex hull membership in parameter space (legacy).
          May have weak separation for multi-parameter PDEs with nonlinear solution manifolds.

        Args:
            method: Splitting method ("solution_percentile" or "convex_hull")
            percentile: For solution_percentile, the distance percentile threshold (default: 50.0)
            Other args: See inline docs
        """

        # Reuse authoritative test split only for defining the train pool.
        test_idx = self._load_test_indices()
        all_idx = np.arange(len(self.u), dtype=np.int64)
        is_test = np.zeros(len(self.u), dtype=bool)
        is_test[test_idx] = True
        train_pool = all_idx[~is_test]

        if n_train > len(train_pool):
            raise ValueError(f"Requested n_train={n_train} but train_pool has {len(train_pool)} samples.")

        rng = np.random.default_rng(int(seed))
        train_few_idx = rng.choice(train_pool, size=int(n_train), replace=False)

        candidates_idx = np.setdiff1d(all_idx, train_few_idx, assume_unique=False)

        if method == "solution_percentile":
            # Solution-space percentile splitting
            union_idx = np.unique(np.concatenate([train_few_idx, candidates_idx]).astype(np.int64))
            X = vectorize_solutions(
                self.u[union_idx],
                slice_axis=solution_slice_axis,
                slice_index=solution_slice_index,
            )
            _, Z = fit_solution_pca(X, n_components=int(solution_n_components))
            row = {int(i): j for j, i in enumerate(union_idx.tolist())}

            Z_train = Z[np.asarray([row[int(i)] for i in train_few_idx], dtype=np.int64)]
            Z_candidates = Z[np.asarray([row[int(i)] for i in candidates_idx], dtype=np.int64)]

            # Compute NN distances
            nn_distances = pairwise_min_distances(Z_candidates, Z_train)

            # Split by percentile
            threshold = np.percentile(nn_distances, percentile)
            is_interp = nn_distances <= threshold
            is_extrap = nn_distances > threshold

            interp_idx = candidates_idx[is_interp]
            extrap_idx = candidates_idx[is_extrap]
        else:
            # Legacy convex hull method
            train_params = self.params[train_few_idx]
            cand_params = self.params[candidates_idx]

            is_interp, is_extrap = split_interpolation_extrapolation(
                train_params=train_params,
                candidate_params=cand_params,
                method=method,
                bounds_margin=bounds_margin,
            )

            interp_idx = candidates_idx[is_interp]
            extrap_idx = candidates_idx[is_extrap]

        # Optional balancing/subsampling (use a separate RNG stream to keep train_few stable)
        sub_rng = np.random.default_rng(int(seed) + 1_000_003)

        def subsample(idx: np.ndarray, n: int) -> np.ndarray:
            if n < 0:
                raise ValueError("n must be >= 0")
            if len(idx) < n and not replace:
                if on_insufficient == "error":
                    raise ValueError(f"Requested n={n} but only {len(idx)} available (set replace=True or cap).")
                # cap
                n = len(idx)
            if n == len(idx):
                return idx
            return sub_rng.choice(idx, size=int(n), replace=bool(replace))

        if balance:
            target = min(len(interp_idx), len(extrap_idx))
            if n_each is not None:
                target = min(target, int(n_each))

            if balance_strategy == "random":
                # Random sampling, but ensure interp samples are closer than extrap samples in solution space
                # This prevents counterintuitive results where extrap appears closer than interp
                interp_idx_sampled = subsample(interp_idx, target)
                extrap_idx_sampled = subsample(extrap_idx, target)
                
                # Verify solution-space ordering: compute distances and ensure interp < extrap
                # Fit PCA on union of all candidates (not just sampled) for consistent embedding
                union_idx = np.unique(np.concatenate([train_few_idx, interp_idx, extrap_idx]).astype(np.int64))
                X = vectorize_solutions(
                    self.u[union_idx],
                    slice_axis=solution_slice_axis,
                    slice_index=solution_slice_index,
                )
                _, Z = fit_solution_pca(X, n_components=int(solution_n_components))
                row = {int(i): j for j, i in enumerate(union_idx.tolist())}
                
                Z_train = Z[np.asarray([row[int(i)] for i in train_few_idx], dtype=np.int64)]
                Z_interp_sampled = Z[np.asarray([row[int(i)] for i in interp_idx_sampled], dtype=np.int64)]
                Z_extrap_sampled = Z[np.asarray([row[int(i)] for i in extrap_idx_sampled], dtype=np.int64)]
                
                d_interp_sampled = pairwise_min_distances(Z_interp_sampled, Z_train)
                d_extrap_sampled = pairwise_min_distances(Z_extrap_sampled, Z_train)
                
                # If interp samples are not closer than extrap samples, enforce ordering
                # by selecting closer interp and farther extrap samples
                if d_interp_sampled.mean() > d_extrap_sampled.mean():
                    # Re-rank: pick closest interp and farthest extrap from all candidates
                    Z_interp_all = Z[np.asarray([row[int(i)] for i in interp_idx], dtype=np.int64)]
                    Z_extrap_all = Z[np.asarray([row[int(i)] for i in extrap_idx], dtype=np.int64)]
                    
                    d_interp_all = pairwise_min_distances(Z_interp_all, Z_train)
                    d_extrap_all = pairwise_min_distances(Z_extrap_all, Z_train)
                    
                    interp_order = np.argsort(d_interp_all)[:target]
                    extrap_order = np.argsort(d_extrap_all)[::-1][:target]
                    
                    interp_idx = interp_idx[interp_order]
                    extrap_idx = extrap_idx[extrap_order]
                else:
                    interp_idx = interp_idx_sampled
                    extrap_idx = extrap_idx_sampled
            elif balance_strategy == "solution_nn":
                # Enforce: interp = closest-to-train, extrap = farthest-from-train in solution-PCA space
                # Fit PCA on union of (train_few + interp candidates + extrap candidates) for stability.
                union_idx = np.unique(np.concatenate([train_few_idx, interp_idx, extrap_idx]).astype(np.int64))
                X = vectorize_solutions(
                    self.u[union_idx],
                    slice_axis=solution_slice_axis,
                    slice_index=solution_slice_index,
                )
                _, Z = fit_solution_pca(X, n_components=int(solution_n_components))
                row = {int(i): j for j, i in enumerate(union_idx.tolist())}

                Z_train = Z[np.asarray([row[int(i)] for i in train_few_idx], dtype=np.int64)]
                Z_interp = Z[np.asarray([row[int(i)] for i in interp_idx], dtype=np.int64)]
                Z_extrap = Z[np.asarray([row[int(i)] for i in extrap_idx], dtype=np.int64)]

                d_interp = pairwise_min_distances(Z_interp, Z_train)
                d_extrap = pairwise_min_distances(Z_extrap, Z_train)

                # Base ranking: closest `target` interp, farthest `target` extrap
                interp_order = np.argsort(d_interp)  # ascending
                extrap_order = np.argsort(d_extrap)[::-1]  # descending

                if not diversify:
                    interp_idx = interp_idx[interp_order[:target]]
                    extrap_idx = extrap_idx[extrap_order[:target]]
                else:
                    m = int(max(target, min(len(interp_idx), len(extrap_idx))) )
                    pool = int(max(target, int(diversify_pool_mult) * target))
                    pool_i = min(pool, len(interp_idx))
                    pool_e = min(pool, len(extrap_idx))

                    interp_pool_idx = interp_order[:pool_i]
                    extrap_pool_idx = extrap_order[:pool_e]

                    # Select diverse subset within the pools in PCA space
                    sel_i = farthest_point_subset(Z_interp[interp_pool_idx], target, seed=int(seed) + 7)
                    sel_e = farthest_point_subset(Z_extrap[extrap_pool_idx], target, seed=int(seed) + 11)

                    interp_idx = interp_idx[interp_pool_idx[sel_i]]
                    extrap_idx = extrap_idx[extrap_pool_idx[sel_e]]
            else:
                raise ValueError("balance_strategy must be 'random' or 'solution_nn'")
        else:
            if n_interp is not None:
                interp_idx = subsample(interp_idx, int(n_interp))
            if n_extrap is not None:
                extrap_idx = subsample(extrap_idx, int(n_extrap))

        return {
            "train_few": self.select(train_few_idx),
            "interp": self.select(interp_idx),
            "extrap": self.select(extrap_idx),
        }

    def plot_samples(
        self,
        *,
        split: Literal["test", "all"] | "PDEDataset" = "test",
        n: int = 25,
        seed: int = 0,
        slice_index: int | None = None,
        slice_axis: int = 1,
        save_path: Path | str | None = None,
        title: str | None = None,
    ):
        """Plot a grid of sample solutions.

        For the standard smoke test, use `split='test'` and `n=25`.
        """

        if isinstance(split, str):
            if split == "test":
                view = self.select(self._load_test_indices())
            elif split == "all":
                view = self
            else:
                raise ValueError("split must be 'test', 'all', or a PDEDataset view")
        else:
            view = split

        idx = np.asarray(view.indices, dtype=np.int64)
        sol = self.u[idx]
        pmat = self.params[idx]
        pdicts = [
            {k: float(v) for k, v in zip(self.param_names, row)}  # small & readable titles
            for row in pmat
        ]

        return plot_solution_grid(
            solutions=sol,
            param_dicts=pdicts,
            n=n,
            seed=seed,
            slice_index=slice_index,
            slice_axis=slice_axis,
            save_path=save_path,
            title=title or f"{self.equation} ({'test' if isinstance(split,str) else 'subset'}) n={n}",
        )

    def plot_param_splits(
        self,
        *,
        splits: Mapping[str, "PDEDataset"],
        projection: Literal["auto", "raw", "pca"] = "auto",
        save_path: Path | str | None = None,
        title: str | None = None,
        seed: int = 0,
        side_by_side: bool = False,
        side_param_index: int = 0,
        side_title: str | None = None,
        side_mode: str = "param_hist",
        side_scatter: Mapping[str, np.ndarray] | None = None,
        side_values: Mapping[str, np.ndarray] | None = None,
        side_cmap: str = "viridis",
        side_colorbar_label: str | None = None,
        side_density: bool = False,
        side_log_y: bool = False,
    ):
        """Plot parameter distributions for provided splits (train/test or train_few/interp/extrap)."""

        param_arrays = {name: self.params[np.asarray(ds.indices, dtype=np.int64)] for name, ds in splits.items()}
        return plot_param_points(
            splits=param_arrays,
            param_names=self.param_names,
            title=title,
            projection=projection,
            seed=seed,
            save_path=save_path,
            side_by_side=side_by_side,
            side_param_index=side_param_index,
            side_title=side_title,
            side_mode=side_mode,
            side_scatter=side_scatter,
            side_values=side_values,
            side_cmap=side_cmap,
            side_colorbar_label=side_colorbar_label,
            side_density=side_density,
            side_log_y=side_log_y,
        )

    def solution_similarity_report(
        self,
        *,
        splits: Mapping[str, "PDEDataset"],
        train_key: str,
        n_components: int = 5,
        slice_axis: int = 1,
        slice_index: int | None = None,
    ) -> Dict[str, Any]:
        """Compute a solution-space similarity report for the provided splits."""

        idx_splits = {k: np.asarray(v.indices, dtype=np.int64) for k, v in splits.items()}
        return solution_similarity_report(
            u=self.u,
            splits=idx_splits,
            train_key=train_key,
            n_components=n_components,
            slice_axis=slice_axis,
            slice_index=slice_index,
        )

    def plot_solution_similarity(
        self,
        *,
        report: Mapping[str, Any],
        metric: str = "nn_to_train",
        density: bool = False,
        log_y: bool = False,
        show_medians: bool = True,
        save_path: Path | str | None = None,
        title: str | None = None,
    ):
        """Plot a histogram of solution-space distances from a similarity report."""

        return plot_solution_similarity_hist(
            report=report,
            metric=metric,
            title=title,
            density=density,
            log_y=log_y,
            show_medians=show_medians,
            save_path=save_path,
        )

    def plot_split_solution_rows(
        self,
        *,
        splits: Mapping[str, "PDEDataset"],
        n_per_row: int = 5,
        seed: int = 0,
        slice_index: int | None = None,
        slice_axis: int = 1,
        plot_style: Literal["imshow", "contourf"] = "imshow",
        contour_levels: int = 12,
        contour_line_color: str = "black",
        show_axes: bool = False,
        save_path: Path | str | None = None,
        title: str | None = None,
    ):
        """Plot one row per split with `n_per_row` solution samples each."""

        solutions_by = {}
        params_by = {}
        gidx_by = {}
        for name, ds in splits.items():
            idx = np.asarray(ds.indices, dtype=np.int64)
            solutions_by[name] = self.u[idx]
            pmat = self.params[idx]
            params_by[name] = [
                {k: float(v) for k, v in zip(self.param_names, row)} for row in pmat
            ]
            gidx_by[name] = idx.tolist()

        extent = None
        if "X_grid" in self.grids and "T_grid" in self.grids:
            X = np.asarray(self.grids["X_grid"])
            T = np.asarray(self.grids["T_grid"])
            extent = (float(X.min()), float(X.max()), float(T.min()), float(T.max()))

        return plot_solution_rows(
            solutions_by_split=solutions_by,
            params_by_split=params_by,
            global_indices_by_split=gidx_by,
            n_per_row=n_per_row,
            seed=seed,
            slice_index=slice_index,
            slice_axis=slice_axis,
            plot_style=plot_style,
            contour_levels=contour_levels,
            contour_line_color=contour_line_color,
            extent=extent,
            show_axes=show_axes,
            save_path=save_path,
            title=title,
        )


def load_pde_dataset(
    equation: str,
    data_dir: Path | str = Path("datasets"),
    *,
    cache: bool = True,
) -> PDEDataset:
    """Load an equation dataset from local disk.

    Args:
        equation: directory name under `data_dir` (e.g. "helmholtz2D", "burgers")
        data_dir: base datasets directory (default: ./datasets)
        cache: if True, memoize loads within the process
    """

    data_dir = Path(data_dir)
    if cache:
        return _load_pde_dataset_cached(equation, data_dir)
    return _load_pde_dataset_uncached(equation, data_dir)


@lru_cache(maxsize=32)
def _load_pde_dataset_cached(equation: str, data_dir: Path) -> PDEDataset:
    return _load_pde_dataset_uncached(equation, data_dir)


def _load_pde_dataset_uncached(equation: str, data_dir: Path) -> PDEDataset:
    root_dir = data_dir / equation
    ground_truth_dir = root_dir / "ground_truth"
    if not ground_truth_dir.exists():
        # Some datasets (e.g. helmholtz3D) currently store artifacts directly under the equation dir.
        alt = root_dir
        if alt.exists() and any(alt.glob("*_dataset.pkl")):
            ground_truth_dir = alt
        else:
            raise FileNotFoundError(f"Missing ground_truth dir: {ground_truth_dir}")

    pkl_path = _find_main_dataset_pkl(ground_truth_dir)
    raw = _load_pickle(pkl_path)
    inferred = infer_pde_pickle(raw, param_order="file")

    return PDEDataset(
        equation=equation,
        root_dir=root_dir,
        ground_truth_dir=ground_truth_dir,
        u=inferred.u,
        params=inferred.params,
        param_names=inferred.param_names,
        grids=inferred.grids,
        metadata=inferred.metadata,
        grid_info=inferred.grid_info,
        _indices=None,
    )


def _find_main_dataset_pkl(ground_truth_dir: Path) -> Path:
    # Prefer the non-OOD dataset pickle.
    candidates = sorted(ground_truth_dir.glob("*_dataset.pkl"))
    candidates = [p for p in candidates if "ood" not in p.name.lower() and "backup" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No main '*_dataset.pkl' found under {ground_truth_dir}")
    if len(candidates) > 1:
        # Choose the shortest name deterministically (usually the main one).
        candidates.sort(key=lambda p: (len(p.name), p.name))
    return candidates[0]


def _load_pickle(path: Path) -> Any:
    import pickle

    return pickle.loads(path.read_bytes())


def _budget_to_n_train(budget: int | BudgetName) -> int:
    if isinstance(budget, int):
        return int(budget)
    if budget == "low":
        return 25
    if budget == "medium":
        return 50
    if budget == "high":
        return 75
    raise ValueError(f"Unknown budget: {budget}")


