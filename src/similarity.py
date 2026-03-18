from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class SolutionPCAModel:
    mean: np.ndarray  # (D,)
    components: np.ndarray  # (K, D)
    explained_variance_ratio: np.ndarray  # (K,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.mean
        return Xc @ self.components.T


def vectorize_solutions(
    u: np.ndarray,
    *,
    slice_axis: int = 1,
    slice_index: int | None = None,
) -> np.ndarray:
    """Convert solution tensors into (N, D) feature vectors.

    Defaults:
    - (N,H,W): flatten
    - (N,S,H,W): take middle slice along `slice_axis` (default 1), then flatten
    - (N,H,W,S): if slice_axis=3, slice there, then flatten

    Note: For full 3D volumetric similarity, you may want a different featurization
    (e.g., downsample and flatten). This default is designed to be robust and cheap.
    """

    u = np.asarray(u)
    if u.ndim == 3:
        return u.reshape(u.shape[0], -1).astype(np.float32)

    if u.ndim == 4:
        ax = int(slice_axis)
        if ax < 0:
            ax = u.ndim + ax
        S = u.shape[ax]
        if slice_index is None:
            slice_index = S // 2
        slice_index = int(slice_index)
        if not (0 <= slice_index < S):
            raise ValueError(f"slice_index={slice_index} out of bounds for axis size {S}")

        if ax != 1:
            u = np.moveaxis(u, ax, 1)  # (N,S,H,W)
        u2 = u[:, slice_index, :, :]
        return u2.reshape(u2.shape[0], -1).astype(np.float32)

    raise ValueError(f"Unsupported solution shape for vectorization: {u.shape}")


def fit_solution_pca(
    X: np.ndarray,
    *,
    n_components: int = 5,
) -> tuple[SolutionPCAModel, np.ndarray]:
    """Fit PCA (via SVD) and return (model, embedding)."""

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D (N,D)")
    N, D = X.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to fit PCA")

    K = int(min(n_components, N, D))
    mean = X.mean(axis=0)
    Xc = X - mean

    # Economy SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:K].astype(np.float32)  # (K,D)

    # explained variance ratio
    # variance per PC = S^2 / (N-1)
    var = (S**2) / max(N - 1, 1)
    total_var = float(var.sum()) if var.size else 1.0
    evr = (var[:K] / total_var).astype(np.float32)

    model = SolutionPCAModel(mean=mean.astype(np.float32), components=comps, explained_variance_ratio=evr)
    Z = model.transform(X)
    return model, Z


def pairwise_min_distances(
    A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """For each row in A, return min Euclidean distance to any row in B.

    Uses the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    """

    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")

    aa = (A * A).sum(axis=1, keepdims=True)  # (na,1)
    bb = (B * B).sum(axis=1, keepdims=True).T  # (1,nb)
    d2 = aa + bb - 2.0 * (A @ B.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2.min(axis=1))


def farthest_point_subset(Z: np.ndarray, k: int, *, seed: int = 0) -> np.ndarray:
    """Pick a diverse subset of size k using greedy farthest-point sampling.

    Returns indices into Z.
    """

    Z = np.asarray(Z, dtype=np.float32)
    n = Z.shape[0]
    k = int(k)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    # Start from a random point
    first = int(rng.integers(0, n))
    chosen = [first]

    # Track min distance to the chosen set for each point
    d = np.sqrt(((Z - Z[first]) ** 2).sum(axis=1))
    d[first] = 0.0

    for _ in range(1, k):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        # update min distances
        dn = np.sqrt(((Z - Z[nxt]) ** 2).sum(axis=1))
        d = np.minimum(d, dn)
        d[nxt] = 0.0

    return np.asarray(chosen, dtype=np.int64)


def solution_similarity_report(
    *,
    u: np.ndarray,
    splits: Mapping[str, np.ndarray],
    train_key: str,
    n_components: int = 5,
    slice_axis: int = 1,
    slice_index: int | None = None,
) -> Dict[str, Any]:
    """Compute a similarity report in solution-space for provided index splits.

    - Fit PCA on all samples referenced by `splits` (union)
    - Embed each split into PCA space
    - Compute nearest-neighbor distances from each split to `train_key` split in PCA space
    """

    if train_key not in splits:
        raise KeyError(f"train_key={train_key!r} not in splits keys: {list(splits.keys())}")

    # Union of indices for fitting PCA
    all_idx = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in splits.values()]))
    X_all = vectorize_solutions(u[all_idx], slice_axis=slice_axis, slice_index=slice_index)

    model, Z_all = fit_solution_pca(X_all, n_components=n_components)
    # map idx -> embedding row
    idx_to_row = {int(i): j for j, i in enumerate(all_idx.tolist())}

    def embed(indices: np.ndarray) -> np.ndarray:
        rows = [idx_to_row[int(i)] for i in np.asarray(indices, dtype=np.int64)]
        return Z_all[np.asarray(rows, dtype=np.int64)]

    Z_train = embed(np.asarray(splits[train_key], dtype=np.int64))
    train_centroid = Z_train.mean(axis=0)

    out: Dict[str, Any] = {
        "train_key": train_key,
        "n_components": int(n_components),
        "explained_variance_ratio": model.explained_variance_ratio,
        "splits": {},
    }

    for name, idx in splits.items():
        Z = embed(np.asarray(idx, dtype=np.int64))
        nn = pairwise_min_distances(Z, Z_train)
        dc = np.linalg.norm(Z - train_centroid[None, :], axis=1)
        out["splits"][name] = {
            "n": int(len(idx)),
            "nn_to_train_mean": float(nn.mean()) if nn.size else float("nan"),
            "nn_to_train_median": float(np.median(nn)) if nn.size else float("nan"),
            "nn_to_train": nn,
            "dist_to_train_centroid_mean": float(dc.mean()) if dc.size else float("nan"),
            "dist_to_train_centroid_median": float(np.median(dc)) if dc.size else float("nan"),
            "dist_to_train_centroid": dc,
        }

    return out


