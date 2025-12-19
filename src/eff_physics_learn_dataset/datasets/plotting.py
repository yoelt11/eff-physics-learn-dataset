from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_STYLE_APPLIED = False


def _apply_pitayasmoothie_like_rcparams() -> None:
    """Apply a local equivalent of pitayasmoothie-light.mplstyle.

    We avoid fetching remote stylesheets at runtime (offline-safe) and instead set the
    key rcParams here.
    """

    import matplotlib as mpl

    try:
        from cycler import cycler
    except Exception:
        cycler = None  # type: ignore

    rc = {
        # Seaborn common parameters
        "text.color": ".15",
        "axes.labelcolor": ".15",
        "xtick.color": ".15",
        "ytick.color": ".15",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "grid.linestyle": "-",
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
        # Seaborn darkgrid parameters
        "axes.grid": True,
        "axes.facecolor": "EAEAF2",
        "axes.edgecolor": "white",
        "axes.linewidth": 0.0,
        "grid.color": "white",
        "xtick.major.size": 0.0,
        "ytick.major.size": 0.0,
        "ytick.minor.size": 0.0,
        # Custom
        "font.sans-serif": [
            "Overpass",
            "Helvetica",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ],
        "image.cmap": "RdPu",
        "figure.facecolor": "fefeff",
        "savefig.facecolor": "fefeff",
    }

    if cycler is not None:
        rc["axes.prop_cycle"] = cycler(
            "color", ["7A76C2", "ff6e9c98", "f62196", "18c0c4", "f3907e", "66E9EC"]
        )

    mpl.rcParams.update(rc)


def _apply_rbf_rcparams() -> None:
    """Apply the PINNs project RBF rcParams."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 18,
            "axes.labelweight": "regular",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 24,
            "figure.titleweight": "bold",
            "axes.grid": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",  # keep text as text for editability
            "svg.hashsalt": None,  # stable svg output
        }
    )


def maybe_apply_style() -> None:
    """Apply user's preferred plotting style if available.

    This repo is used across machines; the custom style paths may not exist.
    """

    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Let users disable styling entirely (useful for downstream apps that manage styles).
    if os.environ.get("EFF_PHYSICS_LEARN_STYLE", "").lower() in {"0", "off", "false"}:
        _STYLE_APPLIED = True
        return

    # 1) Explicit env override: point to a .mplstyle on disk.
    # Example:
    #   export EFF_PHYSICS_LEARN_MPLSTYLE=/path/to/pitayasmoothie-light.mplstyle
    env_style = os.environ.get("EFF_PHYSICS_LEARN_MPLSTYLE")
    if env_style and os.path.exists(env_style):
        try:
            plt.style.use(env_style)
            _apply_rbf_rcparams()
            _STYLE_APPLIED = True
            return
        except Exception:
            pass

    # 2) If your custom plotting module is installed, prefer it (keeps parity across projects).
    # The user mentioned:
    #   /home/etorres/Documents/github/personal/research-project-pinns/projects/plotting_module
    try:
        from plotting_module.src.plot import set_style  # type: ignore

        set_style(mode="light")
        _apply_rbf_rcparams()
        _STYLE_APPLIED = True
        return
    except Exception:
        pass

    # Allow override via env var for portability across machines.
    # 3) User-provided default style path (may or may not exist on this machine)
    style_path = "/home/etorres/Documents/github/personal/research-project-pinns/style/pitayasmoothie-light.mplstyle"
    if os.path.exists(style_path):
        try:
            plt.style.use(style_path)
            _apply_rbf_rcparams()
            _STYLE_APPLIED = True
            return
        except Exception:
            pass

    # 4) Offline-safe built-in approximation of the pitayasmoothie+RBF look.
    _apply_pitayasmoothie_like_rcparams()
    _apply_rbf_rcparams()
    _STYLE_APPLIED = True


def plot_solution_grid(
    *,
    solutions: np.ndarray,
    param_dicts: Sequence[Mapping[str, float]] | None = None,
    n: int = 25,
    seed: int = 0,
    slice_index: int | None = None,
    slice_axis: int = 1,
    save_path: Path | str | None = None,
    title: str | None = None,
) -> Any:
    """Plot N random 2D solution snapshots in a grid.

    Supports:
    - `(N, H, W)` solutions
    - `(N, S, H, W)` solutions (e.g. time series): slices along `slice_axis` (default: 1)
    """

    import matplotlib.pyplot as plt

    maybe_apply_style()

    solutions = np.asarray(solutions)
    if solutions.ndim == 4:
        # Common case in this repo: (N, T, H, W) with T_grid length T
        ax = int(slice_axis)
        if ax < 0:
            ax = solutions.ndim + ax
        if ax not in (1, 3):
            raise ValueError(f"slice_axis must be 1 or 3 for 4D solutions, got {slice_axis}")

        S = solutions.shape[ax]
        if slice_index is None:
            slice_index = S // 2
        slice_index = int(slice_index)
        if not (0 <= slice_index < S):
            raise ValueError(f"slice_index={slice_index} out of bounds for axis size {S}")

        # Move slice axis to position 1 so we can select [:, slice_index, ...]
        if ax != 1:
            solutions = np.moveaxis(solutions, ax, 1)
        solutions = solutions[:, slice_index, :, :]

        if title is None:
            title = f"slice={slice_index}"

    if solutions.ndim != 3:
        raise ValueError(
            f"Expected solutions with shape (N,H,W) or (N,S,H,W), got {solutions.shape}"
        )

    N = solutions.shape[0]
    n = int(min(n, N))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.arange(N), size=n, replace=False)

    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(2.3 * side, 2.3 * side), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(side, side)

    if title is not None:
        fig.suptitle(title)

    for ax in axes.ravel():
        ax.axis("off")

    for k, i in enumerate(chosen):
        r, c = divmod(k, side)
        ax = axes[r, c]
        im = ax.imshow(solutions[i], origin="lower", aspect="auto")
        ax.axis("off")
        if param_dicts is not None:
            pd = param_dicts[i]
            small = ", ".join([f"{kk}={vv:.3g}" for kk, vv in list(pd.items())[:3]])
            ax.set_title(small, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig


def plot_param_points(
    *,
    splits: Mapping[str, np.ndarray],
    param_names: Sequence[str],
    title: str | None = None,
    projection: str = "auto",
    seed: int = 0,
    save_path: Path | str | None = None,
) -> Any:
    """Plot param-space distributions for multiple splits.

    - If P==1: overlaid histograms
    - If P==2: scatter in (p0, p1)
    - If P==3 and projection=='3d': 3D scatter in (p0,p1,p2)
    - Otherwise: 2D projection (default PCA) for scatter
    """

    import matplotlib.pyplot as plt

    maybe_apply_style()

    # Stack for projection statistics
    arrays = [np.asarray(v, dtype=np.float64) for v in splits.values()]
    if not arrays:
        raise ValueError("splits is empty")

    P = int(arrays[0].shape[1])
    if any(a.ndim != 2 or a.shape[1] != P for a in arrays):
        raise ValueError("All split param arrays must be 2D with same feature dimension")

    if projection == "auto":
        projection = "pca" if P > 2 else "raw"
    if projection not in ("raw", "pca", "3d"):
        raise ValueError("projection must be 'auto', 'raw', 'pca', or '3d'")

    # Colors
    labels = list(splits.keys())
    cmap = plt.get_cmap("tab10")
    colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    fig = plt.figure(figsize=(7.5, 6.2), constrained_layout=True)

    if title:
        fig.suptitle(title)

    if P == 1:
        ax = fig.add_subplot(1, 1, 1)
        for lab, a in splits.items():
            ax.hist(a[:, 0], bins=30, alpha=0.45, label=lab, color=colors[lab], density=True)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel("density")
        ax.legend()
    else:
        if P == 3 and projection == "3d":
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            ax = fig.add_subplot(1, 1, 1, projection="3d")
            for lab, a in splits.items():
                ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=14, alpha=0.7, label=lab, color=colors[lab])
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_zlabel(param_names[2])
            ax.legend()
        else:
            ax = fig.add_subplot(1, 1, 1)
            if P == 2 or projection == "raw":
                for lab, a in splits.items():
                    ax.scatter(a[:, 0], a[:, 1], s=14, alpha=0.7, label=lab, color=colors[lab])
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
            else:
                # PCA to 2D using SVD (no sklearn dependency)
                X = np.concatenate(arrays, axis=0)
                mu = X.mean(axis=0, keepdims=True)
                Xc = X - mu
                # Vt shape (P,P); first 2 PCs are Vt[:2]
                _, _, vt = np.linalg.svd(Xc, full_matrices=False)
                W = vt[:2].T  # (P,2)

                for lab, a in splits.items():
                    z = (a - mu) @ W
                    ax.scatter(z[:, 0], z[:, 1], s=14, alpha=0.7, label=lab, color=colors[lab])
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("PCA projection (params)")

            ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig


def plot_solution_similarity_hist(
    *,
    report: Mapping[str, Any],
    metric: str = "nn_to_train",
    title: str | None = None,
    density: bool = False,
    log_y: bool = False,
    show_medians: bool = True,
    save_path: Path | str | None = None,
) -> Any:
    """Plot distance distributions from a `solution_similarity_report`."""

    import matplotlib.pyplot as plt

    maybe_apply_style()

    splits = report.get("splits", {})
    fig = plt.figure(figsize=(7.5, 5.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    if title:
        fig.suptitle(title)

    labels = list(splits.keys())
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(labels):
        d = np.asarray(splits[lab].get(metric, []), dtype=np.float32)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        ax.hist(
            d,
            bins=30,
            alpha=0.45,
            density=bool(density),
            label=lab,
            color=cmap(i % 10),
        )
        if show_medians:
            med = float(np.median(d))
            ax.axvline(med, color=cmap(i % 10), linestyle="--", linewidth=1.5, alpha=0.9)
            ax.text(
                med,
                0.95,
                f"{lab} med={med:.2f}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
                color=cmap(i % 10),
            )

    ax.set_xlabel(metric)
    ax.set_ylabel("density" if density else "count")
    if log_y:
        ax.set_yscale("log")
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig


def plot_solution_rows(
    *,
    solutions_by_split: Mapping[str, np.ndarray],
    params_by_split: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
    global_indices_by_split: Mapping[str, Sequence[int]] | None = None,
    n_per_row: int = 5,
    seed: int = 0,
    slice_index: int | None = None,
    slice_axis: int = 1,
    save_path: Path | str | None = None,
    title: str | None = None,
) -> Any:
    """Plot a row per split, with N solutions per row.

    Each split array can be:
    - (N,H,W)
    - (N,S,H,W) (will be sliced to 2D like `plot_solution_grid`)
    """

    import matplotlib.pyplot as plt

    maybe_apply_style()

    split_names = list(solutions_by_split.keys())
    if not split_names:
        raise ValueError("solutions_by_split is empty")

    rng = np.random.default_rng(int(seed))

    rows = len(split_names)
    cols = int(n_per_row)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.4 * cols, 2.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    if title:
        fig.suptitle(title)

    for r, split in enumerate(split_names):
        sol = np.asarray(solutions_by_split[split])
        # Reuse slicing logic from plot_solution_grid for 4D arrays
        if sol.ndim == 4:
            ax = int(slice_axis)
            if ax < 0:
                ax = sol.ndim + ax
            S = sol.shape[ax]
            si = S // 2 if slice_index is None else int(slice_index)
            if not (0 <= si < S):
                raise ValueError(f"slice_index={si} out of bounds for axis size {S}")
            if ax != 1:
                sol = np.moveaxis(sol, ax, 1)  # (N,S,H,W)
            sol = sol[:, si, :, :]

        if sol.ndim != 3:
            raise ValueError(f"Expected (N,H,W) or (N,S,H,W) per split; got {sol.shape} for {split}")

        N = sol.shape[0]
        if N == 0:
            for c in range(cols):
                axes[r, c].axis("off")
            continue

        choose = min(cols, N)
        idx = rng.choice(np.arange(N), size=choose, replace=False)
        # If not enough samples, pad with repeats to fill the row
        if choose < cols:
            pad = rng.choice(idx, size=cols - choose, replace=True)
            idx = np.concatenate([idx, pad], axis=0)

        for c in range(cols):
            axc = axes[r, c]
            axc.axis("off")
            im = axc.imshow(sol[int(idx[c])], origin="lower", aspect="auto")
            if c == 0:
                axc.set_ylabel(split, rotation=90, fontsize=11)

            title_parts = []
            if params_by_split is not None and split in params_by_split:
                pdicts = params_by_split[split]
                if int(idx[c]) < len(pdicts):
                    pd = pdicts[int(idx[c])]
                    small = ", ".join([f"{kk}={vv:.3g}" for kk, vv in list(pd.items())[:3]])
                    title_parts.append(small)

            if global_indices_by_split is not None and split in global_indices_by_split:
                gidx = global_indices_by_split[split]
                if int(idx[c]) < len(gidx):
                    title_parts.append(f"idx={int(gidx[int(idx[c])])}")

            if title_parts:
                axc.set_title(" | ".join(title_parts), fontsize=8)

            fig.colorbar(im, ax=axc, fraction=0.046, pad=0.01)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig
