from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_STYLE_APPLIED = False

_SPLIT_COLORS: dict[str, str] = {
    "train_few": "#1b9e77", "train": "#1b9e77",
    "interp": "#7570b3",    "extrap": "#d95f02",
    "test": "#e7298a",      "val": "#66a61e",
}
_FALLBACK_COLORS: list[str] = ["#a6761d", "#e6ab02", "#666666"]


def _apply_rbf_rcparams() -> None:
    """Apply the ePIL-RBF project standard rcParams."""

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
            "grid.alpha": 0.3,  # Light and subtle grid
            "axes.facecolor": "white",  # White background instead of gray
            "axes.edgecolor": "black",  # Black frame border
            "axes.linewidth": 1.5,  # Thicker border
            "figure.facecolor": "white",  # White figure background
            "savefig.facecolor": "white",  # White saved figure background
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

    # 3) Simple, fully local, publication-friendly defaults.
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Liberation Sans",
                "Arial",
                "sans-serif",
            ],
            "image.cmap": "viridis",
            "axes.prop_cycle": cycler(
                "color",
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            ),
        }
    )
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
    plot_style: str = "contourf",
    contour_levels: int = 12,
    contour_line_color: str = "black",
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
        # Smaller + snake_case for consistent titles in generated plots.
        title_snake = re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip().lower()).strip("_")
        fig.suptitle(title_snake, fontsize=10, y=1.03)

    for ax in axes.ravel():
        ax.axis("off")

    for k, i in enumerate(chosen):
        r, c = divmod(k, side)
        ax = axes[r, c]
        field = solutions[i]

        if plot_style == "contourf":
            cf = ax.contourf(
                field,
                levels=int(contour_levels),
                origin="lower",
            )
            ax.contour(
                field,
                levels=int(contour_levels),
                colors=contour_line_color,
                linewidths=0.22,
                alpha=0.55,
                origin="lower",
            )
            # Keep panel visually square regardless of subplot box size.
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1.0)
            mappable = cf
        else:
            im = ax.imshow(field, origin="lower", aspect="auto")
            mappable = im
        ax.axis("off")

        # Add a subtle border around the panel (use axes coords so it works with axis("off")).
        if plot_style == "contourf":
            import matplotlib.patches as mpatches

            border = mpatches.Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                fill=False,
                linewidth=1.0,
                edgecolor="black",
                zorder=10,
            )
            ax.add_patch(border)
        if param_dicts is not None:
            pd = param_dicts[i]
            small = ", ".join([f"{kk}={vv:.3g}" for kk, vv in list(pd.items())[:3]])
            ax.set_title(small, fontsize=8)

        if plot_style == "contourf":
            # Put colorbar outside the panel to avoid covering contours.
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
        else:
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
        # Make the colorbar visually lighter than the panel.
        cbar.ax.tick_params(labelsize=9, width=0.8)
        cbar.outline.set_linewidth(0.8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both PNG and SVG for publication quality
        fig.savefig(save_path, dpi=300, format='png')
        svg_path = save_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', dpi=300)

    return fig


def _render_param_panel(
    ax: Any,
    splits: Mapping[str, np.ndarray],
    arrays: list[np.ndarray],
    colors: dict[str, str],
    param_names: Sequence[str],
    projection: str,
    P: int,
) -> None:
    """Render the main parameter-space panel onto an existing axis."""

    if P == 1:
        for lab, a in splits.items():
            ax.hist(a[:, 0], bins=30, alpha=0.45, label=lab, color=colors[lab], density=True)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel("density")
        ax.legend()
    elif P == 3 and projection == "3d":
        for lab, a in splits.items():
            ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=50, alpha=0.8,
                       label=lab, color=colors[lab], edgecolors='black', linewidth=0.5)
        ax.set_xlabel(param_names[0], fontweight='bold')
        ax.set_ylabel(param_names[1], fontweight='bold')
        ax.set_zlabel(param_names[2], fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    elif P == 2 or projection == "raw":
        for lab, a in splits.items():
            ax.scatter(a[:, 0], a[:, 1], s=50, alpha=0.8,
                       label=lab, color=colors[lab], edgecolors='black', linewidth=0.5)
        ax.set_xlabel(param_names[0], fontweight='bold')
        ax.set_ylabel(param_names[1], fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
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
            ax.scatter(z[:, 0], z[:, 1], s=50, alpha=0.8,
                       label=lab, color=colors[lab], edgecolors='black', linewidth=0.5)
        ax.set_xlabel("PC1", fontweight='bold')
        ax.set_ylabel("PC2", fontweight='bold')
        ax.set_title("PCA projection (params)", fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)


def _render_side_panel(
    ax_side: Any,
    fig: Any,
    labels: list[str],
    colors: dict[str, str],
    splits: Mapping[str, np.ndarray],
    arrays: list[np.ndarray],
    P: int,
    param_names: Sequence[str],
    *,
    side_mode: str,
    side_param_index: int,
    side_title: str | None,
    side_scatter: Mapping[str, np.ndarray] | None,
    side_values: Mapping[str, np.ndarray] | None,
    side_cmap: str,
    side_colorbar_label: str | None,
    side_density: bool,
    side_log_y: bool,
) -> None:
    """Render the side panel onto an existing axis."""

    if side_mode == "param_hist":
        if not (0 <= int(side_param_index) < P):
            raise ValueError(f"side_param_index must be in [0, {P - 1}]")
        idx = int(side_param_index)
        if P == 1:
            all_params = np.concatenate(arrays, axis=0)
            ax_side.hist(
                all_params[:, idx],
                bins=30,
                alpha=0.65,
                color="#666666",
                density=True,
                label="all",
            )
            ax_side.legend()
        else:
            for lab, a in splits.items():
                ax_side.hist(
                    a[:, idx],
                    bins=30,
                    alpha=0.45,
                    label=lab,
                    color=colors[lab],
                    density=True,
                )
            ax_side.legend()
        ax_side.set_xlabel(param_names[idx])
        ax_side.set_ylabel("density")
    elif side_mode == "solution_pca":
        if side_scatter is None:
            raise ValueError("side_scatter is required when side_mode='solution_pca'")
        all_vals = None
        if side_values is not None:
            all_vals = np.concatenate(
                [np.asarray(v, dtype=np.float32).ravel() for v in side_values.values()],
                axis=0,
            )
        vmin = float(np.min(all_vals)) if all_vals is not None and all_vals.size else None
        vmax = float(np.max(all_vals)) if all_vals is not None and all_vals.size else None
        markers = {
            "train_few": "o",
            "train": "o",
            "interp": "s",
            "extrap": "^",
            "test": "D",
            "val": "P",
        }
        mappable = None
        for i, (lab, z) in enumerate(side_scatter.items()):
            z = np.asarray(z)
            if z.ndim != 2 or z.shape[1] < 2:
                raise ValueError("side_scatter entries must be (N,2+) arrays")
            edge_color = colors.get(lab, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
            vals = None
            if side_values is not None:
                vals = np.asarray(side_values.get(lab, []), dtype=np.float32).ravel()
                if vals.shape[0] != z.shape[0]:
                    raise ValueError("side_values must match side_scatter lengths per split")
            sc = ax_side.scatter(
                z[:, 0],
                z[:, 1],
                s=60,
                alpha=0.85,
                label=lab,
                c=vals,
                cmap=side_cmap if vals is not None else None,
                vmin=vmin,
                vmax=vmax,
                edgecolors=edge_color,
                linewidth=0.7,
                marker=markers.get(lab, "o"),
            )
            if vals is not None:
                mappable = sc
        ax_side.set_xlabel("PC1", fontweight="bold")
        ax_side.set_ylabel("PC2", fontweight="bold")
        ax_side.legend(loc="best", frameon=True, fancybox=True, shadow=True)
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax_side, fraction=0.046, pad=0.04)
            if side_colorbar_label:
                cbar.set_label(side_colorbar_label)
    elif side_mode == "nn_hist":
        if side_values is None:
            raise ValueError("side_values is required when side_mode='nn_hist'")
        for i, lab in enumerate(labels):
            vals = np.asarray(side_values.get(lab, []), dtype=np.float32).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            color = colors.get(lab, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
            ax_side.hist(
                vals,
                bins=30,
                alpha=0.6,
                label=lab,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                density=bool(side_density),
            )
        ax_side.set_xlabel("nn_to_train", fontweight="bold")
        ax_side.set_ylabel("density" if side_density else "count", fontweight="bold")
        if side_log_y:
            ax_side.set_yscale("log")
        ax_side.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    else:
        raise ValueError("side_mode must be 'param_hist', 'solution_pca', or 'nn_hist'")

    if side_title:
        ax_side.set_title(side_title)


def plot_param_points(
    *,
    splits: Mapping[str, np.ndarray],
    param_names: Sequence[str],
    title: str | None = None,
    projection: str = "auto",
    seed: int = 0,
    save_path: Path | str | None = None,
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
) -> Any:
    """Plot param-space distributions for multiple splits.

    - If P==1: overlaid histograms
    - If P==2: scatter in (p0, p1)
    - If P==3 and projection=='3d': 3D scatter in (p0,p1,p2)
    - Otherwise: 2D projection (default PCA) for scatter
    """

    import matplotlib.pyplot as plt

    maybe_apply_style()

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

    labels = list(splits.keys())
    colors = {
        lab: _SPLIT_COLORS.get(lab, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        for i, lab in enumerate(labels)
    }

    fig = plt.figure(
        figsize=(13.0, 6.2) if side_by_side else (7.5, 6.2),
        constrained_layout=True,
    )

    if title:
        title_snake = re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip().lower()).strip("_")
        fig.suptitle(title_snake, fontsize=10, y=1.03)

    use_3d = P == 3 and projection == "3d"
    if use_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        subplot_kwargs: dict[str, Any] = {"projection": "3d"}
    else:
        subplot_kwargs = {}

    ax = (
        fig.add_subplot(1, 2, 1, **subplot_kwargs)
        if side_by_side
        else fig.add_subplot(1, 1, 1, **subplot_kwargs)
    )
    _render_param_panel(ax, splits, arrays, colors, param_names, projection, P)

    if side_by_side:
        ax_side = fig.add_subplot(1, 2, 2)
        _render_side_panel(
            ax_side, fig, labels, colors, splits, arrays, P, param_names,
            side_mode=side_mode,
            side_param_index=side_param_index,
            side_title=side_title,
            side_scatter=side_scatter,
            side_values=side_values,
            side_cmap=side_cmap,
            side_colorbar_label=side_colorbar_label,
            side_density=side_density,
            side_log_y=side_log_y,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both PNG and SVG for publication quality
        fig.savefig(save_path, dpi=300, format='png')
        svg_path = save_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', dpi=300)

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
        fig.suptitle(title, fontweight='bold')

    labels = list(splits.keys())
    for i, lab in enumerate(labels):
        d = np.asarray(splits[lab].get(metric, []), dtype=np.float32)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue

        color = _SPLIT_COLORS.get(lab, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        ax.hist(
            d,
            bins=30,
            alpha=0.6,
            density=bool(density),
            label=lab,
            color=color,
            edgecolor='black',
            linewidth=0.5,
        )
        if show_medians:
            med = float(np.median(d))
            ax.axvline(med, color=color, linestyle="--", linewidth=2.0, alpha=0.9)
            ax.text(
                med,
                0.95,
                f"{lab}\nmed={med:.2f}",
                transform=ax.get_xaxis_transform(),
                rotation=0,
                va="top",
                ha="center",
                fontsize=14,
                fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2)
            )

    ax.set_xlabel(metric, fontweight='bold')
    ax.set_ylabel("Density" if density else "Count", fontweight='bold')
    if log_y:
        ax.set_yscale("log")
    # Legend below plot, horizontal orientation, no shadow
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels),
              frameon=True, fancybox=False, shadow=False, fontsize=14)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both PNG and SVG for publication quality
        fig.savefig(save_path, dpi=300, format='png')
        svg_path = save_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', dpi=300)

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
    plot_style: str = "imshow",
    contour_levels: int = 12,
    contour_line_color: str = "black",
    extent: tuple[float, float, float, float] | None = None,
    show_axes: bool = False,
    colorbar_uniform: bool = False,
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
        # Slightly larger base size for clearer panels and labels.
        figsize=(3.0 * cols, 3.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    if title:
        title_snake = re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip().lower()).strip("_")
        fig.suptitle(title_snake, fontsize=10, y=1.03)

    if plot_style not in {"imshow", "contourf"}:
        raise ValueError("plot_style must be 'imshow' or 'contourf'")

    # Preprocess solutions (handle 4D slicing) so we can optionally compute global
    # vmin/vmax for uniform colorbars across all panels.
    processed: dict[str, np.ndarray] = {}
    for split in split_names:
        sol = np.asarray(solutions_by_split[split])
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
        processed[split] = sol

    vmin = vmax = None
    if colorbar_uniform:
        all_vals = np.concatenate([processed[s].ravel() for s in split_names], axis=0)
        if all_vals.size:
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))

    for r, split in enumerate(split_names):
        sol = processed[split]

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
            field = sol[int(idx[c])]
            if plot_style == "contourf":
                cf = axc.contourf(
                    field,
                    levels=int(contour_levels),
                    origin="lower",
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                )
                axc.contour(
                    field,
                    levels=int(contour_levels),
                    colors=contour_line_color,
                    linewidths=0.22,
                    alpha=0.55,
                    origin="lower",
                    extent=extent,
                )
                # Keep the panel visually square independent of subplot box size.
                # (Colorbars with constrained_layout can otherwise shrink the panel.)
                axc.set_aspect("equal", adjustable="box")
                axc.set_box_aspect(1.0)
                mappable = cf
            else:
                im = axc.imshow(
                    field,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                )
                mappable = im
            if show_axes:
                label_fs = 14  # Keep x and t visually consistent.
                if r == rows - 1:
                    axc.set_xlabel("x", fontsize=label_fs)
                else:
                    axc.set_xticks([])
                if c == 0:
                    # For a single row, show a clean, uniform 't' label.
                    # For multiple rows, keep the split name but use the same font size.
                    y_label = "t" if rows == 1 else f"{split}\n t"
                    axc.set_ylabel(y_label, rotation=90, fontsize=label_fs)
                else:
                    axc.set_yticks([])
            else:
                axc.axis("off")

            # Add a subtle border around contourf panels (even when axis("off")).
            if plot_style == "contourf":
                import matplotlib.patches as mpatches

                border = mpatches.Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    transform=axc.transAxes,
                    fill=False,
                    linewidth=1.0,
                    edgecolor="black",
                    zorder=10,
                )
                axc.add_patch(border)

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

            if plot_style == "contourf":
                # Colorbar outside the panel to avoid covering contours.
                cbar = fig.colorbar(mappable, ax=axc, fraction=0.04, pad=0.02)
            else:
                cbar = fig.colorbar(mappable, ax=axc, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=9, width=0.8)
            cbar.outline.set_linewidth(0.8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both PNG and SVG for publication quality
        fig.savefig(save_path, dpi=300, format='png')
        svg_path = save_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', dpi=300)

    return fig
