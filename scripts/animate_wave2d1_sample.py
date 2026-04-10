#!/usr/bin/env python3
"""Save an animated GIF of the full time sequence for one wave2d1 sample (txy layout)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from eff_physics_learn_dataset import load_pde_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="GIF of u(t,x,y) over all time indices for one sample.")
    ap.add_argument("--equation", "-e", default="wave2d1", help="Dataset directory name (default: wave2d1)")
    ap.add_argument("--data-dir", "-d", default="datasets/new", type=Path, help="Datasets root")
    ap.add_argument(
        "--index",
        "-i",
        type=int,
        default=0,
        help="Sample index in the dataset (default: 0)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("docs/_assets/results/wave2d1/wave2d1_sample_sequence.gif"),
        help="Output .gif path",
    )
    ap.add_argument("--fps", type=float, default=12.0, help="Frames per second (default: 12)")
    ap.add_argument("--dpi", type=int, default=100, help="Figure DPI (default: 100)")
    ap.add_argument(
        "--fig-size",
        type=float,
        nargs=2,
        default=(5.0, 5.0),
        metavar=("W", "H"),
        help="Figure size in inches (default: 5 5)",
    )
    args = ap.parse_args()

    ds = load_pde_dataset(args.equation, data_dir=args.data_dir)
    if ds.field_layout.layout_id != "txy":
        raise SystemExit(f"Expected field layout 'txy', got {ds.field_layout.layout_id!r}")

    u = np.asarray(ds.u[args.index], dtype=np.float64)
    if u.ndim != 3:
        raise SystemExit(f"Expected one sample shaped (n_t, n_x, n_y), got {u.shape}")

    nt, nx, ny = u.shape
    Xg = np.asarray(ds.grids.get("X_grid"))
    Yg = np.asarray(ds.grids.get("Y_grid"))
    if Xg.ndim == 2 and Yg.ndim == 2:
        extent = (float(Xg.min()), float(Xg.max()), float(Yg.min()), float(Yg.max()))
    else:
        extent = None

    vmax = float(np.max(np.abs(u)))
    if vmax <= 0:
        vmax = 1.0
    vmin, vmax = -vmax, vmax

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(args.fig_size), layout="constrained")
    im_kw: dict = {
        "origin": "lower",
        "aspect": "equal",
        "cmap": "RdBu_r",
        "vmin": vmin,
        "vmax": vmax,
    }
    if extent is not None:
        im_kw["extent"] = extent
    im = ax.imshow(u[0], **im_kw)
    if extent is None:
        ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("u")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{args.equation} sample {args.index}  (n_t={nt})")

    def update(frame: int):
        im.set_array(u[frame])
        ax.set_title(f"{args.equation} sample {args.index}  t-index {frame}/{nt - 1}")
        return (im,)

    ani = FuncAnimation(fig, update, frames=nt, interval=1000.0 / max(args.fps, 0.1), blit=False)
    writer = PillowWriter(fps=args.fps)
    ani.save(args.out, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote: {args.out} ({nt} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
