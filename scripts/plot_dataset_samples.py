#!/usr/bin/env python3

import argparse
from pathlib import Path

from eff_physics_learn_dataset.datasets import load_pde_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a 25-sample plot for a dataset split.")
    ap.add_argument("--equation", "-e", required=True, help="Dataset directory name (e.g. helmholtz2D)")
    ap.add_argument("--data-dir", "-d", default="datasets", help="Datasets root directory")
    ap.add_argument("--split", "-s", default="test", choices=["test", "all"], help="Which split to plot")
    ap.add_argument("--n", type=int, default=25, help="Number of samples to plot")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for selecting samples")
    ap.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="For 2D+1 datasets (N,S,H,W), which slice/time index to plot (default: middle).",
    )
    ap.add_argument(
        "--slice-axis",
        type=int,
        default=1,
        help="For 4D solutions, which axis is the slice/time axis (default: 1).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: docs/_assets/results/{equation})",
    )
    ap.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output file path (overrides --out-dir).",
    )
    args = ap.parse_args()

    ds = load_pde_dataset(args.equation, data_dir=Path(args.data_dir))
    out = args.out
    if out is None:
        base = Path(args.out_dir) if args.out_dir is not None else Path("docs/_assets/results") / args.equation
        out = base / f"{args.equation}_{args.split}{args.n}.png"

    ds.plot_samples(
        split=args.split,
        n=args.n,
        seed=args.seed,
        slice_index=args.slice_index,
        slice_axis=args.slice_axis,
        save_path=Path(out),
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()


