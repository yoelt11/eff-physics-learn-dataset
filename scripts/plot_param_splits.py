#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np

from eff_physics_learn_dataset.datasets import load_pde_dataset
from eff_physics_learn_dataset.datasets.solution_similarity import fit_solution_pca, vectorize_solutions


def _to_snake_case(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot parameter-space distributions for dataset splits.")
    ap.add_argument("--equation", "-e", required=True, help="Dataset directory name (e.g. helmholtz2D)")
    ap.add_argument("--data-dir", "-d", default="datasets", help="Datasets root directory")
    ap.add_argument(
        "--mode",
        "-m",
        default="train_test",
        choices=["train_test", "parametric"],
        help="Which split mode to visualize",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for selecting train budget/few-shot set")
    ap.add_argument("--n-train", type=int, default=25, help="Training samples for the budget / few-shot set")
    ap.add_argument(
        "--balance",
        action="store_true",
        help="(parametric mode) Balance interp/extrap by subsampling both to equal size.",
    )
    ap.add_argument(
        "--n-each",
        type=int,
        default=None,
        help="(parametric mode) Target samples per split when using --balance (caps if insufficient).",
    )
    ap.add_argument(
        "--balance-strategy",
        default="random",
        choices=["random", "solution_nn"],
        help="(parametric mode) How to pick balanced subsets: random or solution_nn (closest interp / farthest extrap).",
    )
    ap.add_argument(
        "--method",
        default="solution_percentile",
        choices=["solution_percentile", "convex_hull"],
        help="(parametric mode) Splitting method: solution_percentile (default, robust) or convex_hull (legacy).",
    )
    ap.add_argument(
        "--percentile",
        type=float,
        default=50.0,
        help="(parametric mode, solution_percentile) Distance percentile threshold (default: 50.0).",
    )
    ap.add_argument(
        "--projection",
        default="auto",
        choices=["auto", "raw", "pca", "3d"],
        help="How to project params when P>2 (default: PCA)",
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

    if args.mode == "train_test":
        splits = ds.train_test_splits(seed=args.seed, n_train=args.n_train)
        # only plot train/test (not train_rest)
        to_plot = {"train": splits["train"], "test": splits["test"]}
        title = _to_snake_case(
            f"{args.equation} train/test params (n_train={args.n_train}, seed={args.seed})"
        )
    else:
        ps = ds.parametric_splits(
            seed=args.seed,
            n_train=args.n_train,
            method=args.method,
            percentile=args.percentile,
            balance=bool(args.balance),
            n_each=args.n_each,
            balance_strategy=args.balance_strategy,
        )
        to_plot = {"train_few": ps["train_few"], "interp": ps["interp"], "extrap": ps["extrap"]}
        title = _to_snake_case(
            f"{args.equation} parametric splits (n_train={args.n_train}, seed={args.seed})"
        )

    out = args.out
    if out is None:
        base = Path(args.out_dir) if args.out_dir is not None else Path("docs/_assets/results") / args.equation
        out = base / f"{args.equation}_{args.mode}_params.png"

    split_indices = {name: ds_view.indices for name, ds_view in to_plot.items()}
    all_idx = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in split_indices.values()]))
    X_all = vectorize_solutions(ds.u[all_idx])
    row = {int(i): j for j, i in enumerate(all_idx.tolist())}
    train_key = "train_few" if "train_few" in split_indices else "train"
    train_idx = np.asarray(split_indices[train_key], dtype=np.int64)
    X_train = vectorize_solutions(ds.u[train_idx])
    model, _ = fit_solution_pca(X_train, n_components=2)
    Z_all = model.transform(X_all)
    train_center = Z_all[np.asarray([row[int(i)] for i in train_idx], dtype=np.int64)].mean(axis=0)
    side_scatter = {
        name: Z_all[np.asarray([row[int(i)] for i in idx], dtype=np.int64)]
        for name, idx in split_indices.items()
    }
    side_values = {
        name: np.linalg.norm(z - train_center[None, :], axis=1)
        for name, z in side_scatter.items()
    }
    side_title = _to_snake_case(f"{args.equation} {args.mode} solution_pca_train_anchor")
    ds.plot_param_splits(
        splits=to_plot,
        projection=args.projection,
        save_path=Path(out),
        title=title,
        side_by_side=True,
        side_title=side_title,
        side_mode="solution_pca",
        side_scatter=side_scatter,
        side_values=side_values,
        side_cmap="magma",
        side_colorbar_label="distance_to_train_centroid",
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()


