#!/usr/bin/env python3

import argparse
from pathlib import Path

from eff_physics_learn_dataset.datasets import load_pde_dataset


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
        title = f"{args.equation} train/test params (n_train={args.n_train}, seed={args.seed})"
    else:
        ps = ds.parametric_splits(
            seed=args.seed,
            n_train=args.n_train,
            balance=bool(args.balance),
            n_each=args.n_each,
            balance_strategy=args.balance_strategy,
        )
        to_plot = {"train_few": ps["train_few"], "interp": ps["interp"], "extrap": ps["extrap"]}
        title = f"{args.equation} parametric splits (n_train={args.n_train}, seed={args.seed})"

    out = args.out
    if out is None:
        base = Path(args.out_dir) if args.out_dir is not None else Path("docs/_assets/results") / args.equation
        out = base / f"{args.equation}_{args.mode}_params.png"

    ds.plot_param_splits(splits=to_plot, projection=args.projection, save_path=Path(out), title=title)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()


