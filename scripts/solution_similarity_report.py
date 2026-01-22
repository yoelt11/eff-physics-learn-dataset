#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from eff_physics_learn_dataset.datasets import load_pde_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute solution-space similarity report for dataset splits.")
    ap.add_argument("--equation", "-e", required=True, help="Dataset directory name (e.g. helmholtz2D)")
    ap.add_argument("--data-dir", "-d", default="datasets", help="Datasets root directory")
    ap.add_argument(
        "--mode",
        "-m",
        default="parametric",
        choices=["train_test", "parametric"],
        help="Which splits to analyze",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for selecting train budget/few-shot set")
    ap.add_argument("--n-train", type=int, default=10, help="Training samples for the budget / few-shot set")
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
        "--diversify",
        action="store_true",
        help="(parametric mode) When using solution_nn balancing, pick a diverse subset within the top candidates.",
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
    ap.add_argument("--n-components", type=int, default=5, help="PCA components in solution space")
    ap.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="For 4D solutions, which slice index to use when vectorizing (default: middle).",
    )
    ap.add_argument(
        "--slice-axis",
        type=int,
        default=1,
        help="For 4D solutions, which axis is the slice axis (default: 1).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Write report JSON (overrides --out-dir).",
    )
    ap.add_argument(
        "--out-plot",
        default=None,
        help="Write histogram PNG (overrides --out-dir).",
    )
    ap.add_argument(
        "--out-rows",
        default=None,
        help="Write row-compare PNG (overrides --out-dir).",
    )
    ap.add_argument(
        "--rows-n",
        type=int,
        default=5,
        help="Number of samples per row for the row-compare image (default: 5).",
    )
    ap.add_argument(
        "--row-plot-style",
        default="imshow",
        choices=["imshow", "contourf"],
        help="Row plot style for solutions (default: imshow).",
    )
    ap.add_argument(
        "--row-contour-levels",
        type=int,
        default=12,
        help="Contour levels when using --row-plot-style contourf.",
    )
    ap.add_argument(
        "--row-contour-color",
        default="black",
        help="Contour line color when using --row-plot-style contourf.",
    )
    ap.add_argument(
        "--row-show-axes",
        action="store_true",
        help="Show x/t axes on row plots (default: off).",
    )
    ap.add_argument(
        "--out-params",
        default=None,
        help="Write parameter scatter plot PNG (overrides --out-dir).",
    )
    ap.add_argument(
        "--param-projection",
        default="auto",
        choices=["auto", "raw", "pca", "3d"],
        help="How to plot params when P>2 (use 3d only when P==3).",
    )
    ap.add_argument(
        "--param-nn-density",
        action="store_true",
        help="(param scatter) Plot nn-to-train histogram as density.",
    )
    ap.add_argument(
        "--param-nn-log-y",
        action="store_true",
        help="(param scatter) Use log y-scale for nn-to-train histogram.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for all artifacts (default: docs/_assets/results/{equation}). Individual --out-* overrides this.",
    )
    ap.add_argument(
        "--density",
        action="store_true",
        help="Plot density-normalized histograms (can hide curves if distributions are very peaked).",
    )
    ap.add_argument(
        "--log-y",
        action="store_true",
        help="Use log y-scale for histogram.",
    )
    args = ap.parse_args()

    ds = load_pde_dataset(args.equation, data_dir=Path(args.data_dir))
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path("docs/_assets/results") / args.equation
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train_test":
        splits = ds.train_test_splits(seed=args.seed, n_train=args.n_train)
        to_use = {"train": splits["train"], "test": splits["test"]}
        train_key = "train"
        rows_splits = {"train": splits["train"], "test": splits["test"]}
    else:
        ps = ds.parametric_splits(
            seed=args.seed,
            n_train=args.n_train,
            method=args.method,
            percentile=args.percentile,
            balance=bool(args.balance),
            n_each=args.n_each,
            balance_strategy=args.balance_strategy,
            diversify=bool(args.diversify),
        )
        to_use = {"train_few": ps["train_few"], "interp": ps["interp"], "extrap": ps["extrap"]}
        train_key = "train_few"
        rows_splits = {"train_few": ps["train_few"], "interp": ps["interp"], "extrap": ps["extrap"]}

    report = ds.solution_similarity_report(
        splits=to_use,
        train_key=train_key,
        n_components=args.n_components,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
    )

    out_json = args.out_json
    if out_json is None:
        out_json = out_dir / f"{args.equation}_{args.mode}_solution_similarity.json"
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON smaller: keep summary stats, drop raw arrays by default
    compact = {
        "train_key": report["train_key"],
        "n_components": report["n_components"],
        "explained_variance_ratio": report["explained_variance_ratio"].tolist(),
        "splits": {
            k: {kk: vv for kk, vv in v.items() if not kk.endswith("_to_train") and not kk.startswith("dist_")}
            | {
                "nn_to_train_mean": v["nn_to_train_mean"],
                "nn_to_train_median": v["nn_to_train_median"],
                "dist_to_train_centroid_mean": v["dist_to_train_centroid_mean"],
                "dist_to_train_centroid_median": v["dist_to_train_centroid_median"],
            }
            for k, v in report["splits"].items()
        },
    }
    out_json.write_text(json.dumps(compact, indent=2))

    out_plot = args.out_plot
    if out_plot is None:
        out_plot = out_dir / f"{args.equation}_{args.mode}_solution_similarity.png"
    ds.plot_solution_similarity(
        report=report,
        metric="nn_to_train",
        save_path=Path(out_plot),
        title=f"{args.equation} {args.mode} solution similarity (nn_to_{train_key})",
        density=bool(args.density),
        log_y=bool(args.log_y),
    )

    out_rows = args.out_rows
    if out_rows is None:
        out_rows = out_dir / f"{args.equation}_{args.mode}_solution_rows.png"
    ds.plot_split_solution_rows(
        splits=rows_splits,
        n_per_row=int(args.rows_n),
        seed=int(args.seed),
        slice_axis=int(args.slice_axis),
        slice_index=args.slice_index,
        plot_style=args.row_plot_style,
        contour_levels=int(args.row_contour_levels),
        contour_line_color=args.row_contour_color,
        show_axes=bool(args.row_show_axes),
        save_path=Path(out_rows),
        title=f"{args.equation}: {', '.join(rows_splits.keys())} ({int(args.rows_n)} samples each)",
    )

    out_params = args.out_params
    if out_params is None:
        out_params = out_dir / f"{args.equation}_{args.mode}_param_scatter.png"
    # Use same splits as the row plot
    nn_values = {
        name: np.asarray(report["splits"][name]["nn_to_train"], dtype=np.float32)
        for name in rows_splits.keys()
    }
    ds.plot_param_splits(
        splits=rows_splits,
        projection=args.param_projection,
        save_path=Path(out_params),
        title=f"{args.equation} {args.mode} param distribution",
        seed=int(args.seed),
        side_by_side=True,
        side_mode="nn_hist",
        side_values=nn_values,
        side_title="solution_space_nn_to_train",
        side_density=bool(args.param_nn_density),
        side_log_y=bool(args.param_nn_log_y),
    )

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_plot}")
    print(f"Wrote: {out_rows}")
    print(f"Wrote: {out_params}")


if __name__ == "__main__":
    main()


