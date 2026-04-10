#!/usr/bin/env python3
"""Generate parametric split indices for each dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import importlib
import sys
import types


def _load_pde_dataset():
    """Load load_pde_dataset without importing top-level package __init__."""
    root = Path(__file__).resolve().parents[1] / "src" / "eff_physics_learn_dataset"
    pkg_name = "eff_physics_learn_dataset"
    subpkg_name = f"{pkg_name}.datasets"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root)]
        sys.modules[pkg_name] = pkg

    if subpkg_name not in sys.modules:
        subpkg = types.ModuleType(subpkg_name)
        subpkg.__path__ = [str(root / "datasets")]
        sys.modules[subpkg_name] = subpkg

    module = importlib.import_module(f"{subpkg_name}.pde_dataset")
    return module.load_pde_dataset


def _join_indices(indices: list[int] | tuple[int, ...]) -> str:
    return " ".join(str(int(i)) for i in indices)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write parametric split indices to a CSV-like file.")
    ap.add_argument("--data-dir", default="datasets", help="Datasets root directory")
    ap.add_argument(
        "--equations",
        nargs="+",
        default=["allen_cahn", "burgers", "convection", "helmholtz2D"],
        help="Dataset names to include",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to include")
    ap.add_argument("--n-train", type=int, default=10, help="Few-shot training samples")
    ap.add_argument("--n-each", type=int, default=20, help="Balanced samples per split")
    ap.add_argument("--method", default="solution_percentile", help="Parametric split method")
    ap.add_argument("--percentile", type=float, default=50.0, help="Solution-space percentile threshold")
    ap.add_argument(
        "--out",
        default="datasets/parametric_split_indices.csv",
        help="Output file path",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    lines = ["equation,seed,train_indices,interp_indices,extrap_indices"]
    for equation in args.equations:
        load_pde_dataset = _load_pde_dataset()
        ds = load_pde_dataset(equation, data_dir=data_dir)
        for seed in args.seeds:
            splits = ds.parametric_splits(
                seed=int(seed),
                n_train=int(args.n_train),
                method=str(args.method),
                percentile=float(args.percentile),
                balance=True,
                n_each=int(args.n_each),
                balance_strategy="random",
            )
            train_idx = _join_indices(splits["train_few"].indices)
            interp_idx = _join_indices(splits["interp"].indices)
            extrap_idx = _join_indices(splits["extrap"].indices)
            lines.append(f"{equation},{seed},{train_idx},{interp_idx},{extrap_idx}")

    out_path.write_text("\n".join(lines))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
