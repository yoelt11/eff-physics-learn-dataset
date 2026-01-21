#!/usr/bin/env python3
"""Validate generated datasets - verify they load correctly and match expected structure."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eff_physics_learn_dataset.datasets import load_pde_dataset


def validate_dataset(equation: str, data_dir: Path, subfolder: str = "ground_truth_regenerated") -> dict:
    """Validate a generated dataset.

    Args:
        equation: Equation name
        data_dir: Base data directory
        subfolder: Subfolder name (e.g., "ground_truth" or "ground_truth_regenerated")

    Returns:
        Validation results dict
    """
    print(f"\n{'='*60}")
    print(f"Validating {equation} ({subfolder})")
    print(f"{'='*60}")

    # Check if dataset directory exists
    dataset_dir = data_dir / equation / subfolder
    if not dataset_dir.exists():
        return {"success": False, "error": f"Directory not found: {dataset_dir}"}

    pkl_file = dataset_dir / f"{equation}_dataset.pkl"
    test_indices_file = dataset_dir / "test_indices.pkl"

    if not pkl_file.exists():
        return {"success": False, "error": f"Dataset pickle not found: {pkl_file}"}

    if not test_indices_file.exists():
        return {"success": False, "error": f"Test indices not found: {test_indices_file}"}

    try:
        # Try to load with the dataset loader (temporarily rename to ground_truth for loading)
        if subfolder != "ground_truth":
            # Temporarily rename for loading
            orig_dir = data_dir / equation / "ground_truth"
            backup_dir = data_dir / equation / "ground_truth_backup_temp"

            # Backup original if it exists
            if orig_dir.exists():
                orig_dir.rename(backup_dir)

            # Rename regenerated to ground_truth temporarily
            dataset_dir.rename(orig_dir)

            try:
                ds = load_pde_dataset(equation, data_dir=data_dir)
            finally:
                # Restore names
                orig_dir.rename(dataset_dir)
                if backup_dir.exists():
                    backup_dir.rename(orig_dir)
        else:
            ds = load_pde_dataset(equation, data_dir=data_dir)

        # Validation checks
        checks = {}

        # Check solution shape
        checks["solution_shape"] = ds.u.shape
        checks["n_samples"] = len(ds)
        checks["grid_size"] = ds.u.shape[1:]

        # Check parameters
        checks["param_names"] = ds.param_names
        checks["param_shape"] = ds.params.shape

        # Check grids
        checks["grid_keys"] = list(ds.grids.keys())

        # Check test split loads
        try:
            test_idx = ds._load_test_indices()
            checks["n_test"] = len(test_idx)
            checks["test_indices_valid"] = all(0 <= i < len(ds) for i in test_idx)
        except Exception as e:
            checks["test_indices_error"] = str(e)

        # Check splits work
        try:
            splits = ds.budget_split("low", seed=0)
            checks["budget_split_works"] = True
            checks["train_size"] = len(splits["train"])
            checks["test_size"] = len(splits["test"])
        except Exception as e:
            checks["budget_split_works"] = False
            checks["budget_split_error"] = str(e)

        # Check parametric splits work
        try:
            ps = ds.parametric_splits(seed=0, n_train=10)
            checks["parametric_split_works"] = True
            checks["train_few_size"] = len(ps["train_few"])
            checks["interp_size"] = len(ps["interp"])
            checks["extrap_size"] = len(ps["extrap"])
        except Exception as e:
            checks["parametric_split_works"] = False
            checks["parametric_split_error"] = str(e)

        # Check sample access
        try:
            sample = ds[0]
            checks["sample_keys"] = list(sample.keys())
            checks["sample_access_works"] = True
        except Exception as e:
            checks["sample_access_works"] = False
            checks["sample_access_error"] = str(e)

        # Solution statistics
        checks["solution_min"] = float(ds.u.min())
        checks["solution_max"] = float(ds.u.max())
        checks["solution_mean"] = float(ds.u.mean())
        checks["solution_std"] = float(ds.u.std())

        # Parameter statistics
        param_stats = {}
        for i, name in enumerate(ds.param_names):
            vals = ds.params[:, i]
            param_stats[name] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
        checks["param_stats"] = param_stats

        # Print results
        print(f"✓ Dataset loaded successfully")
        print(f"  Samples: {checks['n_samples']}")
        print(f"  Shape: {checks['solution_shape']}")
        print(f"  Parameters: {', '.join(checks['param_names'])}")
        print(f"  Test split: {checks['n_test']} samples")
        print(f"  Budget split: {'✓' if checks['budget_split_works'] else '✗'}")
        print(f"  Parametric split: {'✓' if checks['parametric_split_works'] else '✗'}")
        print(f"  Solution range: [{checks['solution_min']:.4f}, {checks['solution_max']:.4f}]")

        print(f"\n  Parameter ranges:")
        for name, stats in param_stats.items():
            print(f"    {name:8s}: [{stats['min']:.6f}, {stats['max']:.6f}]")

        return {"success": True, **checks}

    except Exception as e:
        import traceback
        error_msg = f"Failed to load: {e}\n{traceback.format_exc()}"
        print(f"✗ {error_msg}")
        return {"success": False, "error": error_msg}


def main():
    parser = argparse.ArgumentParser(description="Validate generated datasets")
    parser.add_argument(
        "-e", "--equation",
        nargs="+",
        default=["allen_cahn", "burgers", "convection", "helmholtz2D"],
        help="Equation(s) to validate",
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default=Path("datasets"),
        help="Base data directory",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="ground_truth_regenerated",
        help="Subfolder name (default: ground_truth_regenerated)",
    )

    args = parser.parse_args()

    results = {}
    for eq in args.equation:
        result = validate_dataset(eq, args.data_dir, args.subfolder)
        results[eq] = result

    # Summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")

    all_success = all(r["success"] for r in results.values())

    for eq, result in results.items():
        if result["success"]:
            print(f"  ✓ {eq:15s}: {result['n_samples']} samples, {result['param_names']}")
        else:
            print(f"  ✗ {eq:15s}: {result['error']}")

    if all_success:
        print(f"\n✓ All datasets validated successfully!")
    else:
        print(f"\n✗ Some datasets failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
