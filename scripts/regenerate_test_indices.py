#!/usr/bin/env python3
"""Regenerate test_indices.pkl files with a fixed test size across all datasets."""

import argparse
import pickle
from pathlib import Path

import numpy as np

from eff_physics_learn_dataset.datasets import load_pde_dataset


def regenerate_test_indices(
    equation: str,
    data_dir: Path,
    n_test: int = 40,
    seed: int = 42,
) -> None:
    """Regenerate test_indices.pkl with a fixed test size.
    
    Args:
        equation: Equation name
        data_dir: Base data directory
        n_test: Fixed number of test samples (default: 40)
        seed: Random seed for reproducibility
    """
    ds = load_pde_dataset(equation, data_dir=data_dir)
    total_samples = len(ds)
    
    # Use fixed size for all datasets
    actual_n_test = n_test
    
    if actual_n_test > total_samples:
        raise ValueError(
            f"{equation} only has {total_samples} samples, cannot create {n_test} test samples. "
            f"Please regenerate the dataset with more samples first."
        )
    
    if actual_n_test > total_samples // 5:
        print(f"Warning: {equation} has {total_samples} samples. "
              f"Using {actual_n_test} test samples ({actual_n_test/total_samples*100:.1f}%) "
              f"instead of the typical 20%.")
    
    # Generate test indices with fixed seed
    rng = np.random.default_rng(seed + 1000)
    test_indices = rng.choice(total_samples, size=actual_n_test, replace=False)
    test_indices = np.sort(test_indices).tolist()
    
    # Save to ground_truth directory
    test_path = data_dir / equation / "ground_truth" / "test_indices.pkl"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_path, "wb") as f:
        pickle.dump(test_indices, f)
    
    print(f"✓ Regenerated test indices for {equation}: {len(test_indices)} samples")
    print(f"  Total: {total_samples}, Test: {len(test_indices)}, Train pool: {total_samples - len(test_indices)}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate test_indices.pkl files with fixed test size"
    )
    parser.add_argument(
        "--equation",
        "-e",
        choices=["allen_cahn", "burgers", "convection", "helmholtz2D", "all"],
        default="all",
        help="Equation to process (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("datasets"),
        help="Base data directory (default: datasets)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=40,
        help="Fixed number of test samples (default: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    equations = (
        ["allen_cahn", "burgers", "convection", "helmholtz2D"]
        if args.equation == "all"
        else [args.equation]
    )
    
    print("=" * 60)
    print("Regenerating Test Indices with Fixed Size")
    print("=" * 60)
    print(f"Target test size: {args.n_test}")
    print(f"Seed: {args.seed}")
    print()
    
    for eq in equations:
        try:
            regenerate_test_indices(
                equation=eq,
                data_dir=args.data_dir,
                n_test=args.n_test,
                seed=args.seed,
            )
        except Exception as e:
            print(f"✗ Error processing {eq}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    for eq in equations:
        try:
            ds = load_pde_dataset(eq, data_dir=args.data_dir)
            splits = ds.train_test_splits(seed=0, n_train=25)
            print(f"{eq}: Total={len(ds)}, Train={len(splits['train'])}, Test={len(splits['test'])}")
        except Exception as e:
            print(f"{eq}: Error - {e}")


if __name__ == "__main__":
    main()
