#!/usr/bin/env python3
"""Script to verify and switch between original and regenerated datasets.

Usage:
    # Check status
    python scripts/switch_datasets.py --status

    # Verify regenerated datasets are compatible
    python scripts/switch_datasets.py --verify

    # Backup original and use regenerated
    python scripts/switch_datasets.py --use-regenerated

    # Restore original datasets
    python scripts/switch_datasets.py --restore-original
"""

import argparse
import pickle
import shutil
from pathlib import Path

from eff_physics_learn_dataset.datasets import load_pde_dataset


EQUATIONS = ["allen_cahn", "burgers", "convection", "helmholtz2D"]
DATA_DIR = Path("datasets")


def check_status():
    """Check which datasets exist."""
    print("=" * 60)
    print("Dataset Status")
    print("=" * 60)

    for eq in EQUATIONS:
        print(f"\n{eq}:")

        # Check original
        orig_dir = DATA_DIR / eq / "ground_truth"
        orig_backup = DATA_DIR / eq / "ground_truth_original"
        regen_dir = DATA_DIR / eq / "ground_truth_regenerated"

        if orig_dir.exists():
            dataset_file = list(orig_dir.glob("*_dataset.pkl"))
            if dataset_file:
                with open(dataset_file[0], "rb") as f:
                    data = pickle.load(f)
                n_samples = data["solutions"].shape[0]
                print(f"  ✓ Original active: {n_samples} samples")
            else:
                print(f"  ✗ Original directory exists but no dataset file")
        else:
            print(f"  ✗ No original dataset")

        if orig_backup.exists():
            print(f"  ✓ Backup exists (original preserved)")

        if regen_dir.exists():
            dataset_file = list(regen_dir.glob("*_dataset.pkl"))
            if dataset_file:
                with open(dataset_file[0], "rb") as f:
                    data = pickle.load(f)
                n_samples = data["solutions"].shape[0]
                print(f"  ✓ Regenerated available: {n_samples} samples")
            else:
                print(f"  ✗ Regenerated directory exists but no dataset file")
        else:
            print(f"  ✗ No regenerated dataset")


def verify_compatibility():
    """Verify regenerated datasets are compatible with API."""
    print("=" * 60)
    print("Verifying Dataset Compatibility")
    print("=" * 60)

    for eq in EQUATIONS:
        print(f"\n{eq}:")

        regen_dir = DATA_DIR / eq / "ground_truth_regenerated"
        if not regen_dir.exists():
            print(f"  ✗ No regenerated dataset found")
            continue

        dataset_file = list(regen_dir.glob("*_dataset.pkl"))
        if not dataset_file:
            print(f"  ✗ No dataset file in regenerated directory")
            continue

        try:
            # Load and check structure
            with open(dataset_file[0], "rb") as f:
                data = pickle.load(f)

            required_keys = ["solutions", "pde_params", "grid_info", "metadata"]
            missing_keys = [k for k in required_keys if k not in data]

            if missing_keys:
                print(f"  ✗ Missing keys: {missing_keys}")
                continue

            n_samples = data["solutions"].shape[0]
            print(f"  ✓ Structure valid: {n_samples} samples")
            print(f"    - Solution shape: {data['solutions'].shape}")
            print(f"    - Parameters: {list(data['pde_params'].keys())}")

            # Check test indices
            test_indices_file = regen_dir / "test_indices.pkl"
            if test_indices_file.exists():
                with open(test_indices_file, "rb") as f:
                    test_indices = pickle.load(f)
                print(f"    - Test indices: {len(test_indices)} samples")
            else:
                print(f"    ✗ No test_indices.pkl found")

        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")


def use_regenerated():
    """Backup original and activate regenerated datasets."""
    print("=" * 60)
    print("Activating Regenerated Datasets")
    print("=" * 60)

    for eq in EQUATIONS:
        print(f"\n{eq}:")

        orig_dir = DATA_DIR / eq / "ground_truth"
        orig_backup = DATA_DIR / eq / "ground_truth_original"
        regen_dir = DATA_DIR / eq / "ground_truth_regenerated"

        if not regen_dir.exists():
            print(f"  ✗ No regenerated dataset found, skipping")
            continue

        # Backup original if it exists and hasn't been backed up
        if orig_dir.exists() and not orig_backup.exists():
            print(f"  → Backing up original to ground_truth_original/")
            shutil.move(str(orig_dir), str(orig_backup))
        elif orig_dir.exists() and orig_backup.exists():
            print(f"  → Removing current ground_truth/ (backup exists)")
            shutil.rmtree(orig_dir)

        # Move regenerated to ground_truth
        print(f"  → Activating regenerated dataset")
        shutil.move(str(regen_dir), str(orig_dir))
        print(f"  ✓ Done")

    print("\n" + "=" * 60)
    print("Regenerated datasets are now active!")
    print("Original datasets backed up to ground_truth_original/")
    print("=" * 60)


def restore_original():
    """Restore original datasets from backup."""
    print("=" * 60)
    print("Restoring Original Datasets")
    print("=" * 60)

    for eq in EQUATIONS:
        print(f"\n{eq}:")

        orig_dir = DATA_DIR / eq / "ground_truth"
        orig_backup = DATA_DIR / eq / "ground_truth_original"
        regen_backup = DATA_DIR / eq / "ground_truth_regenerated"

        if not orig_backup.exists():
            print(f"  ✗ No backup found, skipping")
            continue

        # Move current ground_truth to regenerated (if it's the regenerated version)
        if orig_dir.exists():
            if not regen_backup.exists():
                print(f"  → Saving current dataset to ground_truth_regenerated/")
                shutil.move(str(orig_dir), str(regen_backup))
            else:
                print(f"  → Removing current ground_truth/")
                shutil.rmtree(orig_dir)

        # Restore original
        print(f"  → Restoring original from backup")
        shutil.move(str(orig_backup), str(orig_dir))
        print(f"  ✓ Done")

    print("\n" + "=" * 60)
    print("Original datasets restored!")
    print("Regenerated datasets saved to ground_truth_regenerated/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Manage dataset versions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Show dataset status")
    group.add_argument("--verify", action="store_true", help="Verify regenerated datasets")
    group.add_argument("--use-regenerated", action="store_true", help="Activate regenerated datasets")
    group.add_argument("--restore-original", action="store_true", help="Restore original datasets")

    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.verify:
        verify_compatibility()
    elif args.use_regenerated:
        use_regenerated()
    elif args.restore_original:
        restore_original()


if __name__ == "__main__":
    main()
