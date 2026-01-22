#!/usr/bin/env python3
"""Generate a comprehensive summary markdown file for all datasets."""

import json
from pathlib import Path
from typing import Any, Dict

from eff_physics_learn_dataset.datasets import load_pde_dataset


def get_param_range_str(param_names: tuple[str, ...], params: Any) -> str:
    """Format parameter ranges as a string."""
    ranges = []
    for name in param_names:
        p_min = float(params[:, param_names.index(name)].min())
        p_max = float(params[:, param_names.index(name)].max())
        ranges.append(f"{name}: [{p_min:.3f}, {p_max:.3f}]")
    return "<br>".join(ranges)


def get_standard_splits_info(ds: Any) -> Dict[str, Any]:
    """Get standard train/test split information."""
    splits = ds.train_test_splits(seed=0, n_train=25)  # Standard budget
    return {
        "n_train": len(splits["train"]),
        "n_test": len(splits["test"]),
    }


def get_parametric_splits_info(
    ds: Any, seed: int, n_train: int = 10, n_each: int = 20, method: str = "solution_percentile"
) -> Dict[str, Any]:
    """Get parametric split information and distances."""
    ps = ds.parametric_splits(
        seed=seed,
        n_train=n_train,
        method=method,
        balance=True,
        n_each=n_each,
        balance_strategy="random",
    )
    
    to_use = {
        "train_few": ps["train_few"],
        "interp": ps["interp"],
        "extrap": ps["extrap"],
    }
    
    report = ds.solution_similarity_report(
        splits=to_use,
        train_key="train_few",
    )
    
    return {
        "n_interp": len(ps["interp"]),
        "n_extrap": len(ps["extrap"]),
        "interp_mean": report["splits"]["interp"]["nn_to_train_mean"],
        "extrap_mean": report["splits"]["extrap"]["nn_to_train_mean"],
    }


def main():
    data_dir = Path("datasets")
    equations = ["allen_cahn", "burgers", "convection", "helmholtz2D"]
    
    # Collect data for all equations
    parametric_data = []
    standard_data = []
    
    for eq in equations:
        try:
            print(f"Processing {eq}...")
            ds = load_pde_dataset(eq, data_dir=data_dir)
            
            # Get parameter ranges
            param_range = get_param_range_str(ds.param_names, ds.params)
            
            # Standard splits info
            std_info = get_standard_splits_info(ds)
            standard_data.append({
                "equation": eq,
                "param_range": param_range,
                "n_train": std_info["n_train"],
                "n_test": std_info["n_test"],
            })
            
            # Parametric splits info for each seed
            seed_data = {}
            for seed in [0, 1, 2]:
                try:
                    info = get_parametric_splits_info(ds, seed=seed, n_train=10, n_each=20)
                    seed_data[f"seed_{seed}_interp"] = info["interp_mean"]
                    seed_data[f"seed_{seed}_extrap"] = info["extrap_mean"]
                    if seed == 0:
                        seed_data["n_interp"] = info["n_interp"]
                        seed_data["n_extrap"] = info["n_extrap"]
                except Exception as e:
                    print(f"  Warning: Failed to process {eq} seed {seed}: {e}")
                    seed_data[f"seed_{seed}_interp"] = None
                    seed_data[f"seed_{seed}_extrap"] = None
            
            parametric_data.append({
                "equation": eq,
                "param_range": param_range,
                "n_train": 10,
                **seed_data,
            })
            
        except Exception as e:
            print(f"Error processing {eq}: {e}")
            continue
    
    # Generate markdown
    md_lines = [
        "# Dataset Summary",
        "",
        "## Parametric Splits Summary",
        "",
        "This table shows parametric interpolation/extrapolation splits with balanced sampling.",
        "**Method**: `solution_percentile` (default) - splits by distance in solution-space PCA.",
        "Distances show mean nearest-neighbor distance to training samples in solution space.",
        "",
        "| Equation | Parameter Range | n_train | n_interp | n_extrap | Seed 0 Interp | Seed 0 Extrap | Seed 0 Ratio | Seed 1 Interp | Seed 1 Extrap | Seed 1 Ratio | Seed 2 Interp | Seed 2 Extrap | Seed 2 Ratio |",
        "|----------|----------------|---------|----------|----------|---------------|---------------|--------------|---------------|---------------|--------------|---------------|---------------|--------------|",
    ]
    
    for data in parametric_data:
        # Compute ratios
        ratio_0 = data.get('seed_0_extrap', 0) / data.get('seed_0_interp', 1) if data.get('seed_0_interp') else None
        ratio_1 = data.get('seed_1_extrap', 0) / data.get('seed_1_interp', 1) if data.get('seed_1_interp') else None
        ratio_2 = data.get('seed_2_extrap', 0) / data.get('seed_2_interp', 1) if data.get('seed_2_interp') else None

        row = [
            data["equation"],
            data["param_range"],
            str(data["n_train"]),
            str(data.get("n_interp", "N/A")),
            str(data.get("n_extrap", "N/A")),
            f"{data.get('seed_0_interp', 'N/A'):.2f}" if data.get('seed_0_interp') is not None else "N/A",
            f"{data.get('seed_0_extrap', 'N/A'):.2f}" if data.get('seed_0_extrap') is not None else "N/A",
            f"{ratio_0:.2f}x" if ratio_0 is not None else "N/A",
            f"{data.get('seed_1_interp', 'N/A'):.2f}" if data.get('seed_1_interp') is not None else "N/A",
            f"{data.get('seed_1_extrap', 'N/A'):.2f}" if data.get('seed_1_extrap') is not None else "N/A",
            f"{ratio_1:.2f}x" if ratio_1 is not None else "N/A",
            f"{data.get('seed_2_interp', 'N/A'):.2f}" if data.get('seed_2_interp') is not None else "N/A",
            f"{data.get('seed_2_extrap', 'N/A'):.2f}" if data.get('seed_2_extrap') is not None else "N/A",
            f"{ratio_2:.2f}x" if ratio_2 is not None else "N/A",
        ]
        md_lines.append("| " + " | ".join(row) + " |")
    
    md_lines.extend([
        "",
        "## Standard Train/Test Splits Summary",
        "",
        "This table shows standard train/test splits (n_train=25, seed=0).",
        "",
        "| Equation | Parameter Range | n_train | n_test |",
        "|----------|----------------|---------|--------|",
    ])
    
    for data in standard_data:
        row = [
            data["equation"],
            data["param_range"],
            str(data["n_train"]),
            str(data["n_test"]),
        ]
        md_lines.append("| " + " | ".join(row) + " |")
    
    # Write markdown file
    output_path = data_dir / "dataset_summary.md"
    output_path.write_text("\n".join(md_lines))
    print(f"\nSummary written to: {output_path}")
    
    # Also write JSON for programmatic access
    json_path = data_dir / "dataset_summary.json"
    json_path.write_text(json.dumps({
        "parametric": parametric_data,
        "standard": standard_data,
    }, indent=2))
    print(f"JSON summary written to: {json_path}")


if __name__ == "__main__":
    main()
