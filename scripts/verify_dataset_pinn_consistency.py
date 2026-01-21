#!/usr/bin/env python3
"""Verify generated datasets match PINN loss expectations using verification module.

This script loads generated datasets and verifies each sample (or a subset) 
using the verification functions to ensure 100% PINN loss agreement.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import verification functions directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "verification",
    src_path / "eff_physics_learn_dataset" / "datasets" / "verification.py"
)
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)

verify_solution_allen_cahn = verification.verify_solution_allen_cahn
verify_solution_burgers = verification.verify_solution_burgers
verify_solution_convection = verification.verify_solution_convection
verify_solution_helmholtz2d = verification.verify_solution_helmholtz2d

# Also import analytical verification functions if available (for comparison)
try:
    solver_path = Path(__file__).parent.parent / "scripts" / "dataset_generation" / "solvers"
    if (solver_path / "generate_convection_analytical.py").exists():
        spec_conv = importlib.util.spec_from_file_location(
            "convection_analytical",
            solver_path / "generate_convection_analytical.py"
        )
        conv_analytical = importlib.util.module_from_spec(spec_conv)
        spec_conv.loader.exec_module(conv_analytical)
        verify_analytical_convection = conv_analytical.verify_analytical_convection
    else:
        verify_analytical_convection = None
        
    if (solver_path / "generate_helmholtz2d_analytical.py").exists():
        spec_helm = importlib.util.spec_from_file_location(
            "helmholtz2d_analytical",
            solver_path / "generate_helmholtz2d_analytical.py"
        )
        helm_analytical = importlib.util.module_from_spec(spec_helm)
        spec_helm.loader.exec_module(helm_analytical)
        verify_analytical_helmholtz2d = helm_analytical.verify_analytical_helmholtz2d
    else:
        verify_analytical_helmholtz2d = None
except Exception:
    verify_analytical_convection = None
    verify_analytical_helmholtz2d = None


def verify_dataset_samples(
    equation: str,
    data_dir: Path,
    subfolder: str = "ground_truth_regenerated",
    n_samples: int = None,
    sample_indices: List[int] = None,
    use_analytical: bool = False,
) -> Dict:
    """Verify samples from a dataset using PINN verification functions.
    
    Args:
        equation: Equation name
        data_dir: Base data directory
        subfolder: Subfolder name (e.g., "ground_truth_regenerated")
        n_samples: Number of random samples to verify (None = all)
        sample_indices: Specific sample indices to verify (overrides n_samples)
        
    Returns:
        Dictionary with verification results
    """
    # Handle case variations in directory names (e.g., helmholtz2D vs helmholtz2d)
    eq_dir = data_dir / equation
    if not eq_dir.exists():
        # Try alternative case
        alt_equation = equation.replace("helmholtz2d", "helmholtz2D")
        eq_dir = data_dir / alt_equation
        if eq_dir.exists():
            equation = alt_equation
    
    dataset_dir = eq_dir / subfolder
    pkl_file = dataset_dir / f"{equation}_dataset.pkl"
    
    if not pkl_file.exists():
        # Try to find any *_dataset.pkl file
        pkl_files = list(dataset_dir.glob("*_dataset.pkl"))
        if pkl_files:
            pkl_file = pkl_files[0]
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_dir / f'{equation}_dataset.pkl'}")
    
    # Load dataset
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    solutions = np.asarray(data["solutions"])
    pde_params = data["pde_params"]
    
    # Get coordinate arrays
    if equation.lower() in ["helmholtz2d", "helmholtz2d"]:
        x_coords = data.get("x", None)
        y_coords = data.get("y", None)
        if x_coords is None or y_coords is None:
            # Extract from grids
            X_grid = data.get("X_grid")
            Y_grid = data.get("Y_grid")
            if X_grid is not None and Y_grid is not None:
                # X_grid and Y_grid are (n_x, n_y) meshgrids
                x_coords = X_grid[:, 0]  # First column gives x coordinates
                y_coords = Y_grid[0, :]  # First row gives y coordinates
            elif X_grid is not None:
                # Fallback: extract from X_grid only
                x_coords = X_grid[:, 0] if X_grid.ndim == 2 else np.linspace(0, 1, solutions.shape[1])
                y_coords = X_grid[0, :] if X_grid.ndim == 2 else np.linspace(0, 1, solutions.shape[2])
            else:
                # Default: assume [0, 1] x [0, 1] domain
                x_coords = np.linspace(0, 1, solutions.shape[1])
                y_coords = np.linspace(0, 1, solutions.shape[2])
    else:
        x_coords = data.get("x", None)
        t_coords = data.get("t", None)
        if x_coords is None or t_coords is None:
            # Extract from grids (try multiple possible keys)
            X_grid = data.get("X_grid")
            if X_grid is None:
                X_grid = data.get("X_grid_storage")
            if X_grid is None:
                X_grid = data.get("X_grid_solver")
            
            T_grid = data.get("T_grid")
            if T_grid is None:
                T_grid = data.get("T_grid_storage")
            if T_grid is None:
                T_grid = data.get("T_grid_solver")
            if X_grid is not None:
                X_grid = np.asarray(X_grid)
                if X_grid.ndim == 2:
                    # For meshgrid: X_grid[i, j] = x_coords[j], so extract first row
                    x_coords = X_grid[0, :]
                else:
                    x_coords = np.linspace(-1, 1, solutions.shape[2])
            else:
                x_coords = np.linspace(-1, 1, solutions.shape[2])
            if T_grid is not None:
                T_grid = np.asarray(T_grid)
                if T_grid.ndim == 2:
                    # For meshgrid: T_grid[i, j] = t_coords[i], so extract first column
                    t_coords = T_grid[:, 0]
                else:
                    t_coords = np.linspace(0, 1, solutions.shape[1])
            else:
                t_coords = np.linspace(0, 1, solutions.shape[1])
    
    # Determine which samples to verify
    total_samples = solutions.shape[0]
    if sample_indices is not None:
        indices_to_verify = sample_indices
    elif n_samples is not None:
        indices_to_verify = np.random.choice(total_samples, size=min(n_samples, total_samples), replace=False).tolist()
    else:
        indices_to_verify = list(range(total_samples))
    
    print(f"\n{'='*70}")
    print(f"Verifying {equation} dataset ({subfolder})")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Verifying: {len(indices_to_verify)} samples")
    if len(indices_to_verify) < total_samples:
        print(f"Sample indices: {indices_to_verify[:10]}{'...' if len(indices_to_verify) > 10 else ''}")
    
    # Normalize equation name for threshold lookup
    eq_normalized = equation.lower()
    
    # Verification thresholds
    thresholds = {
        "allen_cahn": {"pde": 0.1, "bc": 1e-3, "ic": 0.1},
        "burgers": {"pde": 5.0, "bc": 0.1, "ic": 1e-3},
        "convection": {"pde": 1.0, "bc": 1e-3, "ic": 1e-3},
        "helmholtz2d": {"pde": 100.0, "bc": 1e-4, "ic": None},
    }
    
    if eq_normalized not in thresholds:
        raise ValueError(f"Unknown equation: {equation}")
    
    results = {
        "total": len(indices_to_verify),
        "passed": 0,
        "failed": 0,
        "samples": [],
        "summary": {},
    }
    
    # Verify each sample
    for idx in indices_to_verify:
        solution = solutions[idx]
        
        # Get parameters for this sample
        if eq_normalized == "allen_cahn":
            eps = pde_params["eps"][idx]
            lam = pde_params["lam"][idx]
            is_valid, metrics = verify_solution_allen_cahn(
                solution=solution,
                x_coords=x_coords,
                t_coords=t_coords,
                eps=eps,
                lam=lam,
                pde_threshold=thresholds[eq_normalized]["pde"],
                bc_threshold=thresholds[eq_normalized]["bc"],
                ic_threshold=thresholds[eq_normalized]["ic"],
            )
            
        elif eq_normalized == "burgers":
            nu = pde_params["nu"][idx]
            A = pde_params["A"][idx]
            k = pde_params["k"][idx]
            is_valid, metrics = verify_solution_burgers(
                solution=solution,
                x_coords=x_coords,
                t_coords=t_coords,
                nu=nu,
                A=A,
                k=k,
                pde_threshold=thresholds[eq_normalized]["pde"],
                bc_threshold=thresholds[eq_normalized]["bc"],
                ic_threshold=thresholds[eq_normalized]["ic"],
            )
            
        elif eq_normalized == "convection":
            beta = pde_params["beta"][idx]
            # Use analytical verification if requested (matches generation method)
            if use_analytical and verify_analytical_convection is not None:
                is_valid, metrics = verify_analytical_convection(
                    u=solution,
                    x=x_coords,
                    t=t_coords,
                    beta=beta,
                    pde_threshold=thresholds[eq_normalized]["pde"],
                    bc_threshold=thresholds[eq_normalized]["bc"],
                    ic_threshold=thresholds[eq_normalized]["ic"],
                )
            else:
                # Use numerical derivatives (matches PINN behavior)
                is_valid, metrics = verify_solution_convection(
                    solution=solution,
                    x_coords=x_coords,
                    t_coords=t_coords,
                    beta=beta,
                    pde_threshold=thresholds[eq_normalized]["pde"],
                    bc_threshold=thresholds[eq_normalized]["bc"],
                    ic_threshold=thresholds[eq_normalized]["ic"],
                )
            
        elif eq_normalized in ["helmholtz2d", "helmholtz2d"]:
            k = pde_params["k"][idx]
            a1 = pde_params["a1"][idx]
            a2 = pde_params["a2"][idx]
            # Use analytical verification if requested (matches generation method)
            if use_analytical and verify_analytical_helmholtz2d is not None:
                is_valid, metrics = verify_analytical_helmholtz2d(
                    u=solution,
                    x=x_coords,
                    y=y_coords,
                    k=k,
                    a1=a1,
                    a2=a2,
                    pde_threshold=thresholds[eq_normalized]["pde"],
                    bc_threshold=thresholds[eq_normalized]["bc"],
                )
            else:
                # Use numerical derivatives (matches PINN behavior)
                is_valid, metrics = verify_solution_helmholtz2d(
                    solution=solution,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    k=k,
                    a1=a1,
                    a2=a2,
                    pde_threshold=thresholds[eq_normalized]["pde"],
                    bc_threshold=thresholds[eq_normalized]["bc"],
                )
        else:
            raise ValueError(f"Unknown equation: {equation}")
        
        sample_result = {
            "index": idx,
            "valid": is_valid,
            "metrics": metrics,
        }
        results["samples"].append(sample_result)
        
        if is_valid:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Compute summary statistics
    pde_losses = [s["metrics"]["pde_loss"] for s in results["samples"]]
    bc_losses = [s["metrics"]["bc_loss"] for s in results["samples"]]
    ic_losses = [s["metrics"].get("ic_loss", 0.0) for s in results["samples"]]
    
    results["summary"] = {
        "pde_loss": {
            "mean": float(np.mean(pde_losses)),
            "std": float(np.std(pde_losses)),
            "min": float(np.min(pde_losses)),
            "max": float(np.max(pde_losses)),
        },
        "bc_loss": {
            "mean": float(np.mean(bc_losses)),
            "std": float(np.std(bc_losses)),
            "min": float(np.min(bc_losses)),
            "max": float(np.max(bc_losses)),
        },
        "ic_loss": {
            "mean": float(np.mean(ic_losses)),
            "std": float(np.std(ic_losses)),
            "min": float(np.min(ic_losses)),
            "max": float(np.max(ic_losses)),
        },
    }
    
    # Print results
    print(f"\nVerification Results:")
    print(f"  Passed: {results['passed']}/{results['total']} ({100*results['passed']/results['total']:.1f}%)")
    print(f"  Failed: {results['failed']}/{results['total']}")
    
    print(f"\nLoss Statistics:")
    print(f"  PDE Loss: mean={results['summary']['pde_loss']['mean']:.6e}, "
          f"std={results['summary']['pde_loss']['std']:.6e}, "
          f"range=[{results['summary']['pde_loss']['min']:.6e}, {results['summary']['pde_loss']['max']:.6e}]")
    print(f"  BC Loss:  mean={results['summary']['bc_loss']['mean']:.6e}, "
          f"std={results['summary']['bc_loss']['std']:.6e}, "
          f"range=[{results['summary']['bc_loss']['min']:.6e}, {results['summary']['bc_loss']['max']:.6e}]")
    if eq_normalized not in ["helmholtz2d", "helmholtz2d"]:
        print(f"  IC Loss:  mean={results['summary']['ic_loss']['mean']:.6e}, "
              f"std={results['summary']['ic_loss']['std']:.6e}, "
              f"range=[{results['summary']['ic_loss']['min']:.6e}, {results['summary']['ic_loss']['max']:.6e}]")
    
    # Show failed samples
    failed_samples = [s for s in results["samples"] if not s["valid"]]
    if failed_samples:
        print(f"\nFailed Samples:")
        for sample in failed_samples[:10]:  # Show first 10 failures
            print(f"  Sample {sample['index']}:")
            print(f"    PDE: {sample['metrics']['pde_loss']:.6e} ({'✓' if sample['metrics']['pde_ok'] else '✗'})")
            print(f"    BC:  {sample['metrics']['bc_loss']:.6e} ({'✓' if sample['metrics']['bc_ok'] else '✗'})")
            if 'ic_ok' in sample['metrics']:
                print(f"    IC:  {sample['metrics']['ic_loss']:.6e} ({'✓' if sample['metrics']['ic_ok'] else '✗'})")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify generated datasets match PINN loss expectations"
    )
    parser.add_argument(
        "-e", "--equation",
        type=str,
        required=True,
        choices=["allen_cahn", "burgers", "convection", "helmholtz2d"],
        help="Equation to verify",
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default=Path("datasets"),
        help="Base data directory (default: ./datasets)",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="ground_truth_regenerated",
        help="Subfolder name (default: ground_truth_regenerated)",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=None,
        help="Number of random samples to verify (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to verify (e.g., --samples 0 1 2)",
    )
    parser.add_argument(
        "--use-analytical",
        action="store_true",
        help="Use analytical derivatives for convection/helmholtz2d (matches generation, not PINN)",
    )
    
    args = parser.parse_args()
    
    try:
        results = verify_dataset_samples(
            equation=args.equation,
            data_dir=args.data_dir,
            subfolder=args.subfolder,
            n_samples=args.n_samples,
            sample_indices=args.samples,
            use_analytical=args.use_analytical,
        )
        
        # Exit with error code if any samples failed
        if results["failed"] > 0:
            print(f"\n✗ Verification failed: {results['failed']} samples did not pass")
            sys.exit(1)
        else:
            print(f"\n✓ All samples passed verification!")
            sys.exit(0)
            
    except Exception as e:
        import traceback
        print(f"\n✗ Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
