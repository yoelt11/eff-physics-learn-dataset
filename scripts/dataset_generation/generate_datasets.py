#!/usr/bin/env python3
"""Unified script to generate verified PDE datasets.

This script generates new datasets with 100% PINN loss agreement using
JAX-based solvers and integrated verification.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eff_physics_learn_dataset.solvers.allen_cahn import generate_allen_cahn_sample
from eff_physics_learn_dataset.solvers.burgers import generate_burgers_sample
from eff_physics_learn_dataset.solvers.convection import generate_convection_sample_analytical
from eff_physics_learn_dataset.solvers.helmholtz2d import generate_helmholtz2d_sample_analytical
from eff_physics_learn_dataset.solvers.helmholtz3d import generate_helmholtz3d_sample_analytical
from eff_physics_learn_dataset.solvers.wave2d1 import generate_wave2d1_sample


# Parameter ranges from inspection
PARAM_RANGES = {
    "allen_cahn": {
        "eps": {"min": 0.0001, "max": 0.001},
        "lam": {"min": 1.0, "max": 5.0},
    },
    "burgers": {
        "A": {"min": 0.5, "max": 2.3},
        "k": {"min": 0.5, "max": 2.5},
        "nu": {"min": 0.01, "max": 0.3},
    },
    "convection": {
        "beta": {"min": 0.7, "max": 30.0},
    },
    "helmholtz2D": {
        "a1": {"min": 0.5, "max": 5.0},
        "a2": {"min": 0.5, "max": 5.0},
        "k": {"min": 0.5, "max": 5.0},
    },
    "helmholtz3D": {
        "a1": {"min": 0.5, "max": 5.0},
        "a2": {"min": 0.5, "max": 5.0},
        "a3": {"min": 0.5, "max": 5.0},
        "k": {"min": 0.5, "max": 5.0},
    },
    # 2D wave: u_tt = c^2 (u_xx + u_yy); sum of 1–3 Gaussian ICs [x0,y0,sd,amp] each
    "wave2d1": {
        "c0": {"min": 0.75, "max": 1.35},
        "n_sources": {"min": 1, "max": 3, "integer": True},
        "src1_x0": {"min": -1.5, "max": 1.5},
        "src1_y0": {"min": -1.5, "max": 1.5},
        "src1_sd": {"min": 0.18, "max": 0.42},
        "src1_amp": {"min": 0.6, "max": 1.4},
        "src2_x0": {"min": -1.5, "max": 1.5},
        "src2_y0": {"min": -1.5, "max": 1.5},
        "src2_sd": {"min": 0.18, "max": 0.42},
        "src2_amp": {"min": 0.6, "max": 1.4},
        "src3_x0": {"min": -1.5, "max": 1.5},
        "src3_y0": {"min": -1.5, "max": 1.5},
        "src3_sd": {"min": 0.18, "max": 0.42},
        "src3_amp": {"min": 0.6, "max": 1.4},
    },
}

# Domain configurations
DOMAIN_CONFIGS = {
    "allen_cahn": {"x_domain": (-1.0, 1.0), "t_domain": (0.0, 1.0)},
    "burgers": {"x_domain": (-1.0, 1.0), "t_domain": (0.0, 1.0)},
    "convection": {"x_domain": (0.0, 2.0 * np.pi), "t_domain": (0.0, 1.0)},
    "helmholtz2D": {"x_domain": (-1.0, 1.0), "y_domain": (-1.0, 1.0)},
    "helmholtz3D": {
        "x_domain": (-1.0, 1.0),
        "y_domain": (-1.0, 1.0),
        "z_domain": (-1.0, 1.0),
    },
    "wave2d1": {"x_domain": (-3.0, 3.0), "y_domain": (-3.0, 3.0)},
}

# Adjusted verification thresholds (relaxed for convection based on existing data)
VERIFY_THRESHOLDS = {
    "allen_cahn": {"pde_threshold": 0.1, "bc_threshold": 1e-3, "ic_threshold": 0.1},
    "burgers": {"pde_threshold": 5.0, "bc_threshold": 0.1, "ic_threshold": 1e-3},
    "convection": {"pde_threshold": 500.0, "bc_threshold": 1e-3, "ic_threshold": 1e-3},  # Relaxed
    "helmholtz2D": {"pde_threshold": 100.0, "bc_threshold": 1e-4},
    "helmholtz3D": {"pde_threshold": 100.0, "bc_threshold": 1e-4},
    # Downsampled finite-difference + np.gradient verification (softer than analytical cases)
    "wave2d1": {"pde_threshold": 8.0, "bc_threshold": 0.45, "ic_threshold": 5e-3},
}


def sample_parameters(param_ranges: dict, n: int, seed: int) -> list[dict]:
    """Sample parameters uniformly from ranges.

    Args:
        param_ranges: Dict mapping param name to ``{min, max}`` or, for integers,
            ``{min, max, integer: True}`` (inclusive ``max``).
        n: Number of samples
        seed: Random seed

    Returns:
        List of n parameter dicts
    """
    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n):
        params = {}
        for name, bounds in param_ranges.items():
            if bounds.get("integer"):
                lo, hi = int(bounds["min"]), int(bounds["max"])
                params[name] = int(rng.integers(lo, hi + 1))
            else:
                params[name] = float(rng.uniform(bounds["min"], bounds["max"]))
        samples.append(params)

    return samples


def generate_equation_dataset(
    equation: str,
    n_samples: int = 100,
    seed: int = 0,
    output_dir: Path = Path("datasets"),
    solver_resolution: int = 512,
    target_resolution: int = 64,
    max_retries: int = 3,
    *,
    ground_truth_subdir: str = "ground_truth",
) -> dict:
    """Generate dataset for a single equation.

    Args:
        equation: Equation name
        n_samples: Number of samples to generate
        seed: Random seed
        output_dir: Base output directory
        solver_resolution: High-res solver grid size
        target_resolution: Target grid size (for downsampling)
        max_retries: Max attempts per sample

    Returns:
        Generation statistics dict
    """
    print(f"\n{'='*60}")
    print(f"Generating {equation} dataset")
    print(f"{'='*60}")

    param_ranges = PARAM_RANGES[equation]
    domain_config = DOMAIN_CONFIGS[equation]
    verify_thresholds = VERIFY_THRESHOLDS[equation]

    # Wave2D: cap spatial solver size so default --solver-res stays tractable (Nt × Nx × Ny FDTD).
    wave_solver_n = int(max(96, min(solver_resolution, 200)))
    wave_t_steps = int(max(200, min(320, int(1.35 * wave_solver_n))))

    # Sample parameters
    param_samples = sample_parameters(param_ranges, n_samples, seed)

    # Storage for valid solutions
    solutions = []
    param_dict = {k: [] for k in param_ranges.keys()}
    failed_samples = []

    # Generation loop
    start_time = time.time()

    for i, params in enumerate(param_samples):
        print(f"\nSample {i+1}/{n_samples}: {params}")

        success = False
        for attempt in range(max_retries):
            try:
                # Call appropriate generator (Helmholtz2D uses JAX GPU)
                if equation == "allen_cahn":
                    u, x, t, is_valid, metrics = generate_allen_cahn_sample(
                        eps=params["eps"],
                        lam=params["lam"],
                        solver_nx=solver_resolution,
                        solver_nt=solver_resolution,
                        target_nx=target_resolution,
                        target_nt=target_resolution,
                        verify_thresholds=verify_thresholds,
                        **domain_config,
                    )
                elif equation == "burgers":
                    u, x, t, is_valid, metrics = generate_burgers_sample(
                        nu=params["nu"],
                        A=params["A"],
                        k=int(round(float(params["k"]))),
                        solver_nx=solver_resolution,
                        solver_nt=solver_resolution,
                        target_nx=target_resolution,
                        target_nt=target_resolution,
                        verify_thresholds=verify_thresholds,
                        **domain_config,
                    )
                elif equation == "convection":
                    u, x, t, is_valid, metrics = generate_convection_sample_analytical(
                        beta=params["beta"],
                        target_nx=target_resolution,
                        target_nt=target_resolution,
                        verify_thresholds=verify_thresholds,
                        **domain_config,
                    )
                elif equation == "helmholtz2D":
                    u, x, y, is_valid, metrics = generate_helmholtz2d_sample_analytical(
                        k=params["k"],
                        a1=params["a1"],
                        a2=params["a2"],
                        target_nx=target_resolution,
                        target_ny=target_resolution,
                        verify_thresholds=verify_thresholds,
                        **domain_config,
                    )
                elif equation == "helmholtz3D":
                    u, x, y, z, is_valid, metrics = generate_helmholtz3d_sample_analytical(
                        k=params["k"],
                        a1=params["a1"],
                        a2=params["a2"],
                        a3=params["a3"],
                        target_nx=target_resolution,
                        target_ny=target_resolution,
                        target_nz=target_resolution,
                        verify_thresholds=verify_thresholds,
                        **domain_config,
                    )
                elif equation == "wave2d1":
                    n_src = int(params["n_sources"])
                    src_rows = [
                        [params["src1_x0"], params["src1_y0"], params["src1_sd"], params["src1_amp"]],
                        [params["src2_x0"], params["src2_y0"], params["src2_sd"], params["src2_amp"]],
                        [params["src3_x0"], params["src3_y0"], params["src3_sd"], params["src3_amp"]],
                    ][:n_src]
                    sources = np.asarray(src_rows, dtype=np.float64)
                    u, x, y, t, is_valid, metrics = generate_wave2d1_sample(
                        c0=params["c0"],
                        sources=sources,
                        velocity_type="constant",
                        t_steps=wave_t_steps,
                        solver_nx=wave_solver_n,
                        solver_ny=wave_solver_n,
                        target_nx=target_resolution,
                        target_ny=target_resolution,
                        target_nt=target_resolution,
                        verify_thresholds=verify_thresholds,
                        x_domain=domain_config["x_domain"],
                        y_domain=domain_config["y_domain"],
                        backend="numpy",
                        dtype=np.float64,
                    )
                else:
                    raise ValueError(f"Unknown equation: {equation}")

                if is_valid:
                    solutions.append(u)
                    for k, v in params.items():
                        param_dict[k].append(v)
                    success = True
                    print(f"  ✓ Valid (attempt {attempt+1})")
                    print(f"    PDE: {metrics['pde_loss']:.2e}, BC: {metrics['bc_loss']:.2e}, IC: {metrics.get('ic_loss', 0):.2e}")
                    break
                else:
                    print(f"  ✗ Failed verification (attempt {attempt+1}): {metrics}")

            except Exception as e:
                print(f"  ✗ Error (attempt {attempt+1}): {e}")

        if not success:
            failed_samples.append((i, params))
            print(f"  ⚠ Giving up after {max_retries} attempts")

    elapsed = time.time() - start_time

    # Convert to arrays
    solutions = np.array(solutions, dtype=np.float64)
    pde_params = {k: np.array(v, dtype=np.float64) for k, v in param_dict.items()}

    # Create grids
    if equation == "helmholtz2D":
        X_grid, Y_grid = np.meshgrid(x, y, indexing='ij')
        grids = {"X_grid": X_grid, "Y_grid": Y_grid}
        grid_info = {
            "x_domain": domain_config["x_domain"],
            "y_domain": domain_config["y_domain"],
            "nx": target_resolution,
            "ny": target_resolution,
            "grid_size": (target_resolution, target_resolution),
        }
    elif equation == "helmholtz3D":
        X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing="ij")
        grids = {"X_grid": X_grid, "Y_grid": Y_grid, "Z_grid": Z_grid}
        grid_info = {
            "x_domain": domain_config["x_domain"],
            "y_domain": domain_config["y_domain"],
            "z_domain": domain_config["z_domain"],
            "nx": target_resolution,
            "ny": target_resolution,
            "nz": target_resolution,
            "grid_size": (target_resolution, target_resolution, target_resolution),
        }
    elif equation == "wave2d1":
        X_grid, Y_grid = np.meshgrid(x, y, indexing="ij")
        _, T_grid = np.meshgrid(x, t, indexing="ij")
        grids = {"X_grid": X_grid, "Y_grid": Y_grid, "T_grid": T_grid}
        grid_info = {
            "x_domain": domain_config["x_domain"],
            "y_domain": domain_config["y_domain"],
            "t_span": (float(np.asarray(t)[0]), float(np.asarray(t)[-1])),
            "nx": target_resolution,
            "ny": target_resolution,
            "nt": target_resolution,
            "grid_size": (target_resolution, target_resolution, target_resolution),
        }
    else:
        X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
        grids = {"X_grid": X_grid, "T_grid": T_grid}
        grid_info = {
            "x_domain": domain_config["x_domain"],
            "t_domain": domain_config["t_domain"],
            "nx": target_resolution,
            "nt": target_resolution,
            "grid_size": (target_resolution, target_resolution),
        }

    gsz = grid_info["grid_size"]
    if equation == "wave2d1":
        solver_gsz = (wave_t_steps, wave_solver_n, wave_solver_n)
    else:
        solver_gsz = tuple(solver_resolution for _ in range(len(gsz)))

    # Metadata
    metadata = {
        "domain": domain_config.get("x_domain", domain_config.get("x_domain")),
        "n_samples": len(solutions),
        "parameter_ranges": param_ranges,
        "grid_size": gsz,
        "solver_grid_size": solver_gsz,
        "sampling_method": "uniform_random",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timing": {"total_seconds": elapsed, "per_sample_seconds": elapsed / len(solutions) if len(solutions) > 0 else 0},
        "verification_thresholds": verify_thresholds,
        "seed": seed,
    }
    if equation == "helmholtz2D":
        metadata["field_layout"] = {"layout_id": "xy"}
    elif equation == "helmholtz3D":
        metadata["field_layout"] = {"layout_id": "xyz"}
    elif equation == "wave2d1":
        metadata["field_layout"] = {"layout_id": "txy"}

    # Save dataset (default subdir matches load_pde_dataset: .../ground_truth/*_dataset.pkl)
    out_dir = output_dir / equation / ground_truth_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = {
        "solutions": solutions,
        "pde_params": pde_params,
        "grid_info": grid_info,
        "metadata": metadata,
        **grids,
    }

    pkl_path = out_dir / f"{equation}_dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Saved to: {pkl_path}")

    # Generate test indices (20% of samples, rounded down)
    rng = np.random.default_rng(seed + 1000)
    n_test = max(1, len(solutions) // 5)
    test_indices = rng.choice(len(solutions), size=n_test, replace=False)
    test_indices = np.sort(test_indices).tolist()

    test_path = out_dir / "test_indices.pkl"
    with open(test_path, "wb") as f:
        pickle.dump(test_indices, f)

    print(f"✓ Saved test indices ({len(test_indices)} samples): {test_path}")

    # Statistics
    stats = {
        "equation": equation,
        "n_requested": n_samples,
        "n_generated": len(solutions),
        "n_failed": len(failed_samples),
        "success_rate": len(solutions) / n_samples,
        "elapsed_seconds": elapsed,
        "failed_params": failed_samples,
    }

    print(f"\nStats:")
    print(f"  Requested: {stats['n_requested']}")
    print(f"  Generated: {stats['n_generated']}")
    print(f"  Failed: {stats['n_failed']}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    per_s = elapsed / len(solutions) if len(solutions) else 0.0
    print(f"  Time: {elapsed:.1f}s ({per_s:.2f}s per sample)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate verified PDE datasets")
    parser.add_argument(
        "-e", "--equation",
        choices=[
            "allen_cahn",
            "burgers",
            "convection",
            "helmholtz2D",
            "helmholtz3D",
            "wave2d1",
            "all",
        ],
        default="all",
        help="Equation to generate (default: all)",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=100,
        help="Number of samples per equation (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Output directory (default: ./datasets)",
    )
    parser.add_argument(
        "--solver-res",
        type=int,
        default=512,
        help="Solver grid resolution (default: 512)",
    )
    parser.add_argument(
        "--target-res",
        type=int,
        default=64,
        help="Target grid resolution (default: 64)",
    )
    parser.add_argument(
        "--gt-subdir",
        type=str,
        default="ground_truth",
        help="Subdirectory under <output>/<equation>/ for pickles (default: ground_truth)",
    )

    args = parser.parse_args()

    equations = (
        ["allen_cahn", "burgers", "convection", "helmholtz2D", "helmholtz3D", "wave2d1"]
        if args.equation == "all"
        else [args.equation]
    )

    print("=" * 60)
    print("Verified PDE Dataset Generation")
    print("=" * 60)
    print(f"Equations: {', '.join(equations)}")
    print(f"Samples per equation: {args.n_samples}")
    print(f"Seed: {args.seed}")
    print(f"Solver resolution: {args.solver_res}")
    print(f"Target resolution: {args.target_res}")
    print(f"Output directory: {args.output_dir}")

    all_stats = []
    for eq in equations:
        try:
            stats = generate_equation_dataset(
                equation=eq,
                n_samples=args.n_samples,
                seed=args.seed,
                output_dir=args.output_dir,
                solver_resolution=args.solver_res,
                target_resolution=args.target_res,
                ground_truth_subdir=args.gt_subdir,
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"\n✗ Failed to generate {eq}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    print("Generation Complete")
    print(f"{'='*60}")
    for stats in all_stats:
        print(f"{stats['equation']:15s}: {stats['n_generated']}/{stats['n_requested']} samples "
              f"({stats['success_rate']*100:.1f}% success) in {stats['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
