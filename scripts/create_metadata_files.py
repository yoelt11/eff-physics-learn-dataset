#!/usr/bin/env python3
"""Create metadata.txt files for regenerated datasets."""

import pickle
from pathlib import Path
import numpy as np

# Generation statistics from generation_200.log
GENERATION_STATS = {
    "allen_cahn": {
        "requested": 200,
        "generated": 200,
        "success_rate": 100.0,
        "time_seconds": 4.3,
        "solver": "scipy.integrate.solve_ivp (CPU)",
        "solver_resolution": "512x512",
        "verification": "PINN-based (PDE + BC + IC losses)"
    },
    "burgers": {
        "requested": 200,
        "generated": 143,
        "success_rate": 71.5,
        "time_seconds": 1946.8,
        "solver": "scipy.integrate.solve_ivp (CPU)",
        "solver_resolution": "512x512",
        "verification": "PINN-based (PDE + BC + IC losses)"
    },
    "convection": {
        "requested": 200,
        "generated": 200,
        "success_rate": 100.0,
        "time_seconds": 0.1,
        "solver": "Analytical solution with analytical derivatives",
        "solver_resolution": "64x64 (direct evaluation)",
        "verification": "PINN-based (PDE + BC + IC losses)"
    },
    "helmholtz2D": {
        "requested": 200,
        "generated": 200,
        "success_rate": 100.0,
        "time_seconds": 0.1,
        "solver": "Analytical solution with analytical derivatives",
        "solver_resolution": "64x64 (direct evaluation)",
        "verification": "PINN-based (PDE + BC + IC losses)"
    }
}

EQUATION_METADATA = {
    "allen_cahn": {
        "title": "Allen-Cahn Equation",
        "pde": "u_t = ε² u_xx - λ(u³ - u)",
        "domain_x": "(-1, 1)",
        "domain_t": "(0, 1.0)",
        "bc": "Dirichlet: u(-1,t) = u(1,t) = 0",
        "ic": "u(x,0) = sin(πx)"
    },
    "burgers": {
        "title": "Burgers Equation",
        "pde": "u_t + u·u_x = ν u_xx",
        "domain_x": "(-1, 1)",
        "domain_t": "(0, 1.0)",
        "bc": "Periodic: u(-1,t) = u(1,t)",
        "ic": "u(x,0) = A·sin(k·π·x)"
    },
    "convection": {
        "title": "Linear Convection Equation",
        "pde": "u_t + β u_x = 0",
        "domain_x": "(0, L) where L=1",
        "domain_t": "(0, 1.0)",
        "bc": "Periodic: u(0,t) = u(L,t)",
        "ic": "u(x,0) = 1 + sin(2πx/L)"
    },
    "helmholtz2D": {
        "title": "Helmholtz 2D Equation",
        "pde": "Δu + k²u = q(x,y)",
        "domain_x": "(0, 1)",
        "domain_y": "(0, 1)",
        "bc": "Dirichlet: u=0 on boundary",
        "solution": "u(x,y) = sin(a₁πx)sin(a₂πy)"
    }
}


def create_metadata(equation: str, dataset_dir: Path):
    """Create metadata.txt file for a regenerated dataset."""

    # Load dataset
    dataset_path = dataset_dir / f"{equation}_dataset.pkl"
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    solutions = data['solutions']
    params = data['pde_params']
    grid_info = data.get('grid_info', {})

    n_samples = solutions.shape[0]
    solution_shape = solutions.shape[1:]

    # Get generation stats
    stats = GENERATION_STATS[equation]
    eq_meta = EQUATION_METADATA[equation]

    # Build metadata content
    lines = []
    lines.append(f"{eq_meta['title']} Dataset Metadata (Regenerated)")
    lines.append("=" * 60)
    lines.append("")

    # Dataset info
    lines.append(f"Number of samples: {n_samples}")
    lines.append(f"Generation seed: 42")
    lines.append(f"Solver resolution: {stats['solver_resolution']}")
    lines.append(f"Storage resolution: {solution_shape[0]}x{solution_shape[1]}")
    lines.append(f"Storage grid size: {solution_shape}")

    # Domain info
    if equation in ["allen_cahn", "burgers", "convection"]:
        lines.append(f"Domain x: {eq_meta['domain_x']}")
        lines.append(f"Domain t: {eq_meta['domain_t']}")
    else:  # helmholtz2D
        lines.append(f"Domain x: {eq_meta['domain_x']}")
        lines.append(f"Domain y: {eq_meta['domain_y']}")

    lines.append("")

    # PDE info
    lines.append("PDE Information:")
    lines.append("-" * 40)
    lines.append(f"Equation: {eq_meta['pde']}")
    lines.append(f"Boundary conditions: {eq_meta['bc']}")
    if 'ic' in eq_meta:
        lines.append(f"Initial condition: {eq_meta['ic']}")
    if 'solution' in eq_meta:
        lines.append(f"Analytical solution: {eq_meta['solution']}")
    lines.append("")

    # Parameter ranges
    lines.append("PDE Parameters:")
    lines.append("-" * 40)
    for pname, pvals in params.items():
        pmin, pmax = pvals.min(), pvals.max()
        lines.append(f"  {pname}: [{pmin:.6f}, {pmax:.6f}]")
    lines.append("")

    # Solver info
    lines.append("Solver Information:")
    lines.append("-" * 40)
    lines.append(f"Method: {stats['solver']}")
    lines.append(f"Resolution: {stats['solver_resolution']}")
    lines.append(f"Verification: {stats['verification']}")
    lines.append("")

    # Timing info
    lines.append("Generation Statistics:")
    lines.append("-" * 40)
    lines.append(f"Samples requested: {stats['requested']}")
    lines.append(f"Samples generated: {stats['generated']}")
    lines.append(f"Failed samples: {stats['requested'] - stats['generated']}")
    lines.append(f"Success rate: {stats['success_rate']:.1f}%")
    lines.append(f"Total generation time: {stats['time_seconds']:.2f} seconds")
    lines.append(f"Total generation time: {stats['time_seconds']/60:.2f} minutes")
    avg_time = stats['time_seconds'] / stats['generated'] if stats['generated'] > 0 else 0
    lines.append(f"Average time per sample: {avg_time:.4f} seconds")
    lines.append("")

    # Additional notes
    lines.append("Notes:")
    lines.append("-" * 40)
    if equation == "burgers":
        lines.append("- Lower success rate (71.5%) is expected for Burgers equation due to")
        lines.append("  shock formation and nonlinear dynamics")
        lines.append("- Each sample undergoes 3 verification attempts with random seeds")
    elif equation in ["convection", "helmholtz2D"]:
        lines.append("- Analytical solution with analytical derivatives ensures machine")
        lines.append("  precision accuracy (PDE loss ~10^-28)")
        lines.append("- Fast generation due to direct evaluation (no time-stepping)")
    else:  # allen_cahn
        lines.append("- scipy solver with adaptive time-stepping")
        lines.append("- 100% success rate indicates stable numerical integration")
    lines.append("")

    # Write metadata file
    metadata_path = dataset_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Created {metadata_path}")


def main():
    """Create metadata files for all regenerated datasets."""

    datasets_root = Path("datasets")
    equations = ["allen_cahn", "burgers", "convection", "helmholtz2D"]

    print("Creating metadata files for regenerated datasets...\n")

    for eq in equations:
        dataset_dir = datasets_root / eq / "ground_truth"
        if not dataset_dir.exists():
            print(f"✗ Skipping {eq}: {dataset_dir} not found")
            continue

        create_metadata(eq, dataset_dir)

    print("\n✓ All metadata files created!")


if __name__ == "__main__":
    main()
