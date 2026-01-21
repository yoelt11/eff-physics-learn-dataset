# Dataset Generation

This directory contains scripts to regenerate the PDE datasets from scratch with verification.

## Quick Start

Generate all datasets (200 samples each):
```bash
uv run python scripts/dataset_generation/generate_datasets.py --n-samples 200 --seed 42
```

## What's Included

- **`generate_datasets.py`** - Main script to generate all 4 PDE datasets
- **`solvers/`** - Clean, optimized PDE solvers for each equation
- **Verification** - Built-in PINN-based verification (see `src/eff_physics_learn_dataset/datasets/verification.py`)

## Solvers

Each equation uses the most accurate and efficient method:

| Equation | Method | Success Rate | Speed |
|----------|--------|--------------|-------|
| **Allen-Cahn** | scipy ODE solver | 100% | ~0.02s/sample |
| **Burgers** | scipy ODE solver | ~71% | ~13s/sample |
| **Convection** | **Analytical solution** | 100% | ~0.001s/sample |
| **Helmholtz2D** | **Analytical solution** | 100% | ~0.001s/sample |

### Key Innovation: Analytical Solutions with Analytical Derivatives

For **Convection** and **Helmholtz2D**, we use:
1. Direct evaluation of analytical formula (no PDE solving!)
2. Analytical derivatives (no finite differences!)
3. Result: Machine precision (~10^-28) and 1000x speedup

This eliminates numerical errors that previously caused failures with extreme parameter values.

## Output

Generated datasets are saved to:
```
datasets/{equation}/ground_truth_regenerated/
├── {equation}_dataset.pkl
└── test_indices.pkl
```

Use `scripts/switch_datasets.py` to activate them.

## Verification

All generated solutions are verified against:
- **PDE residuals**: u_t + F(u, u_x, u_xx, ...) = 0
- **Boundary conditions**: Periodic or Dirichlet
- **Initial conditions**: Match specified IC

Thresholds are calibrated to match PINN loss values from the original paper.
