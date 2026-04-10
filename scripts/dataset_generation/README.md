# Dataset Generation

This directory contains scripts to regenerate the PDE datasets from scratch with verification.

## Quick Start

Generate all bundled datasets (200 samples each; **Allen–Cahn, Burgers, convection, Helmholtz2D, Helmholtz3D, wave2d1**):
```bash
uv run python scripts/dataset_generation/generate_datasets.py --n-samples 200 --seed 42
```

Single equation:

```bash
uv run python scripts/dataset_generation/generate_datasets.py -e wave2d1 -n 200 --target-res 64 -o datasets/new
```

## What's Included

- **`generate_datasets.py`** - Main script to generate all bundled PDE datasets
- **Solvers** live under `src/eff_physics_learn_dataset/solvers/`
- **Verification** — residual / BC / IC checks in `src/eff_physics_learn_dataset/verification.py` (used by solvers and some dev scripts)

## Solvers

Each equation uses the most accurate and efficient method:

| Equation | Method | Success Rate | Speed |
|----------|--------|--------------|-------|
| **Allen-Cahn** | scipy ODE solver | 100% | ~0.02s/sample |
| **Burgers** | scipy ODE solver | ~71% | ~13s/sample |
| **Convection** | **Analytical solution** | 100% | ~0.001s/sample |
| **Helmholtz2D** | **Analytical solution** | 100% | ~0.001s/sample |
| **Helmholtz3D** | **Analytical solution** | 100% | ~0.01s/sample |
| **Wave2D1** | Finite-difference + sponge BC | ~100%* | ~0.5s/sample |

\*Verification uses smoothed interior PDE residuals on downsampled grids; thresholds are set accordingly. Initial displacement is a **sum of 1–3** Gaussian sources per sample (`n_sources` in `pde_params`).

### Key Innovation: Analytical Solutions with Analytical Derivatives

For **Convection**, **Helmholtz2D**, and **Helmholtz3D**, we use:
1. Direct evaluation of analytical formula (no PDE solving!)
2. Analytical derivatives (no finite differences!)
3. Result: Machine precision (~10^-28) and 1000x speedup

This eliminates numerical errors that previously caused failures with extreme parameter values.

## Output

By default, pickles are written next to `load_pde_dataset` expectations:
```
datasets/{equation}/ground_truth/
├── {equation}_dataset.pkl
└── test_indices.pkl
```

Pass `--gt-subdir ground_truth_regenerated` to match older dev tooling, or use `scripts/switch_datasets.py` where applicable.

To preview a **`wave2d1`** sample over all time frames, use `scripts/animate_wave2d1_sample.py` (writes a GIF).

## Verification

All generated solutions are verified against:
- **PDE residuals**: u_t + F(u, u_x, u_xx, ...) = 0
- **Boundary conditions**: Periodic or Dirichlet
- **Initial conditions**: Match specified IC

Thresholds are calibrated to match PINN loss values from the original paper.
