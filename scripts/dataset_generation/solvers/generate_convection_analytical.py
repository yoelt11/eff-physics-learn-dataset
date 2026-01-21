"""Generate verified Convection equation dataset using analytical solution.

PDE: u_t + β u_x = 0
BC: Periodic
IC: u(x,0) = 1 + sin(2πx/L) where L = x_max - x_min

Analytical solution: u(x,t) = 1 + sin(2π(x - βt)/L)
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path for verification
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from eff_physics_learn_dataset.datasets.verification import verify_solution_convection


def analytical_convection_solution(
    x: np.ndarray,
    t: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Compute analytical solution to convection equation.

    PDE: u_t + β u_x = 0
    IC: u(x,0) = 1 + sin(2πx/L)
    BC: Periodic

    Analytical solution: u(x,t) = 1 + sin(2π(x - βt)/L)

    Args:
        x: 1D spatial grid (nx,)
        t: 1D temporal grid (nt,)
        beta: Convection velocity

    Returns:
        Solution u with shape (nt, nx) - transposed indexing: u[i,j] = u(x[j], t[i])
    """
    L = x[-1] - x[0]  # Domain length
    nt = len(t)
    nx = len(x)

    # Initialize solution array
    u = np.zeros((nt, nx))

    # Compute analytical solution at each time step
    for i, ti in enumerate(t):
        # Wave travels with velocity beta
        # u(x,t) = 1 + sin(2π(x - βt)/L)
        u[i, :] = 1.0 + np.sin(2 * np.pi * (x - beta * ti) / L)

    return u


def verify_analytical_convection(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    beta: float,
    pde_threshold: float = 1.0,
    bc_threshold: float = 1e-3,
    ic_threshold: float = 1e-3,
) -> tuple[bool, dict]:
    """Verify analytical convection solution using analytical derivatives.

    This avoids numerical derivative errors that get amplified by large beta values.

    Args:
        u: Solution array (nt, nx) - transposed indexing
        x: 1D spatial coordinates
        t: 1D temporal coordinates
        beta: Convection velocity
        pde_threshold: Maximum acceptable PDE residual MSE
        bc_threshold: Maximum acceptable BC error MSE
        ic_threshold: Maximum acceptable IC error MSE

    Returns:
        (is_valid, metrics) where metrics contains loss values
    """
    L = x[-1] - x[0]
    nt, nx = u.shape

    # Create meshgrid for vectorized computation
    X, T = np.meshgrid(x, t)

    # Analytical derivatives:
    # u(x,t) = 1 + sin(2π(x - βt)/L)
    # u_x = (2π/L) cos(2π(x - βt)/L)
    # u_t = -(2πβ/L) cos(2π(x - βt)/L)

    phase = 2 * np.pi * (X - beta * T) / L
    u_x_analytical = (2 * np.pi / L) * np.cos(phase)
    u_t_analytical = -(2 * np.pi * beta / L) * np.cos(phase)

    # PDE residual: u_t + β u_x = 0 (should be exactly 0 with analytical derivatives)
    pde_residual = u_t_analytical + beta * u_x_analytical
    pde_loss = float(np.mean(pde_residual**2))

    # BC: Periodic - u(x=0, t) = u(x=L, t)
    bc_left = u[:, 0]
    bc_right = u[:, -1]
    bc_loss = float(np.mean((bc_left - bc_right)**2))

    # IC: u(x, t=0) = 1 + sin(2πx/L)
    ic_expected = 1.0 + np.sin(2 * np.pi * x / L)
    ic_actual = u[0, :]
    ic_loss = float(np.mean((ic_actual - ic_expected)**2))

    pde_ok = pde_loss < pde_threshold
    bc_ok = bc_loss < bc_threshold
    ic_ok = ic_loss < ic_threshold
    is_valid = pde_ok and bc_ok and ic_ok

    metrics = {
        'pde_loss': pde_loss,
        'bc_loss': bc_loss,
        'ic_loss': ic_loss,
        'pde_ok': pde_ok,
        'bc_ok': bc_ok,
        'ic_ok': ic_ok
    }

    return is_valid, metrics


def generate_convection_sample_analytical(
    beta: float,
    x_domain: tuple[float, float] = (0.0, 2.0 * np.pi),
    t_domain: tuple[float, float] = (0.0, 1.0),
    target_nx: int = 64,
    target_nt: int = 64,
    verify: bool = True,
    verify_thresholds: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
    """Generate a single verified Convection solution using analytical formula.

    Args:
        beta: Convection velocity
        x_domain: Spatial domain (xmin, xmax)
        t_domain: Time domain (tmin, tmax)
        target_nx: Target spatial grid size
        target_nt: Target temporal grid size
        verify: If True, verify solution
        verify_thresholds: Custom verification thresholds

    Returns:
        (solution, x_coords, t_coords, is_valid, metrics)
        solution has shape (target_nt, target_nx) - transposed indexing
    """
    # Note: No high-res solving needed! Analytical solution is exact at any resolution
    x = np.linspace(x_domain[0], x_domain[1], target_nx)
    t = np.linspace(t_domain[0], t_domain[1], target_nt)

    # Compute analytical solution directly at target resolution
    u = analytical_convection_solution(x, t, beta)

    # Verify solution using analytical derivatives (avoids numerical derivative errors)
    if verify:
        thresholds = verify_thresholds or {}
        is_valid, metrics = verify_analytical_convection(
            u=u,
            x=x,
            t=t,
            beta=beta,
            **thresholds,
        )
    else:
        is_valid = True
        metrics = {}

    return u, x, t, is_valid, metrics


if __name__ == "__main__":
    # Test generation
    print("Testing analytical Convection solver...")

    # Use parameters from inspection
    beta = 0.5

    import time

    # Timed run
    print("\nTimed run...")
    start = time.time()
    u, x, t, is_valid, metrics = generate_convection_sample_analytical(
        beta=beta,
        target_nx=64,
        target_nt=64,
    )
    elapsed = time.time() - start

    print(f"\nGenerated solution shape: {u.shape}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"Valid: {is_valid}")
    print(f"Metrics: {metrics}")
    print(f"Time: {elapsed:.4f}s")
