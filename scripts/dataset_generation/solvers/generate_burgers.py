"""Generate verified Burgers equation dataset.

PDE: u_t + u·u_x = ν u_xx
BC: Periodic
IC: u(x,0) = A sin(kπx)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# Add src to path for verification
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from eff_physics_learn_dataset.datasets.verification import verify_solution_burgers

from utils import create_1d_grid, downsample_solution


def initial_condition_burgers(x: np.ndarray, A: float, k: float) -> np.ndarray:
    """Burgers initial condition: u(x,0) = A sin(kπx).

    Args:
        x: 1D spatial grid (assumed domain [-1, 1])
        A: Amplitude
        k: Wavenumber

    Returns:
        Initial condition array
    """
    return A * np.sin(k * np.pi * x)


def spatial_derivative_burgers(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Compute spatial derivative for Burgers equation with periodic BC.

    RHS: -u·u_x + ν·u_xx

    Args:
        u: Solution at current time (n_x,)
        dx: Spatial grid spacing
        nu: Viscosity

    Returns:
        du/dt array
    """
    n = len(u)

    # First derivative (centered, periodic)
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

    # Second derivative (centered, periodic)
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)

    # RHS: -u·u_x + ν·u_xx
    return -u * u_x + nu * u_xx


def solve_burgers(
    x: np.ndarray,
    t: np.ndarray,
    nu: float,
    A: float,
    k: float,
) -> np.ndarray:
    """Solve Burgers equation using scipy.integrate.solve_ivp.

    Args:
        x: Spatial grid (n_x,)
        t: Temporal grid (n_t,)
        nu: Viscosity
        A: IC amplitude
        k: IC wavenumber

    Returns:
        Solution u with shape (n_t, n_x) - transposed indexing!
    """
    dx = x[1] - x[0]
    n_x = len(x)

    # Initial condition
    u0 = initial_condition_burgers(x, A, k)

    # Define ODE system
    def rhs(t_val, u_flat):
        u = u_flat.reshape(n_x)
        dudt = spatial_derivative_burgers(u, dx, nu)
        return dudt.flatten()

    # Solve using solve_ivp
    sol = solve_ivp(
        rhs,
        t_span=(t[0], t[-1]),
        y0=u0.flatten(),
        t_eval=t,
        method='DOP853',
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    # Reshape to (n_t, n_x) - transposed indexing
    u = sol.y.T

    return u


def generate_burgers_sample(
    nu: float,
    A: float,
    k: float,
    x_domain: tuple[float, float] = (-1.0, 1.0),
    t_domain: tuple[float, float] = (0.0, 1.0),
    solver_nx: int = 128,
    solver_nt: int = 128,
    target_nx: int = 64,
    target_nt: int = 64,
    verify: bool = True,
    verify_thresholds: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
    """Generate a single verified Burgers solution.

    Args:
        nu: Viscosity
        A: IC amplitude
        k: IC wavenumber
        x_domain: Spatial domain (xmin, xmax)
        t_domain: Temporal domain (tmin, tmax)
        solver_nx: High-res spatial grid size
        solver_nt: High-res temporal grid size
        target_nx: Target spatial grid size
        target_nt: Target temporal grid size
        verify: If True, verify solution
        verify_thresholds: Custom verification thresholds

    Returns:
        (solution, x_coords, t_coords, is_valid, metrics)
        solution has shape (target_nt, target_nx) - transposed indexing
    """
    # Create high-res grids
    x_hires = create_1d_grid(x_domain, solver_nx)
    t_hires = create_1d_grid(t_domain, solver_nt)

    # Solve on high-res grid
    u_hires = solve_burgers(x_hires, t_hires, nu, A, k)

    # Downsample to target grid
    u = downsample_solution(u_hires, (target_nt, target_nx))

    # Create target grids for verification
    x = create_1d_grid(x_domain, target_nx)
    t = create_1d_grid(t_domain, target_nt)

    # Verify solution
    if verify:
        thresholds = verify_thresholds or {}
        is_valid, metrics = verify_solution_burgers(
            solution=u,
            x_coords=x,
            t_coords=t,
            nu=nu,
            A=A,
            k=k,
            **thresholds,
        )
    else:
        is_valid = True
        metrics = {}

    return u, x, t, is_valid, metrics


if __name__ == "__main__":
    # Test generation
    print("Testing Burgers solver...")

    # Use parameters from inspection (smaller amplitude for better stability)
    nu = 0.15
    A = 1.0
    k = 1.0

    u, x, t, is_valid, metrics = generate_burgers_sample(
        nu=nu,
        A=A,
        k=k,
        solver_nx=512,  # Extra fine for nonlinear problem
        solver_nt=512,
    )

    print(f"\nGenerated solution shape: {u.shape}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"Valid: {is_valid}")
    print(f"Metrics: {metrics}")
