"""Generate verified Allen-Cahn equation dataset.

PDE: u_t = ε² u_xx - λ(u³ - u)
BC: Homogeneous Dirichlet (u = 0 at x = ±1)
IC: u(x,0) = (1 - x²) cos(πx)
"""

from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from ..verification import verify_solution_allen_cahn

from .utils import (
    create_1d_grid,
    downsample_solution,
    downsample_solution_jax,
    validate_solution_coordinates,
)


def initial_condition_allen_cahn(x: np.ndarray) -> np.ndarray:
    """Allen-Cahn initial condition: u(x,0) = (1 - x²) cos(πx).

    Args:
        x: 1D spatial grid

    Returns:
        Initial condition array
    """
    return (1.0 - x**2) * np.cos(np.pi * x)


def spatial_derivative_allen_cahn(u: np.ndarray, dx: float, eps: float, lam: float) -> np.ndarray:
    """Compute spatial derivative for Allen-Cahn equation.

    RHS: ε² u_xx - λ(u³ - u)

    Args:
        u: Solution at current time (n_x,)
        dx: Spatial grid spacing
        eps: Diffusion coefficient
        lam: Reaction parameter

    Returns:
        du/dt array
    """
    # Second derivative using centered finite differences
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)

    # Enforce BC: u_xx = 0 at boundaries (since u = 0 there)
    u_xx[0] = 0.0
    u_xx[-1] = 0.0

    # RHS: ε² u_xx - λ(u³ - u)
    return eps**2 * u_xx - lam * (u**3 - u)


def solve_allen_cahn(
    x: np.ndarray,
    t: np.ndarray,
    eps: float,
    lam: float,
    enforce_bc: bool = True,
) -> np.ndarray:
    """Solve Allen-Cahn equation using scipy.integrate.solve_ivp.

    Args:
        x: Spatial grid (n_x,)
        t: Temporal grid (n_t,)
        eps: Diffusion coefficient
        lam: Reaction parameter
        enforce_bc: If True, enforce Dirichlet BC at each time step

    Returns:
        Solution u with shape (n_x, n_t) - standard indexing: u[i, j] = u(x[i], t[j])
    """
    dx = x[1] - x[0]
    n_x = len(x)

    # Initial condition
    u0 = initial_condition_allen_cahn(x)

    # Enforce Dirichlet BC on IC
    u0[0] = 0.0
    u0[-1] = 0.0

    # Define ODE system
    def rhs(t_val, u_flat):
        u = u_flat.reshape(n_x)
        dudt = spatial_derivative_allen_cahn(u, dx, eps, lam)
        # Enforce BC on derivative (boundary remains 0)
        dudt[0] = 0.0
        dudt[-1] = 0.0
        return dudt.flatten()

    # Solve using solve_ivp (use DOP853 for high accuracy)
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

    # Keep standard indexing (n_x, n_t) to match meshgrid(..., indexing='ij')
    u = sol.y

    # Enforce BC at all time steps (x boundaries)
    if enforce_bc:
        u[0, :] = 0.0
        u[-1, :] = 0.0

    return u


def _solve_allen_cahn_jax(
    *,
    x: np.ndarray,
    t: np.ndarray,
    eps: float,
    lam: float,
) -> np.ndarray:
    """JAX-jitted Allen–Cahn solver (returns NumPy array).

    Uses second-order finite differences in x with Dirichlet BC (u=0 at both ends)
    and RK4 in time.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt):
        raise ValueError("JAX Allen–Cahn backend requires a uniform temporal grid")

    dx = float(x[1] - x[0])
    n = x.shape[0]

    u0 = initial_condition_allen_cahn(x)
    u0[0] = 0.0
    u0[-1] = 0.0
    u0j = jnp.asarray(u0, dtype=jnp.float64)

    def laplacian(u):
        out = jnp.zeros_like(u)
        core = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx**2)
        out = out.at[1:-1].set(core)
        return out

    def rhs(u):
        u_xx = laplacian(u)
        return (eps**2) * u_xx - lam * (u**3 - u)

    def enforce_bc(u):
        return u.at[0].set(0.0).at[-1].set(0.0)

    def step(u, _):
        k1 = rhs(u)
        k2 = rhs(enforce_bc(u + 0.5 * dt * k1))
        k3 = rhs(enforce_bc(u + 0.5 * dt * k2))
        k4 = rhs(enforce_bc(u + dt * k3))
        u_next = enforce_bc(u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        return u_next, u_next

    _, tail = lax.scan(step, u0j, xs=None, length=len(t) - 1)
    U = jnp.concatenate([u0j[None, :], tail], axis=0)
    return np.asarray(jax.device_get(U), dtype=np.float64)


def generate_allen_cahn_sample(
    eps: float,
    lam: float,
    x_domain: tuple[float, float] = (-1.0, 1.0),
    t_domain: tuple[float, float] = (0.0, 1.0),
    solver_nx: int = 128,
    solver_nt: int = 128,
    target_nx: int = 64,
    target_nt: int = 64,
    verify: bool = True,
    verify_thresholds: dict | None = None,
    backend: Literal["numpy", "jax"] = "numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
    """Generate a single verified Allen-Cahn solution.

    Args:
        eps: Diffusion coefficient
        lam: Reaction parameter
        x_domain: Spatial domain (xmin, xmax)
        t_domain: Temporal domain (tmin, tmax)
        solver_nx: High-res spatial grid size
        solver_nt: High-res temporal grid size
        target_nx: Target spatial grid size (for downsampling)
        target_nt: Target temporal grid size (for downsampling)
        verify: If True, verify solution
        verify_thresholds: Custom verification thresholds

    Returns:
        (solution, x_coords, t_coords, is_valid, metrics)
        solution has shape (target_nx, target_nt) - standard indexing
    """
    # Create high-res grids
    x_hires = create_1d_grid(x_domain, solver_nx)
    t_hires = create_1d_grid(t_domain, solver_nt)

    # Solve on high-res grid
    if backend == "jax":
        u_hires = _solve_allen_cahn_jax(x=x_hires, t=t_hires, eps=eps, lam=lam)
    elif backend == "numpy":
        u_hires = solve_allen_cahn(x_hires, t_hires, eps, lam)
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")

    # Downsample to target grid
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        u = np.asarray(
            jax.device_get(downsample_solution_jax(jnp.asarray(u_hires), (target_nx, target_nt))),
            dtype=np.float64,
        )
    else:
        u = downsample_solution(u_hires, (target_nx, target_nt))

    # Create target grids for verification
    x = create_1d_grid(x_domain, target_nx)
    t = create_1d_grid(t_domain, target_nt)

    validate_solution_coordinates(
        u,
        coords={"t": t, "x": x},
        axis_map={"x": 0, "t": 1},
        context="Allen-Cahn output",
    )

    # Verify solution
    if verify:
        thresholds = verify_thresholds or {}
        is_valid, metrics = verify_solution_allen_cahn(
            solution=u,
            x_coords=x,
            t_coords=t,
            eps=eps,
            lam=lam,
            **thresholds,
        )
    else:
        is_valid = True
        metrics = {}

    return u, x, t, is_valid, metrics


if __name__ == "__main__":
    # Test generation
    print("Testing Allen-Cahn solver...")

    # Use parameters from inspection
    eps = 0.0005
    lam = 2.5

    u, x, t, is_valid, metrics = generate_allen_cahn_sample(
        eps=eps,
        lam=lam,
        solver_nx=256,  # Extra high-res for testing
        solver_nt=256,
    )

    print(f"\nGenerated solution shape: {u.shape}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"Valid: {is_valid}")
    print(f"Metrics: {metrics}")
