"""Generate verified Burgers equation dataset.

PDE: u_t + u·u_x = ν u_xx
BC: Periodic
IC: u(x,0) = A sin(kπx), with integer mode k
"""

from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from ..verification import verify_solution_burgers

from .utils import (
    create_1d_grid,
    downsample_solution,
    downsample_solution_jax,
    validate_solution_coordinates,
)


def initial_condition_burgers(x: np.ndarray, A: float, k: int) -> np.ndarray:
    """Burgers initial condition: u(x,0) = A sin(kπx).

    For periodic BC on ``[-1, 1]``, ``k`` must be an integer Fourier mode.

    Args:
        x: 1D spatial grid (assumed domain [-1, 1])
        A: Amplitude
        k: Integer wavenumber (Fourier mode)

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
    k: int,
) -> np.ndarray:
    """Solve Burgers equation using scipy.integrate.solve_ivp.

    Args:
        x: Spatial grid (n_x,)
        t: Temporal grid (n_t,)
        nu: Viscosity
        A: IC amplitude
        k: Integer IC wavenumber (Fourier mode)

    Returns:
        Solution u with shape (n_t, n_x) - transposed indexing!
    """
    # Enforce periodic consistency for IC on [-1, 1].
    if not np.isclose(k, round(k), atol=1e-12):
        raise ValueError(
            "For periodic BC on [-1, 1], k must be an integer so "
            "u(x,0)=A sin(k*pi*x) is periodic."
        )

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


def _solve_burgers_jax(
    *,
    x: np.ndarray,
    t: np.ndarray,
    nu: float,
    A: float,
    k: int,
) -> np.ndarray:
    """JAX-jitted Burgers solver on a periodic domain (returns NumPy array).

    Uses pseudo-spectral derivatives in x and RK4 in time.
    """
    if not np.isclose(k, round(k), atol=1e-12):
        raise ValueError(
            "For periodic BC on [-1, 1], k must be an integer so "
            "u(x,0)=A sin(k*pi*x) is periodic."
        )

    import jax
    import jax.numpy as jnp
    from jax import lax

    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt):
        raise ValueError("JAX Burgers backend requires a uniform temporal grid")

    n = x.shape[0]
    length = float(x[-1] - x[0])
    kk = 2.0 * np.pi * np.fft.fftfreq(n, d=length / (n - 1))
    ik = 1j * kk
    k2 = kk**2

    ikj = jnp.asarray(ik)
    k2j = jnp.asarray(k2)

    u0 = initial_condition_burgers(x, A, k)
    u0j = jnp.asarray(u0, dtype=jnp.float64)

    def rhs(u):
        u_hat = jnp.fft.fft(u)
        u_x = jnp.fft.ifft(ikj * u_hat).real
        u_xx = jnp.fft.ifft(-k2j * u_hat).real
        return -u * u_x + (nu * u_xx)

    def step(u, _):
        k1 = rhs(u)
        k2_ = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2_)
        k4 = rhs(u + dt * k3)
        u_next = u + (dt / 6.0) * (k1 + 2.0 * k2_ + 2.0 * k3 + k4)
        return u_next, u_next

    _, tail = lax.scan(step, u0j, xs=None, length=len(t) - 1)
    U = jnp.concatenate([u0j[None, :], tail], axis=0)
    return np.asarray(jax.device_get(U), dtype=np.float64)


def generate_burgers_sample(
    nu: float,
    A: float,
    k: int,
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
    """Generate a single verified Burgers solution.

    Args:
        nu: Viscosity
        A: IC amplitude
        k: Integer IC wavenumber (Fourier mode)
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
    if backend == "jax":
        u_hires = _solve_burgers_jax(x=x_hires, t=t_hires, nu=nu, A=A, k=k)
    elif backend == "numpy":
        u_hires = solve_burgers(x_hires, t_hires, nu, A, k)
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")

    # Downsample to target grid
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        u = np.asarray(jax.device_get(downsample_solution_jax(jnp.asarray(u_hires), (target_nt, target_nx))), dtype=np.float64)
    else:
        u = downsample_solution(u_hires, (target_nt, target_nx))

    # Create target grids for verification
    x = create_1d_grid(x_domain, target_nx)
    t = create_1d_grid(t_domain, target_nt)

    validate_solution_coordinates(
        u,
        coords={"t": t, "x": x},
        axis_map={"t": 0, "x": 1},
        context="Burgers output",
    )

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
    k = 1

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
