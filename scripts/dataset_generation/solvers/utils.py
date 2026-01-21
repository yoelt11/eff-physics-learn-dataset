"""Shared utilities for PDE solvers."""

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import zoom


def create_1d_grid(domain: tuple[float, float], n: int) -> np.ndarray:
    """Create 1D uniform grid.

    Args:
        domain: (min, max) domain bounds
        n: number of grid points

    Returns:
        1D array of grid points
    """
    return np.linspace(domain[0], domain[1], n)


def create_2d_grid(x_domain: tuple[float, float], y_domain: tuple[float, float],
                   nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Create 2D meshgrid.

    Args:
        x_domain: (xmin, xmax)
        y_domain: (ymin, ymax)
        nx: number of x grid points
        ny: number of y grid points

    Returns:
        X_grid, Y_grid meshgrids with shape (nx, ny)
    """
    x = np.linspace(x_domain[0], x_domain[1], nx)
    y = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def finite_diff_1d(u: jnp.ndarray, dx: float, order: int = 2, axis: int = -1) -> jnp.ndarray:
    """Compute first derivative using centered finite differences.

    Args:
        u: Array to differentiate
        dx: Grid spacing
        order: Derivative order (1 or 2)
        axis: Axis along which to differentiate

    Returns:
        du/dx or d²u/dx² array (same shape as u)
    """
    if order == 1:
        # Centered difference: (u[i+1] - u[i-1]) / (2*dx)
        return (jnp.roll(u, -1, axis=axis) - jnp.roll(u, 1, axis=axis)) / (2 * dx)
    elif order == 2:
        # Centered second derivative: (u[i+1] - 2*u[i] + u[i-1]) / dx²
        return (jnp.roll(u, -1, axis=axis) - 2 * u + jnp.roll(u, 1, axis=axis)) / (dx ** 2)
    else:
        raise ValueError(f"Unsupported order: {order}")


def enforce_dirichlet_bc(u: np.ndarray, value: float = 0.0, axis: int = -1) -> np.ndarray:
    """Enforce Dirichlet boundary conditions.

    Args:
        u: Solution array
        value: BC value
        axis: Spatial axis for BC

    Returns:
        Modified u with BC enforced
    """
    u = u.copy()
    if axis == -1 or axis == u.ndim - 1:
        u[..., 0] = value
        u[..., -1] = value
    elif axis == 0:
        u[0, ...] = value
        u[-1, ...] = value
    elif axis == 1:
        u[:, 0, ...] = value
        u[:, -1, ...] = value
    return u


def enforce_periodic_bc(u: np.ndarray, axis: int = -1) -> np.ndarray:
    """Enforce periodic boundary conditions (for verification).

    Args:
        u: Solution array
        axis: Spatial axis for BC

    Returns:
        Modified u with periodic BC
    """
    # For periodic BC, boundary values should already match due to solver
    # This is mainly for verification purposes
    return u


def downsample_solution(u: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Downsample solution using scipy zoom (order=3 spline interpolation).

    Args:
        u: High-resolution solution
        target_shape: Desired output shape

    Returns:
        Downsampled solution
    """
    zoom_factors = tuple(target_shape[i] / u.shape[i] for i in range(len(target_shape)))
    return zoom(u, zoom_factors, order=3)


def create_laplacian_2d(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    """Create sparse Laplacian matrix for 2D Poisson/Helmholtz equation.

    Args:
        nx: Number of x grid points
        ny: Number of y grid points
        dx: x grid spacing
        dy: y grid spacing

    Returns:
        Laplacian matrix as dense array (n, n) where n = nx * ny
    """
    from scipy.sparse import diags, kron, eye

    # 1D Laplacian operators
    Dx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / (dx ** 2)
    Dy = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / (dy ** 2)

    # 2D Laplacian via Kronecker product
    Ix = eye(nx)
    Iy = eye(ny)
    L = kron(Ix, Dy) + kron(Dx, Iy)

    return L.toarray()


def apply_laplacian_2d_fd(u: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Apply 2D Laplacian using finite differences.

    Args:
        u: 2D array (nx, ny)
        dx: x grid spacing
        dy: y grid spacing

    Returns:
        Laplacian of u (same shape)
    """
    d2u_dx2 = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / (dx ** 2)
    d2u_dy2 = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / (dy ** 2)
    return d2u_dx2 + d2u_dy2


def sample_parameters(param_ranges: dict, n_samples: int, seed: int = 0) -> dict:
    """Sample parameters uniformly from given ranges.

    Args:
        param_ranges: Dict mapping param name to {"min": float, "max": float}
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Dict mapping param name to array of sampled values (n_samples,)
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for name, bounds in param_ranges.items():
        samples[name] = rng.uniform(bounds["min"], bounds["max"], size=n_samples)
    return samples
