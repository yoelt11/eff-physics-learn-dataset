"""Generate verified Helmholtz3D equation dataset using analytical solution.

PDE: Δu + k²u = q(x,y,z) (manufactured solution)
BC: Dirichlet data implied by the exact solution on the boundary
Exact solution: u(x,y,z) = sin(a₁πx)sin(a₂πy)sin(a₃πz)
Source: q = [-(a₁π)² - (a₂π)² - (a₃π)² + k²] sin(a₁πx)sin(a₂πy)sin(a₃πz)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .utils import validate_solution_coordinates


def exact_solution_helmholtz3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, a1: float, a2: float, a3: float
) -> np.ndarray:
    """Exact solution u = sin(a₁πx)sin(a₂πy)sin(a₃πz)."""
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y) * np.sin(a3 * np.pi * z)


def verify_analytical_helmholtz3d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: float,
    a1: float,
    a2: float,
    a3: float,
    pde_threshold: float = 100.0,
    bc_threshold: float = 1e-4,
) -> tuple[bool, dict]:
    """Verify Helmholtz3D solution using analytical derivatives."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    sin_term = np.sin(a1 * np.pi * X) * np.sin(a2 * np.pi * Y) * np.sin(a3 * np.pi * Z)
    laplacian_analytical = (
        -(a1 * np.pi) ** 2 - (a2 * np.pi) ** 2 - (a3 * np.pi) ** 2
    ) * sin_term

    q_expected = (
        -(a1 * np.pi) ** 2 - (a2 * np.pi) ** 2 - (a3 * np.pi) ** 2 + k**2
    ) * sin_term
    pde_residual = laplacian_analytical + k**2 * u - q_expected
    pde_loss = float(np.mean(pde_residual**2))

    Y_yz, Z_yz = np.meshgrid(y, z, indexing="ij")
    bc_x0 = u[0, :, :]
    exp_x0 = exact_solution_helmholtz3d(x[0] * np.ones_like(Y_yz), Y_yz, Z_yz, a1, a2, a3)
    bc_x1 = u[-1, :, :]
    exp_x1 = exact_solution_helmholtz3d(x[-1] * np.ones_like(Y_yz), Y_yz, Z_yz, a1, a2, a3)

    X_xz, Z_xz = np.meshgrid(x, z, indexing="ij")
    bc_y0 = u[:, 0, :]
    exp_y0 = exact_solution_helmholtz3d(X_xz, y[0] * np.ones_like(X_xz), Z_xz, a1, a2, a3)
    bc_y1 = u[:, -1, :]
    exp_y1 = exact_solution_helmholtz3d(X_xz, y[-1] * np.ones_like(X_xz), Z_xz, a1, a2, a3)

    X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")
    bc_z0 = u[:, :, 0]
    exp_z0 = exact_solution_helmholtz3d(X_xy, Y_xy, z[0] * np.ones_like(X_xy), a1, a2, a3)
    bc_z1 = u[:, :, -1]
    exp_z1 = exact_solution_helmholtz3d(X_xy, Y_xy, z[-1] * np.ones_like(X_xy), a1, a2, a3)

    bc_loss = float(
        np.mean(
            [
                np.mean((bc_x0 - exp_x0) ** 2),
                np.mean((bc_x1 - exp_x1) ** 2),
                np.mean((bc_y0 - exp_y0) ** 2),
                np.mean((bc_y1 - exp_y1) ** 2),
                np.mean((bc_z0 - exp_z0) ** 2),
                np.mean((bc_z1 - exp_z1) ** 2),
            ]
        )
    )

    pde_ok = pde_loss < pde_threshold
    bc_ok = bc_loss < bc_threshold
    is_valid = pde_ok and bc_ok

    metrics = {
        "pde_loss": pde_loss,
        "bc_loss": bc_loss,
        "ic_loss": 0.0,
        "pde_ok": pde_ok,
        "bc_ok": bc_ok,
        "ic_ok": True,
    }

    return is_valid, metrics


def generate_helmholtz3d_sample_analytical(
    k: float,
    a1: float,
    a2: float,
    a3: float,
    x_domain: tuple[float, float] = (-1.0, 1.0),
    y_domain: tuple[float, float] = (-1.0, 1.0),
    z_domain: tuple[float, float] = (-1.0, 1.0),
    target_nx: int = 32,
    target_ny: int = 32,
    target_nz: int = 32,
    verify: bool = True,
    verify_thresholds: dict | None = None,
    backend: Literal["numpy", "jax"] = "numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict]:
    """Generate a single verified Helmholtz3D solution using the analytical formula.

    Returns:
        (solution, x_coords, y_coords, z_coords, is_valid, metrics)
        solution has shape (target_nx, target_ny, target_nz).
    """
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        xj = jnp.linspace(x_domain[0], x_domain[1], target_nx)
        yj = jnp.linspace(y_domain[0], y_domain[1], target_ny)
        zj = jnp.linspace(z_domain[0], z_domain[1], target_nz)
        X, Y, Z = jnp.meshgrid(xj, yj, zj, indexing="ij")
        u = jnp.sin(a1 * jnp.pi * X) * jnp.sin(a2 * jnp.pi * Y) * jnp.sin(a3 * jnp.pi * Z)
        x = np.asarray(jax.device_get(xj))
        y = np.asarray(jax.device_get(yj))
        z = np.asarray(jax.device_get(zj))
        u = np.asarray(jax.device_get(u))
    else:
        x = np.linspace(x_domain[0], x_domain[1], target_nx)
        y = np.linspace(y_domain[0], y_domain[1], target_ny)
        z = np.linspace(z_domain[0], z_domain[1], target_nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        u = exact_solution_helmholtz3d(X, Y, Z, a1, a2, a3)

    validate_solution_coordinates(
        u,
        coords={"x": x, "y": y, "z": z},
        axis_map={"x": 0, "y": 1, "z": 2},
        context="Helmholtz3D output",
    )

    if verify:
        thresholds = verify_thresholds or {}
        is_valid, metrics = verify_analytical_helmholtz3d(
            u=u,
            x=x,
            y=y,
            z=z,
            k=k,
            a1=a1,
            a2=a2,
            a3=a3,
            **thresholds,
        )
    else:
        is_valid = True
        metrics = {}

    return u, x, y, z, is_valid, metrics


if __name__ == "__main__":
    u, x, y, z, ok, m = generate_helmholtz3d_sample_analytical(
        k=2.5, a1=2.0, a2=1.5, a3=2.2, target_nx=24, target_ny=24, target_nz=24
    )
    print(f"shape={u.shape} valid={ok} metrics={m}")
