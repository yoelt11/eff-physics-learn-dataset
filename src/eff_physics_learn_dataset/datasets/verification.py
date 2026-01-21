"""Lightweight verification functions for validating PDE solutions during generation.

This module provides standalone verification functions that can be used during
dataset generation without requiring the full dataset loader infrastructure.

Dependencies: numpy only

Usage during generation:
    import numpy as np
    from src.data.verification import verify_solution_allen_cahn

    # After solving PDE
    is_valid, metrics = verify_solution_allen_cahn(
        solution=u,
        x_coords=x_1d,
        t_coords=t_1d,
        eps=eps_val,
        lam=lam_val
    )

    if not is_valid:
        print(f"Warning: Solution validation failed - {metrics}")
"""

import numpy as np
from typing import Dict, Tuple


def compute_derivatives_transposed(u: np.ndarray, x_coords: np.ndarray,
                                   t_coords: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute numerical derivatives for transposed indexing: u[i,j] = u(x[j], t[i]).

    Args:
        u: Solution array of shape (n_t, n_x) where u[i, j] = u(x[j], t[i])
        x_coords: 1D spatial coordinate array
        t_coords: 1D temporal coordinate array

    Returns:
        Dictionary with u, u_t, u_x, u_xx
    """
    u_t = np.gradient(u, t_coords, axis=0)  # t varies along axis 0
    u_x = np.gradient(u, x_coords, axis=1)  # x varies along axis 1
    u_xx = np.gradient(u_x, x_coords, axis=1)

    return {'u': u, 'u_t': u_t, 'u_x': u_x, 'u_xx': u_xx}


def compute_derivatives_standard(u: np.ndarray, x_coords: np.ndarray,
                                 y_coords: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute numerical derivatives for standard indexing: u[i,j] = u(x[i], y[j]).

    Args:
        u: Solution array of shape (n_x, n_y) where u[i, j] = u(x[i], y[j])
        x_coords: 1D x coordinate array
        y_coords: 1D y coordinate array

    Returns:
        Dictionary with u, u_x, u_xx, u_y, u_yy
    """
    u_x = np.gradient(u, x_coords, axis=0)
    u_xx = np.gradient(u_x, x_coords, axis=0)
    u_y = np.gradient(u, y_coords, axis=1)
    u_yy = np.gradient(u_y, y_coords, axis=1)

    return {'u': u, 'u_x': u_x, 'u_xx': u_xx, 'u_y': u_y, 'u_yy': u_yy}


def verify_solution_allen_cahn(solution: np.ndarray, x_coords: np.ndarray,
                                t_coords: np.ndarray, eps: float, lam: float,
                                pde_threshold: float = 0.1,
                                bc_threshold: float = 1e-3,
                                ic_threshold: float = 0.1) -> Tuple[bool, Dict]:
    """Verify Allen-Cahn solution satisfies PDE/BC/IC constraints.

    Args:
        solution: Solution array (n_t, n_x) with transposed indexing
        x_coords: 1D spatial coordinates
        t_coords: 1D temporal coordinates
        eps: Diffusion coefficient
        lam: Reaction parameter
        pde_threshold: Maximum acceptable PDE residual MSE
        bc_threshold: Maximum acceptable BC error MSE
        ic_threshold: Maximum acceptable IC error MSE

    Returns:
        (is_valid, metrics) where metrics contains loss values
    """
    # Compute derivatives
    derivs = compute_derivatives_transposed(solution, x_coords, t_coords)
    u = derivs['u']
    u_t = derivs['u_t']
    u_xx = derivs['u_xx']

    # PDE residual: u_t - eps^2 * u_xx + lam * (u^3 - u) = 0
    pde_residual = u_t - eps**2 * u_xx + lam * (u**3 - u)
    pde_loss = float(np.mean(pde_residual**2))

    # BC: u = 0 at x = x_min and x_max (homogeneous Dirichlet)
    bc_left = u[:, 0]
    bc_right = u[:, -1]
    bc_loss = float(np.mean(bc_left**2) + np.mean(bc_right**2))

    # IC: u(x, t=0) = x^2 * cos(π*x)
    ic_expected = x_coords**2 * np.cos(np.pi * x_coords)
    ic_actual = u[0, :]
    ic_loss = float(np.mean((ic_actual - ic_expected)**2))

    # Check validity
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


def verify_solution_burgers(solution: np.ndarray, x_coords: np.ndarray,
                            t_coords: np.ndarray, nu: float, A: float, k: float,
                            pde_threshold: float = 5.0,
                            bc_threshold: float = 0.1,
                            ic_threshold: float = 1e-3) -> Tuple[bool, Dict]:
    """Verify Burgers solution satisfies PDE/BC/IC constraints.

    Args:
        solution: Solution array (n_t, n_x) with transposed indexing
        x_coords: 1D spatial coordinates
        t_coords: 1D temporal coordinates
        nu: Viscosity coefficient
        A: IC amplitude parameter
        k: IC wavenumber parameter
        pde_threshold: Maximum acceptable PDE residual MSE
        bc_threshold: Maximum acceptable BC error MSE
        ic_threshold: Maximum acceptable IC error MSE

    Returns:
        (is_valid, metrics) where metrics contains loss values
    """
    derivs = compute_derivatives_transposed(solution, x_coords, t_coords)
    u = derivs['u']
    u_t = derivs['u_t']
    u_x = derivs['u_x']
    u_xx = derivs['u_xx']

    # PDE residual: u_t + u * u_x - nu * u_xx = 0
    pde_residual = u_t + u * u_x - nu * u_xx
    pde_loss = float(np.mean(pde_residual**2))

    # BC: Periodic - u(x_min, t) = u(x_max, t)
    bc_left = u[:, 0]
    bc_right = u[:, -1]
    bc_loss = float(np.mean((bc_left - bc_right)**2))

    # IC: u(x, t=0) = A sin(kπx)
    ic_expected = A * np.sin(k * np.pi * x_coords)
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


def verify_solution_convection(solution: np.ndarray, x_coords: np.ndarray,
                               t_coords: np.ndarray, beta: float,
                               pde_threshold: float = 1.0,
                               bc_threshold: float = 1e-3,
                               ic_threshold: float = 1e-3) -> Tuple[bool, Dict]:
    """Verify Convection solution satisfies PDE/BC/IC constraints.

    Args:
        solution: Solution array (n_t, n_x) with transposed indexing
        x_coords: 1D spatial coordinates
        t_coords: 1D temporal coordinates
        beta: Convection velocity
        pde_threshold: Maximum acceptable PDE residual MSE
        bc_threshold: Maximum acceptable BC error MSE
        ic_threshold: Maximum acceptable IC error MSE

    Returns:
        (is_valid, metrics) where metrics contains loss values
    """
    derivs = compute_derivatives_transposed(solution, x_coords, t_coords)
    u = derivs['u']
    u_t = derivs['u_t']
    u_x = derivs['u_x']

    # PDE residual: u_t + beta * u_x = 0
    pde_residual = u_t + beta * u_x
    pde_loss = float(np.mean(pde_residual**2))

    # BC: Periodic - u(x=0, t) = u(x=L, t)
    bc_left = u[:, 0]
    bc_right = u[:, -1]
    bc_loss = float(np.mean((bc_left - bc_right)**2))

    # IC: u(x, t=0) = 1 + sin(2πx/L)
    L = x_coords[-1] - x_coords[0]
    ic_expected = 1.0 + np.sin(2 * np.pi * x_coords / L)
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


def verify_solution_helmholtz2d(solution: np.ndarray, x_coords: np.ndarray,
                                y_coords: np.ndarray, k: float, a1: float, a2: float,
                                pde_threshold: float = 100.0,
                                bc_threshold: float = 1e-4) -> Tuple[bool, Dict]:
    """Verify Helmholtz2D solution satisfies PDE/BC constraints.

    Args:
        solution: Solution array (n_x, n_y) with standard indexing
        x_coords: 1D x coordinates
        y_coords: 1D y coordinates
        k: Wave number
        a1: x-direction parameter
        a2: y-direction parameter
        pde_threshold: Maximum acceptable PDE residual MSE
        bc_threshold: Maximum acceptable BC error MSE

    Returns:
        (is_valid, metrics) where metrics contains loss values
    """
    derivs = compute_derivatives_standard(solution, x_coords, y_coords)
    u = derivs['u']
    u_xx = derivs['u_xx']
    u_yy = derivs['u_yy']

    # Manufactured source term q(x,y)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    q_coeff = -(a1 * np.pi)**2 - (a2 * np.pi)**2 + k**2
    q = q_coeff * np.sin(a1 * np.pi * X_grid) * np.sin(a2 * np.pi * Y_grid)

    # PDE residual: Δu + k²u - q = 0
    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = float(np.mean(pde_residual**2))

    # BC: Dirichlet on all boundaries
    # Exact solution: u(x,y) = sin(a₁πx) sin(a₂πy)
    bc_bottom = u[:, 0]
    bc_top = u[:, -1]
    bc_left = u[0, :]
    bc_right = u[-1, :]

    bc_bottom_expected = np.sin(a1 * np.pi * x_coords) * np.sin(a2 * np.pi * y_coords[0])
    bc_top_expected = np.sin(a1 * np.pi * x_coords) * np.sin(a2 * np.pi * y_coords[-1])
    bc_left_expected = np.sin(a1 * np.pi * x_coords[0]) * np.sin(a2 * np.pi * y_coords)
    bc_right_expected = np.sin(a1 * np.pi * x_coords[-1]) * np.sin(a2 * np.pi * y_coords)

    bc_loss = float(
        np.mean((bc_bottom - bc_bottom_expected)**2) +
        np.mean((bc_top - bc_top_expected)**2) +
        np.mean((bc_left - bc_left_expected)**2) +
        np.mean((bc_right - bc_right_expected)**2)
    )

    pde_ok = pde_loss < pde_threshold
    bc_ok = bc_loss < bc_threshold
    is_valid = pde_ok and bc_ok

    metrics = {
        'pde_loss': pde_loss,
        'bc_loss': bc_loss,
        'ic_loss': 0.0,
        'pde_ok': pde_ok,
        'bc_ok': bc_ok,
        'ic_ok': True
    }

    return is_valid, metrics
