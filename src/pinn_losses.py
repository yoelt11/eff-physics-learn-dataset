"""JAX-based PINN loss functions for PDEs matching the dataset solvers.

All functions use JAX (jax.numpy) and are compatible with JIT and autodiff.
The model must provide a method `fn_derivatives(X, params)` returning a dict
with keys such as 'u', 'u_t', 'u_x', 'u_xx', etc., depending on the equation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp


def compute_derivatives(
    X: jnp.ndarray,
    params: Any,
    model: Any,
    grid_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, jnp.ndarray]:
    """Compute derivatives via the model's derivative function.

    The model must implement a method `fn_derivatives(X, params)` returning
    a dict of arrays (e.g. 'u', 'u_t', 'u_x', 'u_xx'). Optional grid_shape
    may be passed for models that need it (e.g. grid-based architectures).

    Args:
        X: Evaluation points (N, D)
        params: Model parameters
        model: Object with callable fn_derivatives(X, params) -> dict
        grid_shape: Optional grid dimensions for reshaping (e.g. (n_x, n_t))

    Returns:
        Dictionary of derivative arrays
    """
    fn = getattr(model, "fn_derivatives", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            "Model must provide a callable fn_derivatives(X, params) returning "
            "a dict of derivative arrays (e.g. 'u', 'u_t', 'u_x', 'u_xx')."
        )
    if grid_shape is not None:
        try:
            return fn(X, params, grid_shape=grid_shape)
        except TypeError:
            return fn(X, params)
    return fn(X, params)


def _default_grid_shape_2d(X: jnp.ndarray) -> Tuple[int, int]:
    n = int(X.shape[0])
    side = int(n**0.5)
    return (side, side)


def _reshape_xt_standard(
    values: jnp.ndarray,
    X: jnp.ndarray,
    grid_shape: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reshape flattened fields to standard (n_x, n_t) layout.

    Convention:
    - X columns are [x, t]
    - grids are built with meshgrid(x, t, indexing='ij')
    - flattened in C-order
    """
    n_x, n_t = grid_shape
    return (
        values.reshape(n_x, n_t),
        X[:, 0].reshape(n_x, n_t),
        X[:, 1].reshape(n_x, n_t),
    )


def cdr_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    D: float = 0.01,
    v: float = 1.0,
    R: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for Convection-Diffusion-Reaction: ∂u/∂t + v ∂u/∂x = D ∂²u/∂x² - Ru."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_y"))
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]

    pde_residual = u_t + v * u_x - D * u_xx + R * u
    pde_loss = jnp.mean(pde_residual**2)

    if grid_shape is None:
        grid_shape = _default_grid_shape_2d(X)
    n_x, n_t = grid_shape

    u_reshaped, x_coords, _ = _reshape_xt_standard(u, X, grid_shape)

    u_left = u_reshaped[0, :]
    u_right = u_reshaped[-1, :]
    bc_loss = jnp.mean((u_left - u_right) ** 2)

    def ic(x_vals: jnp.ndarray) -> jnp.ndarray:
        L = jnp.max(x_vals) - jnp.min(x_vals)
        return jnp.sin(2 * jnp.pi * x_vals / L)

    u_ic_pred = u_reshaped[:, 0]
    x_ic = x_coords[:, 0]
    ic_loss = jnp.mean((u_ic_pred - ic(x_ic)) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        u_gt = ground_truth_solution.ravel()
        zero_shot_bias_loss = jnp.mean((u - u_gt) ** 2)

    total = (
        pde_weight * pde_loss
        + bc_weight * bc_loss
        + ic_weight * ic_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": bc_weight * bc_loss,
            "ic_loss": ic_weight * ic_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def allen_cahn_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    eps: float = 0.01,
    lam: float = 1.0,
    bc_weight: float = 50.0,
    ic_weight: float = 100.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for Allen-Cahn: u_t - ε² u_xx + λ(u³ - u) = 0."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_y"))
    u_xx = derivs["u_xx"]

    pde_residual = u_t - eps**2 * u_xx + lam * (u**3 - u)
    pde_loss = jnp.mean(pde_residual**2)

    if grid_shape is None:
        grid_shape = _default_grid_shape_2d(X)
    n_x, n_t = grid_shape

    u_reshaped, x_coords, _ = _reshape_xt_standard(u, X, grid_shape)

    u_left_pred = u_reshaped[0, :]
    u_right_pred = u_reshaped[-1, :]
    bc_loss = jnp.mean(u_left_pred**2) + jnp.mean(u_right_pred**2)

    def ic(x_vals: jnp.ndarray) -> jnp.ndarray:
        return x_vals**2 * jnp.cos(jnp.pi * x_vals)

    u_ic_pred = u_reshaped[:, 0]
    x_ic = x_coords[:, 0]
    ic_loss = jnp.mean((u_ic_pred - ic(x_ic)) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + bc_weight * bc_loss
        + ic_weight * ic_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": bc_weight * bc_loss,
            "ic_loss": ic_weight * ic_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def burgers_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    nu: float = 0.01,
    A: float = 1.0,
    k: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for Burgers: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x². IC: u(x,0)=A sin(kπx), periodic BC."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_y"))
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]

    pde_residual = u_t + u * u_x - nu * u_xx
    pde_loss = jnp.mean(pde_residual**2)

    if grid_shape is None:
        grid_shape = _default_grid_shape_2d(X)
    n_x, n_t = grid_shape

    u_reshaped, x_coords, _ = _reshape_xt_standard(u, X, grid_shape)

    bc_loss = jnp.mean((u_reshaped[0, :] - u_reshaped[-1, :]) ** 2)

    def ic(x_vals: jnp.ndarray) -> jnp.ndarray:
        return A * jnp.sin(k * jnp.pi * x_vals)

    u_ic_pred = u_reshaped[:, 0]
    x_ic = x_coords[:, 0]
    ic_loss = jnp.mean((u_ic_pred - ic(x_ic)) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + bc_weight * bc_loss
        + ic_weight * ic_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": bc_weight * bc_loss,
            "ic_loss": ic_weight * ic_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def convection_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    beta: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for pure convection: ∂u/∂t + β ∂u/∂x = 0. IC: 1+sin(2πx/L), periodic BC."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_y"))
    u_x = derivs["u_x"]

    pde_residual = u_t + beta * u_x
    pde_loss = jnp.mean(pde_residual**2)

    if grid_shape is None:
        grid_shape = _default_grid_shape_2d(X)
    n_x, n_t = grid_shape

    u_reshaped, x_coords, _ = _reshape_xt_standard(u, X, grid_shape)

    bc_loss = jnp.mean((u_reshaped[0, :] - u_reshaped[-1, :]) ** 2)

    def ic(x_vals: jnp.ndarray) -> jnp.ndarray:
        L = jnp.max(x_vals) - jnp.min(x_vals)
        return 1.0 + jnp.sin(2 * jnp.pi * x_vals / L)

    u_ic_pred = u_reshaped[:, 0]
    x_ic = x_coords[:, 0]
    ic_loss = jnp.mean((u_ic_pred - ic(x_ic)) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + bc_weight * bc_loss
        + ic_weight * ic_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": bc_weight * bc_loss,
            "ic_loss": ic_weight * ic_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def helmholtz_2d_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    k: float = 1.0,
    a1: float = 1.0,
    a2: float = 1.0,
    bc_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for 2D Helmholtz: Δu + k²u = q(x,y), Dirichlet BC from exact u = sin(a₁πx)sin(a₂πy)."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_xx = derivs["u_xx"]
    u_yy = derivs.get("u_yy", derivs.get("u_tt"))

    x, y = X[:, 0], X[:, 1]
    q_coeff = -(a1 * jnp.pi) ** 2 - (a2 * jnp.pi) ** 2 + k**2
    q = q_coeff * jnp.sin(a1 * jnp.pi * x) * jnp.sin(a2 * jnp.pi * y)

    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = jnp.mean(pde_residual**2)

    if grid_shape is None:
        grid_shape = _default_grid_shape_2d(X)
    n_x, n_y = grid_shape

    u_reshaped = u.reshape(n_x, n_y)
    x_coords = X[:, 0].reshape(n_x, n_y)
    y_coords = X[:, 1].reshape(n_x, n_y)

    def u_exact(xv: jnp.ndarray, yv: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(a1 * jnp.pi * xv) * jnp.sin(a2 * jnp.pi * yv)

    bc_loss = (
        jnp.mean((u_reshaped[:, 0] - u_exact(x_coords[:, 0], y_coords[:, 0])) ** 2)
        + jnp.mean((u_reshaped[:, -1] - u_exact(x_coords[:, -1], y_coords[:, -1])) ** 2)
        + jnp.mean((u_reshaped[0, :] - u_exact(x_coords[0, :], y_coords[0, :])) ** 2)
        + jnp.mean((u_reshaped[-1, :] - u_exact(x_coords[-1, :], y_coords[-1, :])) ** 2)
    )

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + bc_weight * bc_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": bc_weight * bc_loss,
            "ic_loss": 0.0,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": 0.0,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def ks_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for Kuramoto-Sivashinsky: u_t + α u u_x + β u_xx + γ u_xxxx = 0, periodic BC."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_y"))
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]
    u_xxxx = derivs["u_xxxx"]

    pde_residual = u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx
    pde_loss = jnp.mean(pde_residual**2)

    if grid_shape is None:
        grid_shape = _default_grid_shape_2d(X)
    n_x, n_t = grid_shape

    u_reshaped, x_coords, _ = _reshape_xt_standard(u, X, grid_shape)

    bc_loss = jnp.mean((u_reshaped[0, :] - u_reshaped[-1, :]) ** 2)

    def ic(x_vals: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(x_vals / 16.0) * (1.0 + jnp.sin(x_vals / 16.0))

    u_ic_pred = u_reshaped[:, 0]
    x_ic = x_coords[:, 0]
    ic_loss = jnp.mean((u_ic_pred - ic(x_ic)) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + bc_weight * bc_loss
        + ic_weight * ic_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": bc_weight * bc_loss,
            "ic_loss": ic_weight * ic_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def wave2d_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    c: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Loss for 2D wave: u_tt - c²(u_xx + u_yy) = 0. X has columns [x, y, t]."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u_tt = derivs["u_tt"]
    u_xx = derivs["u_xx"]
    u_yy = derivs["u_yy"]

    laplacian_u = u_xx + u_yy
    pde_residual = u_tt - (c**2) * laplacian_u
    pde_loss = jnp.mean(pde_residual**2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        u = derivs["u"]
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = pde_weight * pde_loss + zero_shot_bias_weight * zero_shot_bias_loss

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "bc_loss": 0.0,
            "ic_loss": 0.0,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": 0.0,
            "unweighted_ic_loss": 0.0,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def flow_mixing_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    omega: float = 1.0,
    v_t_max: float = 0.385,
    bc_data: Optional[Dict[str, jnp.ndarray]] = None,
    ic_data: Optional[Dict[str, jnp.ndarray]] = None,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Loss for 2+1D flow mixing: u_t + a(x,y) u_x + b(x,y) u_y = 0."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_z"))
    u_x = derivs["u_x"]
    u_y = derivs["u_y"]

    x, y = X[:, 0], X[:, 1]
    r = jnp.maximum(jnp.sqrt(x**2 + y**2), 1e-8)
    sech_r = 1.0 / jnp.cosh(r)
    v_t = sech_r**2 * jnp.tanh(r)
    a = -(v_t / v_t_max) * y / r
    b = (v_t / v_t_max) * x / r

    pde_residual = u_t + a * u_x + b * u_y
    total = pde_weight * jnp.mean(pde_residual**2)

    if bc_data is not None:
        bc_points = bc_data["points"]
        bc_values = bc_data["values"]
        bc_pred = compute_derivatives(bc_points, params, model, grid_shape)["u"]
        total = total + bc_weight * jnp.mean((bc_pred - bc_values) ** 2)

    if ic_data is not None:
        ic_points = ic_data["points"]
        ic_values = ic_data["values"]
        ic_pred = compute_derivatives(ic_points, params, model, grid_shape)["u"]
        total = total + ic_weight * jnp.mean((ic_pred - ic_values) ** 2)

    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        total = total + zero_shot_bias_weight * jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    return total


def cdr_hyper_lr_pinn_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    beta: float = 0.01,
    nu: float = 1.0,
    rho: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    X_ic: Optional[jnp.ndarray] = None,
    u_ic: Optional[jnp.ndarray] = None,
    X_bc: Optional[jnp.ndarray] = None,
    u_bc: Optional[jnp.ndarray] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """CDR (HLRP variant): ∂u/∂t + β ∂u/∂x - ν ∂²u/∂x² - ρ u(1-u) = 0, point-cloud IC/BC."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_t = derivs.get("u_t", derivs.get("u_y"))
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]

    pde_residual = u_t + beta * u_x - nu * u_xx - rho * u * (1 - u)
    pde_loss = jnp.mean(pde_residual**2)

    ic_loss = 0.0
    if X_ic is not None and u_ic is not None and X_ic.shape[0] > 0:
        u_ic_pred = compute_derivatives(X_ic, params, model, grid_shape)["u"]
        ic_loss = jnp.mean((u_ic_pred - u_ic) ** 2)

    bc_loss = 0.0
    if X_bc is not None and u_bc is not None and X_bc.shape[0] > 0:
        u_bc_pred = compute_derivatives(X_bc, params, model, grid_shape)["u"]
        bc_loss = jnp.mean((u_bc_pred - u_bc) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + ic_weight * ic_loss
        + bc_weight * bc_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "ic_loss": ic_weight * ic_loss,
            "bc_loss": bc_weight * bc_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def helmholtz_hyper_lr_pinn_loss(
    X: jnp.ndarray,
    params: dict,
    model: Any,
    k: float = 1.0,
    a1: float = 1.0,
    a2: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    X_ic: Optional[jnp.ndarray] = None,
    u_ic: Optional[jnp.ndarray] = None,
    X_bc: Optional[jnp.ndarray] = None,
    u_bc: Optional[jnp.ndarray] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    zero_shot_bias_weight: float = 0.0,
    ground_truth_solution: Optional[jnp.ndarray] = None,
    return_components: bool = False,
) -> jnp.ndarray:
    """Helmholtz 2D (HLRP variant): Δu + k²u = q, point-cloud IC/BC."""
    derivs = compute_derivatives(X, params, model, grid_shape)
    u = derivs["u"]
    u_xx = derivs["u_xx"]
    u_yy = derivs.get("u_yy", derivs.get("u_tt"))

    x, y = X[:, 0], X[:, 1]
    q_coeff = -(a1 * jnp.pi) ** 2 - (a2 * jnp.pi) ** 2 + k**2
    q = q_coeff * jnp.sin(a1 * jnp.pi * x) * jnp.sin(a2 * jnp.pi * y)

    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = jnp.mean(pde_residual**2)

    ic_loss = 0.0
    if X_ic is not None and u_ic is not None and X_ic.shape[0] > 0:
        u_ic_pred = compute_derivatives(X_ic, params, model, grid_shape)["u"]
        ic_loss = jnp.mean((u_ic_pred - u_ic) ** 2)

    bc_loss = 0.0
    if X_bc is not None and u_bc is not None and X_bc.shape[0] > 0:
        u_bc_pred = compute_derivatives(X_bc, params, model, grid_shape)["u"]
        bc_loss = jnp.mean((u_bc_pred - u_bc) ** 2)

    zero_shot_bias_loss = 0.0
    if zero_shot_bias_weight > 0.0 and ground_truth_solution is not None:
        zero_shot_bias_loss = jnp.mean((u - ground_truth_solution.ravel()) ** 2)

    total = (
        pde_weight * pde_loss
        + ic_weight * ic_loss
        + bc_weight * bc_loss
        + zero_shot_bias_weight * zero_shot_bias_loss
    )

    if return_components:
        return {
            "total_loss": total,
            "pde_loss": pde_weight * pde_loss,
            "ic_loss": ic_weight * ic_loss,
            "bc_loss": bc_weight * bc_loss,
            "zero_shot_bias_loss": zero_shot_bias_weight * zero_shot_bias_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_ic_loss": ic_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_zero_shot_bias": zero_shot_bias_loss,
        }
    return total


def get_pinn_loss(dataset_name: str):
    """Return the PINN loss function for a dataset name.

    Supports: allen_cahn, burgers, cdr, helmholtz2d, flow_mixing, convection,
    ks, wave2d (and partial matches, e.g. convection_64x64).
    """
    loss_functions = {
        "allen_cahn": allen_cahn_loss,
        "burgers": burgers_loss,
        "cdr": cdr_loss,
        "helmholtz2d": helmholtz_2d_loss,
        "helmholtz2D": helmholtz_2d_loss,
        "flow_mixing": flow_mixing_loss,
        "convection": convection_loss,
        "ks": ks_loss,
        "kuramoto_sivashinsky": ks_loss,
        "wave2d": wave2d_loss,
        "wave2d1": wave2d_loss,
    }

    if dataset_name in loss_functions:
        return loss_functions[dataset_name]

    dataset_lower = dataset_name.lower()
    for key, loss_fn in loss_functions.items():
        if key in dataset_lower:
            return loss_fn

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Available: {list(loss_functions.keys())}"
    )
