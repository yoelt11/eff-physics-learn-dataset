import jax
# Use 32-bit precision for better performance
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Dict, Callable, Optional, Any, Tuple
from jax import grad


def compute_derivatives(X: jnp.ndarray, params: Any, model: Any) -> Dict[str, jnp.ndarray]:
    """Compute derivatives using PINNWrapper for unified interface.

    This function wraps the model and uses PINNWrapper to compute derivatives,
    providing a consistent interface regardless of model type.

    Args:
        X: Evaluation points (N, D)
        params: Model parameters
        model: Model instance

    Returns:
        Dictionary containing derivatives
    """
    from src.models.model_wrapper import PINNWrapper

    # Check if already wrapped
    if isinstance(model, PINNWrapper):
        return model.fn_derivatives(X)

    # Wrap and compute derivatives
    wrapper = PINNWrapper.auto_detect(model, params)
    return wrapper.fn_derivatives(X)


def cdr_loss(X: jnp.ndarray, params: dict, model: Any,
            D: float = 0.01,
            v: float = 1.0,
            R: float = 1.0,
            bc_weight: float = 1.0,
            ic_weight: float = 1.0,
            pde_weight: float = 1.0,
            grid_shape: Optional[Tuple[int, int]] = None,
            return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the Convection-Diffusion-Reaction (CDR) equation.

    PDE: ∂u/∂t + v ∂u/∂x = D ∂²u/∂x² - Ru

    Grid Layout: X is (N, 2) with columns [x, t]
    Grid Shape: (n_x, n_t) where n_x is spatial points, n_t is temporal points
    Domain: x ∈ [0, L], t ∈ [0, T]

    Args:
        X: Evaluation points (N, 2) where columns are [x, t]
        params: Model parameters
        model: Neural network model
        D: Diffusion coefficient
        v: Convection velocity
        R: Reaction rate
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        grid_shape: Optional tuple (n_x, n_t) specifying the grid dimensions
        return_components: If True, return dict with individual loss components

    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives
    derivs = compute_derivatives(X, params, model)

    u = derivs["u"]
    u_t = derivs["u_t"] if "u_t" in derivs else derivs["u_y"]
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]

    # CDR PDE residual: ∂u/∂t + v ∂u/∂x - D ∂²u/∂x² + Ru = 0
    pde_residual = u_t + v * u_x - D * u_xx + R * u
    pde_loss = jnp.mean(pde_residual**2)

    # Use provided grid_shape or compute from X if not provided
    if grid_shape is None:
        n_points = X.shape[0]
        n_sqrt = jnp.astype(jnp.sqrt(n_points), jnp.int32)
        grid_shape = (n_sqrt, n_sqrt)

    n_x, n_t = grid_shape

    # Reshape solution to grid
    u_reshaped = u.reshape(n_x, n_t)
    x_coords = X[:, 0].reshape(n_x, n_t)
    t_coords = X[:, 1].reshape(n_x, n_t)

    # Periodic boundary conditions: u(0, t) = u(L, t)
    u_left_bc_pred = u_reshaped[:, 0]
    u_right_bc_pred = u_reshaped[:, -1]
    bc_loss = jnp.mean((u_left_bc_pred - u_right_bc_pred)**2)

    # Initial condition: u(x, 0) = sin(2πx/L)
    def initial_condition(x_vals):
        L = jnp.max(x_vals) - jnp.min(x_vals)
        return jnp.sin(2 * jnp.pi * x_vals / L)

    u_ic_pred = u_reshaped[0, :]
    x_ic = x_coords[0, :]
    u_ic_gt = initial_condition(x_ic)
    ic_loss = jnp.mean((u_ic_pred - u_ic_gt)**2)

    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss

    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss

    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }

    return total_loss


# Frozen
def allen_cahn_loss(X: jnp.ndarray, params: dict, model: Any,
                   eps: float = 0.01,
                   lam: float = 1.0,
                   bc_weight: float = 1.0,
                   ic_weight: float = 100.0,
                   pde_weight: float = 1.0,
                   grid_shape: Optional[Tuple[int, int]] = None,
                   return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the Allen-Cahn equation.
    
    PDE: u_t - eps^2 * u_xx + lam * (u^3 - u) = 0
    
    Grid Layout: X is (N, 2) with columns [x, t]
    Grid Shape: (n_x, n_t) where n_x is spatial points, n_t is temporal points
    
    Args:
        X: Evaluation points (N, 2) where columns are [x, t]
        params: Model parameters
        model: Neural network model
        eps: Interface parameter (epsilon)
        lam: Diffusion parameter (lambda)  
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        grid_shape: Optional tuple (n_x, n_t) specifying the grid dimensions
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives
    derivs = compute_derivatives(X, params, model)
    
    u = derivs["u"]
    u_t = derivs["u_t"] if "u_t" in derivs else derivs["u_y"]
    u_xx = derivs["u_xx"]
    
    # Allen-Cahn PDE residual loss: ∂u/∂t = ε² ∂²u/∂x² - λ(u³ - u)
    # Rearranged as: ∂u/∂t - ε² ∂²u/∂x² + λ(u³ - u) = 0
    pde_residual = u_t - eps**2 * u_xx + lam * (u**3 - u)
    pde_loss = jnp.mean(pde_residual**2)
    
    # Use provided grid_shape or compute from X if not provided
    # For JAX compatibility, we need to use static shapes
    # In the training script, we're passing grid_shape explicitly, so this should not be an issue
    if grid_shape is None:
        # For JIT compatibility, we can't compute this dynamically
        # We'll use a fixed approach assuming square grids
        # This is a workaround for JIT compilation
        n_points = X.shape[0]
        # We'll assume a square grid for simplicity
        n_sqrt = jnp.astype(jnp.sqrt(n_points), jnp.int32)
        grid_shape = (n_sqrt, n_sqrt)
    
    n_x, n_t = grid_shape
    
    # Reshape solution to grid: u_reshaped[i, j] = u(x_i, t_j)
    u_reshaped = u.reshape(n_x, n_t)
    
    # Reshape coordinates to match solution grid
    x_coords = X[:, 0].reshape(n_x, n_t)  # x coordinates
    t_coords = X[:, 1].reshape(n_x, n_t)  # t coordinates

    # Boundary conditions: u = -1 at spatial boundaries (x = x_min and x = x_max)
    # NOTE: With meshgrid(x, t), u_reshaped[i, j] corresponds to (x[j], t[i])
    def boundary_condition(x_vals):
        return jnp.ones_like(x_vals) * -1.0

    # Extract spatial boundary conditions (fixed x, varying t)
    # Left spatial boundary: x = x_min (first column - minimum x index)
    u_left_bc_pred = u_reshaped[:, 0]      # u(x_min, all t)
    x_left_bc = x_coords[:, 0]             # x_min values (should be constant)
    u_left_bc_gt = boundary_condition(x_left_bc)

    # Right spatial boundary: x = x_max (last column - maximum x index)
    u_right_bc_pred = u_reshaped[:, -1]    # u(x_max, all t)
    x_right_bc = x_coords[:, -1]           # x_max values (should be constant)
    u_right_bc_gt = boundary_condition(x_right_bc)

    # Combine boundary losses
    bc_loss = jnp.mean((u_left_bc_pred - u_left_bc_gt)**2) + \
              jnp.mean((u_right_bc_pred - u_right_bc_gt)**2)

    # Initial condition: u(x, t=0) = x^2 * cos(π*x)
    def initial_condition(x_vals):
        return x_vals**2 * jnp.cos(jnp.pi * x_vals)

    # Extract initial condition (fixed t = t_min, varying x)
    # Initial condition: t = t_min (first row - minimum t index)
    u_ic_pred = u_reshaped[0, :]           # u(all x, t_min)
    x_ic = x_coords[0, :]                  # x values at t_min
    u_ic_gt = initial_condition(x_ic)

    ic_loss = jnp.mean((u_ic_pred - u_ic_gt)**2)
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss

    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss

# Frozen
def burgers_loss(X: jnp.ndarray, params: dict, model: Any,
                 nu: float = 0.01,
                 A: float = 1.0,
                 k: float = 1.0,
                 bc_weight: float = 1.0,
                 ic_weight: float = 1.0,
                 pde_weight: float = 1.0,
                 grid_shape: Optional[Tuple[int, int]] = None,
                 return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the Burgers equation with built-in boundary and initial conditions.
    
    PDE: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    
    Grid Layout: X is (N, 2) with columns [x, t]
    Grid Shape: (n_x, n_t) where n_x is spatial points, n_t is temporal points
    Domain: x ∈ [-1, 1], t ∈ [0, 1]
    
    Initial Condition: u(x, 0) = A sin(kπx)
    Boundary Conditions: Periodic at x = -1 and x = 1
    
    Args:
        X: Evaluation points (N, 2) where columns are [x, t]
        params: Model parameters
        model: Neural network model
        nu: Viscosity parameter
        A: Amplitude parameter for initial condition
        k: Wavenumber parameter for initial condition
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        grid_shape: Optional tuple (n_x, n_t) specifying the grid dimensions
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives
    derivs = compute_derivatives(X, params, model)
    
    u = derivs["u"]
    u_t = derivs["u_t"] if "u_t" in derivs else derivs["u_y"]
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]
    
    # Burgers PDE residual: ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0
    pde_residual = u_t + u * u_x - nu * u_xx
    pde_loss = jnp.mean(pde_residual**2)
    
    # Use provided grid_shape or compute from X if not provided
    if grid_shape is None:
        # For JIT compatibility, we can't compute this dynamically
        n_points = X.shape[0]
        # Assume a square grid for simplicity
        n_sqrt = jnp.astype(jnp.sqrt(n_points), jnp.int32)
        grid_shape = (n_sqrt, n_sqrt)
    
    n_x, n_t = grid_shape
    
    # Reshape solution to grid: u_reshaped[i, j] = u(x_i, t_j)
    u_reshaped = u.reshape(n_x, n_t)
    
    # Reshape coordinates to match solution grid
    x_coords = X[:, 0].reshape(n_x, n_t)  # x coordinates
    t_coords = X[:, 1].reshape(n_x, n_t)  # t coordinates

    # Periodic boundary conditions: u(x_min, t) = u(x_max, t) for all t
    # NOTE: With meshgrid(x, t), u_reshaped[i, j] corresponds to (x[j], t[i])
    # Left boundary: x = x_min = -1 (first column - minimum x index)
    u_left_bc_pred = u_reshaped[:, 0]      # u(x_min, all t)

    # Right boundary: x = x_max = 1 (last column - maximum x index)
    u_right_bc_pred = u_reshaped[:, -1]    # u(x_max, all t)

    # Periodic boundary condition: u(-1, t) = u(1, t)
    bc_loss = jnp.mean((u_left_bc_pred - u_right_bc_pred)**2)

    # Initial condition: u(x, t=0) = A sin(kπx)
    def initial_condition(x_vals):
        return A * jnp.sin(k * jnp.pi * x_vals)

    # Extract initial condition (fixed t = t_min = 0, varying x)
    # Initial condition: t = t_min (first row - minimum t index)
    u_ic_pred = u_reshaped[0, :]           # u(all x, t_min)
    x_ic = x_coords[0, :]                  # x values at t_min
    u_ic_gt = initial_condition(x_ic)

    ic_loss = jnp.mean((u_ic_pred - u_ic_gt)**2)
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss

    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss

# Frozen
def convection_loss(X: jnp.ndarray, params: dict, model: Any,
                   beta: float = 1.0,
                   bc_weight: float = 1.0,
                   ic_weight: float = 1.0,
                   pde_weight: float = 1.0,
                   grid_shape: Optional[Tuple[int, int]] = None,
                   return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the pure Convection (advection) equation with grid-based boundary conditions.
    
    PDE: ∂u/∂t + β ∂u/∂x = 0 (pure advection equation)
    
    Grid Layout: X is (N, 2) with columns [x, t]
    Grid Shape: (n_x, n_t) where n_x is spatial points, n_t is temporal points
    Domain: x ∈ [0, L], t ∈ [0, T]
    
    Initial Condition: u(x, 0) = 1 + sin(2πx/L)
    Boundary Conditions: Periodic - u(x=0,t) = u(x=L,t)
    
    Args:
        X: Evaluation points (N, 2) where columns are [x, t]
        params: Model parameters
        model: Neural network model
        beta: Convection parameter (advection speed)
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        grid_shape: Optional tuple (n_x, n_t) specifying the grid dimensions
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives
    derivs = compute_derivatives(X, params, model)
    
    u = derivs["u"]
    u_t = derivs["u_t"] if "u_t" in derivs else derivs["u_y"]
    u_x = derivs["u_x"]
    
    # Pure convection PDE residual: ∂u/∂t + β ∂u/∂x = 0
    pde_residual = u_t + beta * u_x
    pde_loss = jnp.mean(pde_residual**2)
    
    # Use provided grid_shape or compute from X if not provided
    if grid_shape is None:
        # For JIT compatibility, we can't compute this dynamically
        n_points = X.shape[0]
        # Assume a square grid for simplicity
        n_sqrt = jnp.astype(jnp.sqrt(n_points), jnp.int32)
        grid_shape = (n_sqrt, n_sqrt)
    
    n_x, n_t = grid_shape
    
    # Reshape solution to grid: u_reshaped[i, j] = u(x_i, t_j)
    u_reshaped = u.reshape(n_x, n_t)
    
    # Reshape coordinates to match solution grid
    x_coords = X[:, 0].reshape(n_x, n_t)  # x coordinates
    t_coords = X[:, 1].reshape(n_x, n_t)  # t coordinates

    # Periodic boundary conditions: u(x=0,t) = u(x=L,t) for all t
    # NOTE: With meshgrid(x, t), u_reshaped[i, j] corresponds to (x[j], t[i])
    # Left boundary: x = x_min = 0 (first column - minimum x index)
    u_left_bc_pred = u_reshaped[:, 0]      # u(x_min, all t)

    # Right boundary: x = x_max = L (last column - maximum x index)
    u_right_bc_pred = u_reshaped[:, -1]    # u(x_max, all t)

    # Periodic boundary condition: u(0, t) = u(L, t)
    bc_loss = jnp.mean((u_left_bc_pred - u_right_bc_pred)**2)

    # Initial condition: u(x, t=0) = 1 + sin(2πx/L) with proper scaling
    def initial_condition(x_vals):
        # Determine domain length L from coordinate bounds
        x_min_coord = jnp.min(x_vals)
        x_max_coord = jnp.max(x_vals)
        L = x_max_coord - x_min_coord  # Should be 6.0 for convection

        # Properly scaled IC: u(x,0) = 1 + sin(2πx/L)
        return 1.0 + jnp.sin(2 * jnp.pi * x_vals / L)

    # Extract initial condition (fixed t = t_min, varying x)
    # Initial condition: t = t_min (first row - minimum t index)
    u_ic_pred = u_reshaped[0, :]           # u(all x, t_min)
    x_ic = x_coords[0, :]                  # x values at t_min
    u_ic_gt = initial_condition(x_ic)

    ic_loss = jnp.mean((u_ic_pred - u_ic_gt)**2)
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss
    
    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss

# Frozen
def helmholtz_2d_loss(X: jnp.ndarray, params: dict, model: Any,
                      k: float = 1.0,
                      a1: float = 1.0,
                      a2: float = 1.0,
                      bc_weight: float = 1.0,
                      pde_weight: float = 1.0,
                      grid_shape: Optional[Tuple[int, int]] = None,
                      return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the 2D Helmholtz equation with grid-based boundary conditions.

    PDE: Δu + k²u = q(x,y)
    where q(x,y) = (-(a₁π)² - (a₂π)² + k²) sin(a₁πx) sin(a₂πy)
    Exact solution: u(x,y) = sin(a₁πx) sin(a₂πy)
    Domain: [-1,1] × [-1,1] with Dirichlet BC: u = analytical values on boundaries
    
    Grid Layout: X is (N, 2) with columns [x, y]  
    Grid Shape: (n_x, n_y) where n_x is x-direction points, n_y is y-direction points
    
    Args:
        X: Evaluation points (N, 2) where columns are [x, y]
        params: Model parameters
        model: Neural network model
        k: Wavenumber parameter
        a1, a2: Frequency parameters for manufactured solution
        bc_weight: Weight for boundary condition loss
        pde_weight: Weight for PDE residual loss
        grid_shape: Optional tuple (n_x, n_y) specifying the grid dimensions
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives
    derivs = compute_derivatives(X, params, model)
    
    u = derivs["u"]
    u_xx = derivs["u_xx"]
    u_yy = derivs["u_yy"] if "u_yy" in derivs else derivs["u_tt"]  # u_yy might be stored as u_tt in some implementations
    
    # Extract coordinates
    x, y = X[:, 0], X[:, 1]

    # Pre-compute constant coefficient for manufactured source term (optimization)
    q_coeff = -(a1 * jnp.pi)**2 - (a2 * jnp.pi)**2 + k**2

    # Manufactured source term: q(x,y) = q_coeff * sin(a₁πx) sin(a₂πy)
    q = q_coeff * jnp.sin(a1 * jnp.pi * x) * jnp.sin(a2 * jnp.pi * y)

    # Helmholtz PDE residual: Δu + k²u - q = 0
    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = jnp.mean(pde_residual**2)
    
    # Use provided grid_shape or compute from X if not provided
    if grid_shape is None:
        # For JIT compatibility, we can't compute this dynamically
        n_points = X.shape[0]
        # Assume a square grid for simplicity
        n_sqrt = jnp.astype(jnp.sqrt(n_points), jnp.int32)
        grid_shape = (n_sqrt, n_sqrt)
    
    n_x, n_y = grid_shape

    # CRITICAL FIX: X was created by flatten() on (n_y, n_x) grids, so we must reshape to (n_y, n_x)
    # even though grid_shape is labeled as (n_x, n_y). This is because train_step.py swaps the
    # dimensions when passing: grid_shape = (X_grid.shape[1], X_grid.shape[0])
    # So n_x = X_grid.shape[1] (num columns = x-points) and n_y = X_grid.shape[0] (num rows = y-points)
    # When we flatten X_grid with shape (n_y, n_x), we get (n_y*n_x,) in row-major order.
    # To reshape back, we need reshape(n_y, n_x), which is reshape(grid_shape[1], grid_shape[0])
    u_reshaped = u.reshape(n_y, n_x)  # Swap dimensions to match flattening order

    # Reshape coordinates to match solution grid
    x_coords = X[:, 0].reshape(n_y, n_x)  # x coordinates
    y_coords = X[:, 1].reshape(n_y, n_x)  # y coordinates

    # Dirichlet boundary conditions: u = analytical values on all boundaries of [-1,1] × [-1,1]
    # Exact solution: u(x,y) = sin(a₁πx) sin(a₂πy)
    # Grid structure: u_reshaped[i, j] where i is y-index (row), j is x-index (column)
    # so u_reshaped[i, j] = u(x[j], y[i])

    # Bottom boundary: y = y_min (first row - minimum y index)
    u_bottom_bc_pred = u_reshaped[0, :]    # u(all x, y_min)
    x_bottom_bc = x_coords[0, :]           # x values at bottom boundary
    y_bottom_bc = y_coords[0, :]           # y values at bottom boundary (should all be y_min)
    u_bottom_bc_gt = jnp.sin(a1 * jnp.pi * x_bottom_bc) * jnp.sin(a2 * jnp.pi * y_bottom_bc)

    # Top boundary: y = y_max (last row - maximum y index)
    u_top_bc_pred = u_reshaped[-1, :]      # u(all x, y_max)
    x_top_bc = x_coords[-1, :]             # x values at top boundary
    y_top_bc = y_coords[-1, :]             # y values at top boundary (should all be y_max)
    u_top_bc_gt = jnp.sin(a1 * jnp.pi * x_top_bc) * jnp.sin(a2 * jnp.pi * y_top_bc)

    # Left boundary: x = x_min (first column - minimum x index)
    u_left_bc_pred = u_reshaped[:, 0]      # u(x_min, all y)
    x_left_bc = x_coords[:, 0]             # x values at left boundary (should all be x_min)
    y_left_bc = y_coords[:, 0]             # y values at left boundary
    u_left_bc_gt = jnp.sin(a1 * jnp.pi * x_left_bc) * jnp.sin(a2 * jnp.pi * y_left_bc)

    # Right boundary: x = x_max (last column - maximum x index)
    u_right_bc_pred = u_reshaped[:, -1]    # u(x_max, all y)
    x_right_bc = x_coords[:, -1]           # x values at right boundary (should all be x_max)
    y_right_bc = y_coords[:, -1]           # y values at right boundary
    u_right_bc_gt = jnp.sin(a1 * jnp.pi * x_right_bc) * jnp.sin(a2 * jnp.pi * y_right_bc)

    # Combine boundary losses
    bc_loss = jnp.mean((u_bottom_bc_pred - u_bottom_bc_gt)**2) + \
              jnp.mean((u_top_bc_pred - u_top_bc_gt)**2) + \
              jnp.mean((u_left_bc_pred - u_left_bc_gt)**2) + \
              jnp.mean((u_right_bc_pred - u_right_bc_gt)**2)
    
    # For Helmholtz, there's no initial condition, so ic_loss is 0
    ic_loss = 0.0
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = 0.0  # No IC term for Helmholtz
    
    total_loss = weighted_pde_loss + weighted_bc_loss

    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss

# Frozen
def flow_mixing_loss(X: jnp.ndarray, params: dict, model: Any,
                     omega: float = 1.0,
                     v_t_max: float = 0.385,
                     bc_data: Optional[Dict[str, jnp.ndarray]] = None,
                     ic_data: Optional[Dict[str, jnp.ndarray]] = None,
                     bc_weight: float = 1.0,
                     ic_weight: float = 1.0,
                     pde_weight: float = 1.0) -> jnp.ndarray:
    """Compute the loss for the 2+1D flow mixing problem.
    
    PDE: u_t + a(x,y) u_x + b(x,y) u_y = 0
    where a(x,y) = -(v_t/v_t_max) * y/r, b(x,y) = (v_t/v_t_max) * x/r
    v_t = sech²(r) tanh(r), r = √(x² + y²)
    
    Args:
        X: Evaluation points (N, 3) where columns are [x, y, t]
        params: Model parameters
        model: Neural network model
        omega: Angular frequency parameter
        v_t_max: Maximum tangential velocity
        bc_data: Boundary condition data
        ic_data: Initial condition data
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        
    Returns:
        Total loss value
    """
    # Compute derivatives
    derivs = compute_derivatives(X, params, model)
    
    u = derivs["u"]
    u_t = derivs["u_t"] if "u_t" in derivs else derivs["u_z"]  # t is 3rd dimension
    u_x = derivs["u_x"]
    u_y = derivs["u_y"]
    
    # Extract coordinates
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    
    # Compute velocity field
    r = jnp.sqrt(x**2 + y**2)
    r = jnp.maximum(r, 1e-8)  # Avoid division by zero
    
    # v_t = sech²(r) tanh(r)
    sech_r = 1.0 / jnp.cosh(r)
    v_t = sech_r**2 * jnp.tanh(r)
    
    # Velocity components
    a = -(v_t / v_t_max) * y / r
    b = (v_t / v_t_max) * x / r
    
    # Flow mixing PDE residual: u_t + a(x,y) u_x + b(x,y) u_y = 0
    pde_residual = u_t + a * u_x + b * u_y
    pde_loss = jnp.mean(pde_residual**2)
    
    total_loss = pde_weight * pde_loss
    
    # Boundary condition loss
    if bc_data is not None:
        bc_points = bc_data["points"]
        bc_values = bc_data["values"]
        bc_pred = compute_derivatives(bc_points, params, model)["u"]
        bc_loss = jnp.mean((bc_pred - bc_values)**2)
        total_loss += bc_weight * bc_loss
    
    # Initial condition loss  
    if ic_data is not None:
        ic_points = ic_data["points"]
        ic_values = ic_data["values"]
        ic_pred = compute_derivatives(ic_points, params, model)["u"]
        ic_loss = jnp.mean((ic_pred - ic_values)**2)
        total_loss += ic_weight * ic_loss
    
    return total_loss


def get_pinn_loss(dataset_name: str):
    """Get the appropriate PINN loss function for a given dataset.

    Args:
        dataset_name: Name of the dataset ('allen_cahn', 'burgers', 'cdr', 'helmholtz2d', 'flow_mixing', 'convection')

    Returns:
        Loss function for the specified dataset
    """
    loss_functions = {
        'allen_cahn': allen_cahn_loss,
        'burgers': burgers_loss,
        'cdr': cdr_loss,
        'helmholtz2d': helmholtz_2d_loss,
        'helmholtz2D': helmholtz_2d_loss,  # Alternative naming
        'flow_mixing': flow_mixing_loss,
        'convection': convection_loss,
    }

    # Handle exact matches first
    if dataset_name in loss_functions:
        return loss_functions[dataset_name]

    # Handle partial matches for dataset names with suffixes (e.g., "convection_64x64")
    dataset_lower = dataset_name.lower()
    for key, loss_fn in loss_functions.items():
        if key in dataset_lower:
            return loss_fn

    # If no match found, raise error
    raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loss_functions.keys())}")

def cdr_hyper_lr_pinn_loss(X: jnp.ndarray,
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
                           return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the CDR equation (HLRP dataset variant).

    This loss function is designed for the HLRP dataset structure where initial
    and boundary conditions are provided as point cloud data.

    PDE: ∂u/∂t + β ∂u/∂x - ν ∂²u/∂x² - ρ u(1-u) = 0

    This is a convection-diffusion-reaction equation where:
    - β: convection coefficient
    - ν: diffusion coefficient
    - ρ: reaction coefficient (Fisher-KPP type reaction)

    Args:
        X: Collocation points for PDE residual (N_col, 2) where columns are [x, t]
        params: Model parameters
        model: Neural network model
        beta: Convection coefficient
        nu: Diffusion coefficient
        rho: Reaction coefficient
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        X_ic: Initial condition points (N_ic, 2), optional
        u_ic: Initial condition values (N_ic,), optional
        X_bc: Boundary condition points (N_bc, 2), optional
        u_bc: Boundary condition values (N_bc,), optional
        return_components: If True, return dict with individual loss components

    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives at collocation points
    derivs = compute_derivatives(X, params, model)

    u = derivs["u"]
    u_t = derivs["u_t"] if "u_t" in derivs else derivs["u_y"]
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]

    # CDR PDE residual: ∂u/∂t + β ∂u/∂x - ν ∂²u/∂x² - ρ u(1-u) = 0
    pde_residual = u_t + beta * u_x - nu * u_xx - rho * u * (1 - u)
    pde_loss = jnp.mean(pde_residual**2)

    # Initial condition loss (MSE at IC points)
    if X_ic is not None and u_ic is not None and X_ic.shape[0] > 0:
        # Evaluate model at initial condition points
        derivs_ic = compute_derivatives(X_ic, params, model)
        u_ic_pred = derivs_ic["u"]
        ic_loss = jnp.mean((u_ic_pred - u_ic)**2)
    else:
        ic_loss = 0.0

    # Boundary condition loss (MSE at BC points)
    if X_bc is not None and u_bc is not None and X_bc.shape[0] > 0:
        # Evaluate model at boundary condition points
        derivs_bc = compute_derivatives(X_bc, params, model)
        u_bc_pred = derivs_bc["u"]
        bc_loss = jnp.mean((u_bc_pred - u_bc)**2)
    else:
        bc_loss = 0.0

    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_ic_loss = ic_weight * ic_loss
    weighted_bc_loss = bc_weight * bc_loss

    total_loss = weighted_pde_loss + weighted_ic_loss + weighted_bc_loss

    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'ic_loss': weighted_ic_loss,
            'bc_loss': weighted_bc_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_ic_loss': ic_loss,
            'unweighted_bc_loss': bc_loss
        }

    return total_loss

def helmholtz_hyper_lr_pinn_loss(X: jnp.ndarray, params: dict, model: Any,
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
                                return_components: bool = False) -> jnp.ndarray:
    """Compute the loss for the 2D Helmholtz equation (HLRP dataset variant).

    This loss function is designed for the HLRP dataset structure where initial
    and boundary conditions are provided as point cloud data.

    PDE: Δu + k²u = q(x,y)
    where q(x,y) = (-(a₁π)² - (a₂π)² + k²) sin(a₁πx) sin(a₂πy)
    Exact solution: u(x,y) = sin(a₁πx) sin(a₂πy)

    Args:
        X: Collocation points for PDE residual (N_col, 2) where columns are [x, y]
        params: Model parameters
        model: Neural network model
        k: Wavenumber parameter
        a1, a2: Frequency parameters for manufactured solution
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        X_ic: Initial condition points (N_ic, 2), optional
        u_ic: Initial condition values (N_ic,), optional
        X_bc: Boundary condition points (N_bc, 2), optional
        u_bc: Boundary condition values (N_bc,), optional
        return_components: If True, return dict with individual loss components

    Returns:
        Total loss value or dict with loss components if return_components=True
    """
    # Compute derivatives at collocation points
    derivs = compute_derivatives(X, params, model)

    u = derivs["u"]
    u_xx = derivs["u_xx"]
    u_yy = derivs["u_yy"] if "u_yy" in derivs else derivs["u_tt"]  # u_yy might be stored as u_tt in some implementations

    # Extract coordinates
    x, y = X[:, 0], X[:, 1]

    # Pre-compute constant coefficient for manufactured source term (optimization)
    q_coeff = -(a1 * jnp.pi)**2 - (a2 * jnp.pi)**2 + k**2

    # Manufactured source term: q(x,y) = q_coeff * sin(a₁πx) sin(a₂πy)
    q = q_coeff * jnp.sin(a1 * jnp.pi * x) * jnp.sin(a2 * jnp.pi * y)

    # Helmholtz PDE residual: Δu + k²u - q = 0
    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = jnp.mean(pde_residual**2)

    # Initial condition loss (MSE at IC points)
    if X_ic is not None and u_ic is not None and X_ic.shape[0] > 0:
        # Evaluate model at initial condition points
        derivs_ic = compute_derivatives(X_ic, params, model)
        u_ic_pred = derivs_ic["u"]
        ic_loss = jnp.mean((u_ic_pred - u_ic)**2)
    else:
        ic_loss = 0.0

    # Boundary condition loss (MSE at BC points)
    if X_bc is not None and u_bc is not None and X_bc.shape[0] > 0:
        # Evaluate model at boundary condition points
        derivs_bc = compute_derivatives(X_bc, params, model)
        u_bc_pred = derivs_bc["u"]
        bc_loss = jnp.mean((u_bc_pred - u_bc)**2)
    else:
        bc_loss = 0.0

    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_ic_loss = ic_weight * ic_loss
    weighted_bc_loss = bc_weight * bc_loss

    total_loss = weighted_pde_loss + weighted_ic_loss + weighted_bc_loss

    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'ic_loss': weighted_ic_loss,
            'bc_loss': weighted_bc_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_ic_loss': ic_loss,
            'unweighted_bc_loss': bc_loss
        }

    return total_loss