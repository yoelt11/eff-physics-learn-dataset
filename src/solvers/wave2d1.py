"""Generate verified 2D wave-equation dataset samples.

PDE:
    u_tt = c(x, y)^2 (u_xx + u_yy)

Initial conditions:
    u(x, y, 0)   = sum of Gaussian sources
    u_t(x, y, 0) = 0

Boundary handling:
    Multiplicative absorbing (sponge) mask near all domain edges.

Notes:
    - Main solution indexing is standard:
        u[k, i, j] = u(t[k], x[i], y[j])
      so solution shape is (n_t, n_x, n_y).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import downsample_solution


def make_grid(
    x_domain: tuple[float, float] = (-3.0, 3.0),
    y_domain: tuple[float, float] = (-3.0, 3.0),
    nx: int = 160,
    ny: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create 2D Cartesian grid and spacings."""
    x = np.linspace(x_domain[0], x_domain[1], nx, dtype=np.float64)
    y = np.linspace(y_domain[0], y_domain[1], ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return x, y, xx, yy, dx, dy


def gaussian_sources(xx: np.ndarray, yy: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Build Gaussian-mixture initial displacement u(x,y,0).

    sources has shape (k, 4), columns [x0, y0, sd, amp].
    """
    p0 = np.zeros_like(xx, dtype=np.float64)
    for x0, y0, sd, amp in np.asarray(sources, dtype=np.float64):
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        p0 += amp * np.exp(-0.5 * r2 / (sd**2))
    return p0


def constant_velocity_model(xx: np.ndarray, yy: np.ndarray, c0: float = 1.0) -> np.ndarray:
    """Constant medium c(x,y)=c0."""
    del yy  # shape is implied by xx; keep signature consistent with gaussian model.
    return c0 * np.ones_like(xx, dtype=np.float64)


def gaussian_velocity_model(
    xx: np.ndarray,
    yy: np.ndarray,
    c0: float = 1.0,
    mixture: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian-mixture velocity field.

    mixture has shape (m, 4), columns [x0, y0, sd, amp]:
        c(x,y) = c0 + sum_i amp_i * exp(-0.5 * ||x-mu_i||^2 / sd_i^2)
    """
    c = c0 * np.ones_like(xx, dtype=np.float64)
    if mixture is None:
        return c

    for x0, y0, sd, amp in np.asarray(mixture, dtype=np.float64):
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        c += amp * np.exp(-0.5 * r2 / (sd**2))

    # Keep velocity positive and numerically safe.
    return np.maximum(c, 0.05)


def make_absorbing_mask(
    nx: int,
    ny: int,
    n_absorb: int = 18,
    strength: float = 0.035,
) -> np.ndarray:
    """Create a simple multiplicative sponge mask."""
    mask = np.ones((nx, ny), dtype=np.float64)

    n_absorb = int(max(0, min(n_absorb, nx // 2, ny // 2)))
    if n_absorb == 0:
        return mask

    for i in range(n_absorb):
        coeff = np.exp(-strength * ((n_absorb - i) / n_absorb) ** 2)
        mask[i, :] *= coeff
        mask[nx - 1 - i, :] *= coeff
        mask[:, i] *= coeff
        mask[:, ny - 1 - i] *= coeff

    return mask


def laplacian_2d(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Second-order central-difference Laplacian with zero-edge stencil."""
    lap = np.zeros_like(u, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
        + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
    )
    return lap


def simulate_wave_equation(
    nx: int = 160,
    ny: int = 160,
    nt: int = 220,
    x_domain: tuple[float, float] = (-3.0, 3.0),
    y_domain: tuple[float, float] = (-3.0, 3.0),
    c0: float = 1.0,
    sources: np.ndarray | None = None,
    mixture: np.ndarray | None = None,
    velocity_type: str = "constant",
    save_every: int = 1,
    absorb_width: int = 18,
    absorb_strength: float = 0.035,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray | float]:
    """Solve the 2D wave equation with explicit second-order time stepping."""
    if nt < 2:
        raise ValueError("nt must be >= 2")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    x, y, xx, yy, dx, dy = make_grid(x_domain=x_domain, y_domain=y_domain, nx=nx, ny=ny)

    if sources is None:
        src = np.array([[0.0, 0.0, 0.2, 1.0]], dtype=np.float64)
    else:
        src = np.asarray(sources, dtype=np.float64)
        if src.ndim != 2 or src.shape[1] != 4:
            raise ValueError("sources must have shape (k, 4): [x0, y0, sd, amp]")

    if velocity_type == "constant":
        c = constant_velocity_model(xx, yy, c0=c0)
    elif velocity_type == "gaussian":
        c = gaussian_velocity_model(xx, yy, c0=c0, mixture=mixture)
    else:
        raise ValueError("velocity_type must be 'constant' or 'gaussian'")

    cmax = float(np.max(c))
    dt = 0.45 / (cmax * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
    t = np.arange(nt, dtype=np.float64) * dt

    u0 = gaussian_sources(xx, yy, src)
    mask = make_absorbing_mask(nx, ny, n_absorb=absorb_width, strength=absorb_strength)

    # Second-order consistent initialization with zero initial velocity.
    lap0 = laplacian_2d(u0, dx, dy)
    u1 = u0 + 0.5 * (dt**2) * (c**2) * lap0

    u0 = u0 * mask
    u1 = u1 * mask

    saved: list[np.ndarray] = [u0.astype(dtype, copy=False)]
    saved_time_indices: list[int] = [0]
    if save_every == 1:
        saved.append(u1.astype(dtype, copy=False))
        saved_time_indices.append(1)

    um1 = u0
    u = u1

    for n in range(1, nt - 1):
        lap = laplacian_2d(u, dx, dy)
        up1 = 2.0 * u - um1 + (dt**2) * (c**2) * lap
        up1 *= mask

        if ((n + 1) % save_every) == 0:
            saved.append(up1.astype(dtype, copy=False))
            saved_time_indices.append(n + 1)

        um1, u = u, up1

    frames = np.stack(saved, axis=0)  # (n_saved, nx, ny)
    t_saved = t[np.asarray(saved_time_indices, dtype=int)]

    return {
        "x": x.astype(dtype),
        "y": y.astype(dtype),
        "t": t_saved.astype(dtype),
        "xx": xx.astype(dtype),
        "yy": yy.astype(dtype),
        "c": c.astype(dtype),
        "u0": u0.astype(dtype),
        "frames": frames.astype(dtype),
        "dt": float(dt),
        "dx": float(dx),
        "dy": float(dy),
    }


def verify_wave2d1_solution(
    solution: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    t_coords: np.ndarray,
    c_field: np.ndarray,
    expected_u0: np.ndarray,
    absorb_width: int = 8,
    pde_threshold: float = 2.0,
    bc_threshold: float = 3e-1,
    ic_threshold: float = 1e-3,
) -> tuple[bool, dict[str, float | bool]]:
    """Verify wave solution against PDE residual, boundary damping, and IC."""
    # Derivatives in (t, x, y) indexing.
    u_t = np.gradient(solution, t_coords, axis=0)
    u_tt = np.gradient(u_t, t_coords, axis=0)
    u_x = np.gradient(solution, x_coords, axis=1)
    u_xx = np.gradient(u_x, x_coords, axis=1)
    u_y = np.gradient(solution, y_coords, axis=2)
    u_yy = np.gradient(u_y, y_coords, axis=2)

    pde_residual = u_tt - (c_field[None, :, :] ** 2) * (u_xx + u_yy)

    # Ignore sponge region for PDE residual checks.
    aw = int(max(1, min(absorb_width, solution.shape[1] // 4, solution.shape[2] // 4)))
    interior = pde_residual[:, aw:-aw, aw:-aw]
    pde_loss = float(np.mean(interior**2))

    # Boundary metric: average boundary energy should remain small under damping.
    b_left = solution[:, 0, :]
    b_right = solution[:, -1, :]
    b_bottom = solution[:, :, 0]
    b_top = solution[:, :, -1]
    bc_loss = float(
        0.25
        * (
            np.mean(b_left**2)
            + np.mean(b_right**2)
            + np.mean(b_bottom**2)
            + np.mean(b_top**2)
        )
    )

    ic_loss = float(np.mean((solution[0] - expected_u0) ** 2))

    pde_ok = pde_loss < pde_threshold
    bc_ok = bc_loss < bc_threshold
    ic_ok = ic_loss < ic_threshold
    is_valid = pde_ok and bc_ok and ic_ok

    metrics: dict[str, float | bool] = {
        "pde_loss": pde_loss,
        "bc_loss": bc_loss,
        "ic_loss": ic_loss,
        "pde_ok": pde_ok,
        "bc_ok": bc_ok,
        "ic_ok": ic_ok,
    }
    return is_valid, metrics


def generate_wave2d1_sample(
    c0: float = 1.0,
    sources: np.ndarray | None = None,
    mixture: np.ndarray | None = None,
    velocity_type: str = "constant",
    x_domain: tuple[float, float] = (-3.0, 3.0),
    y_domain: tuple[float, float] = (-3.0, 3.0),
    t_steps: int = 220,
    solver_nx: int = 160,
    solver_ny: int = 160,
    target_nx: int = 64,
    target_ny: int = 64,
    target_nt: int = 64,
    absorb_width: int = 18,
    absorb_strength: float = 0.035,
    verify: bool = True,
    verify_thresholds: dict[str, float] | None = None,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict[str, float | bool]]:
    """Generate one verified 2D wave sample.

    Returns:
        (u, x, y, t, is_valid, metrics)
        with u shape (target_nt, target_nx, target_ny).
    """
    sim = simulate_wave_equation(
        nx=solver_nx,
        ny=solver_ny,
        nt=t_steps,
        x_domain=x_domain,
        y_domain=y_domain,
        c0=c0,
        sources=sources,
        mixture=mixture,
        velocity_type=velocity_type,
        save_every=1,
        absorb_width=absorb_width,
        absorb_strength=absorb_strength,
        dtype=np.float64,
    )

    u_hires = np.asarray(sim["frames"], dtype=np.float64)  # (nt, nx, ny)
    c_hires = np.asarray(sim["c"], dtype=np.float64)
    u0_hires = np.asarray(sim["u0"], dtype=np.float64)

    # Downsample to target tensor sizes.
    u = downsample_solution(u_hires, (target_nt, target_nx, target_ny))
    c_target = downsample_solution(c_hires, (target_nx, target_ny))
    u0_target = downsample_solution(u0_hires, (target_nx, target_ny))

    x = np.linspace(x_domain[0], x_domain[1], target_nx, dtype=np.float64)
    y = np.linspace(y_domain[0], y_domain[1], target_ny, dtype=np.float64)
    t = np.linspace(
        float(np.asarray(sim["t"])[0]),
        float(np.asarray(sim["t"])[-1]),
        target_nt,
        dtype=np.float64,
    )

    if verify:
        thresholds = verify_thresholds or {}
        is_valid, metrics = verify_wave2d1_solution(
            solution=u,
            x_coords=x,
            y_coords=y,
            t_coords=t,
            c_field=c_target,
            expected_u0=u0_target,
            absorb_width=max(1, int(target_nx * absorb_width / max(1, solver_nx))),
            **thresholds,
        )
    else:
        is_valid = True
        metrics = {}

    return (
        u.astype(dtype),
        x.astype(dtype),
        y.astype(dtype),
        t.astype(dtype),
        is_valid,
        metrics,
    )


if __name__ == "__main__":
    print("Testing 2D Wave solver...")

    u, x, y, t, is_valid, metrics = generate_wave2d1_sample(
        c0=1.0,
        velocity_type="constant",
        solver_nx=140,
        solver_ny=140,
        t_steps=180,
        target_nx=64,
        target_ny=64,
        target_nt=64,
        verify=True,
    )

    print(f"Generated solution shape: {u.shape}")
    print(f"x/y/t shapes: {x.shape} / {y.shape} / {t.shape}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"Valid: {is_valid}")
    print(f"Metrics: {metrics}")
