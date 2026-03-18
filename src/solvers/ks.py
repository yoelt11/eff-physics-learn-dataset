"""Generate verified 1D Kuramoto-Sivashinsky equation dataset samples.

PDE:
    u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx = 0

Boundary conditions:
    Periodic on x in [x_min, x_max]

Numerics:
    Fourier pseudospectral spatial discretization with ETDRK4 time stepping.
"""

from __future__ import annotations

import numpy as np

from verification import verify_solution_ks

from .utils import downsample_solution


def create_periodic_grid(domain: tuple[float, float], n: int) -> np.ndarray:
    """Create a periodic 1D grid without duplicating the endpoint."""
    x_min, x_max = domain
    length = x_max - x_min
    return x_min + length * np.arange(n, dtype=np.float64) / n


def initial_condition_ks(
    x: np.ndarray,
    noise_amplitude: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Reference-style KS initial condition with small random perturbation."""
    rng = np.random.default_rng(seed)
    base = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    noise = noise_amplitude * rng.standard_normal(len(x))
    return base + noise


def ks_fourier_wavenumbers(domain: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Fourier wave numbers k and ik for a periodic domain."""
    x_min, x_max = domain
    length = x_max - x_min
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    return k, 1j * k


def ks_etdrk4_coefficients(Lk: np.ndarray, dt: float, M: int = 64) -> tuple[np.ndarray, ...]:
    """Compute Kassam-Trefethen ETDRK4 coefficients."""
    E = np.exp(dt * Lk)
    E2 = np.exp(dt * Lk / 2.0)

    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, None] + r[None, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(
        np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3),
            axis=1,
        )
    )
    f2 = dt * np.real(
        np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=1)
    )
    f3 = dt * np.real(
        np.mean(
            (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3),
            axis=1,
        )
    )
    return E, E2, Q, f1, f2, f3


def nonlinear_hat(v_hat: np.ndarray, alpha: float, ik: np.ndarray) -> np.ndarray:
    """Nonlinear term in Fourier space: -(alpha/2) * d_x (u^2)."""
    u = np.fft.ifft(v_hat).real
    return -0.5 * alpha * ik * np.fft.fft(u**2)


def solve_ks(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    u0: np.ndarray,
) -> np.ndarray:
    """Solve the 1D KS equation on a periodic grid.

    Returns solution with transposed indexing (n_t, n_x).
    """
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1D grid with at least 2 points")
    if x.ndim != 1 or len(x) < 4:
        raise ValueError("x must be a 1D periodic grid with at least 4 points")

    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt):
        raise ValueError("ETDRK4 implementation requires a uniform temporal grid")

    domain = (float(x[0]), float(x[0] + (x[1] - x[0]) * len(x)))
    k, ik = ks_fourier_wavenumbers(domain, len(x))
    Lk = beta * k**2 - gamma * k**4
    E, E2, Q, f1, f2, f3 = ks_etdrk4_coefficients(Lk, dt)

    nsteps = len(t) - 1
    u_hat = np.fft.fft(np.asarray(u0, dtype=np.float64))

    saved = [np.fft.ifft(u_hat).real.copy()]
    for _ in range(nsteps):
        Nv = nonlinear_hat(u_hat, alpha, ik)
        a = E2 * u_hat + Q * Nv
        Na = nonlinear_hat(a, alpha, ik)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear_hat(b, alpha, ik)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear_hat(c, alpha, ik)
        u_hat = E * u_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        saved.append(np.fft.ifft(u_hat).real.copy())

    return np.asarray(saved, dtype=np.float64)


def _downsample_periodic_solution(
    u_hires: np.ndarray,
    target_nt: int,
    target_nx: int,
) -> np.ndarray:
    """Downsample while preserving periodic closure in x."""
    # Append the first column so the periodic seam is continuous at x_max.
    u_closed = np.concatenate([u_hires, u_hires[:, :1]], axis=1)
    u = downsample_solution(u_closed, (target_nt, target_nx))
    u[:, -1] = u[:, 0]
    return np.asarray(u, dtype=np.float64)


def generate_ks_sample(
    alpha: float,
    beta: float,
    gamma: float,
    x_domain: tuple[float, float] = (0.0, 32.0 * np.pi),
    t_domain: tuple[float, float] = (0.0, 150.0),
    solver_nx: int = 256,
    solver_nt: int = 601,
    target_nx: int = 64,
    target_nt: int = 64,
    noise_amplitude: float = 0.01,
    seed: int = 0,
    verify: bool = True,
    verify_thresholds: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
    """Generate a single verified KS solution sample.

    Returns:
        (solution, x_coords, t_coords, is_valid, metrics)
        where solution has shape (target_nt, target_nx).
    """
    x_hires = create_periodic_grid(x_domain, solver_nx)
    t_hires = np.linspace(t_domain[0], t_domain[1], solver_nt, dtype=np.float64)
    u0_hires = initial_condition_ks(x_hires, noise_amplitude=noise_amplitude, seed=seed)

    u_hires = solve_ks(
        x=x_hires,
        t=t_hires,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        u0=u0_hires,
    )

    u = _downsample_periodic_solution(u_hires, target_nt=target_nt, target_nx=target_nx)
    x = np.linspace(x_domain[0], x_domain[1], target_nx, dtype=np.float64)
    t = np.linspace(t_domain[0], t_domain[1], target_nt, dtype=np.float64)
    u0_target = _downsample_periodic_solution(u_hires[:1], target_nt=1, target_nx=target_nx)[0]

    if verify:
        thresholds = verify_thresholds or {}
        is_valid, metrics = verify_solution_ks(
            solution=u,
            x_coords=x,
            t_coords=t,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            expected_u0=u0_target,
            **thresholds,
        )
    else:
        is_valid = True
        metrics = {}

    return u, x, t, is_valid, metrics


if __name__ == "__main__":
    print("Testing KS solver...")

    u, x, t, is_valid, metrics = generate_ks_sample(
        alpha=1.10,
        beta=1.20,
        gamma=1.20,
        solver_nx=256,
        solver_nt=601,
        target_nx=64,
        target_nt=64,
        verify=True,
    )

    print(f"Generated solution shape: {u.shape}")
    print(f"x/t shapes: {x.shape} / {t.shape}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"Valid: {is_valid}")
    print(f"Metrics: {metrics}")
