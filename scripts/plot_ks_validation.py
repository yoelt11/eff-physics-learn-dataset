#!/usr/bin/env python3
"""KS dataset sanity checks: energy spectra vs wavenumber and parameter-space views.

Validation tip (literature, not computed here): for standard chaotic KS, the maximal
Lyapunov exponent is often reported around ~0.6–0.8 (domain- and normalization-
dependent). Use dedicated long-horizon tangent dynamics if you need λ_max numerically;
this script only plots spectral and parameter summaries from the stored pickle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eff_physics_learn_dataset import load_pde_dataset


def _time_mean_one_spectrum(u: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_rad_per_len, E_mean) for one-sided radial wavenumbers (periodic x).

    u: (n_x, n_t). E(k) = time mean of |fft(u)(·,t)|^2 / n_x so sum_k E(k) ~ mean_t sum_x u^2.
    """
    nx = u.shape[0]
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    uh = np.fft.fft(u, axis=0)
    e_xt = (np.abs(uh) ** 2) / nx
    e_mean = e_xt.mean(axis=1)
    sl = slice(0, nx // 2 + 1)
    return k[sl], e_mean[sl]


def _spectral_centroid(k: np.ndarray, e: np.ndarray) -> float:
    """Intensity-weighted mean |k| (skip DC)."""
    if e.size < 2:
        return float("nan")
    k_abs = np.abs(k[1:])
    w = e[1:]
    s = float(np.sum(w))
    if s <= 0.0:
        return float("nan")
    return float(np.sum(k_abs * w) / s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot KS energy spectra and parameter-space views for dataset validation.",
        epilog=(
            "Chaos check: compare against published λ_max ≈ 0.6–0.8 for chaotic KS "
            "(setup-dependent); this script does not estimate Lyapunov exponents."
        ),
    )
    ap.add_argument("--data-dir", "-d", default="datasets/new", help="Datasets root")
    ap.add_argument(
        "--out",
        "-o",
        default="docs/_assets/results/ks_validation.png",
        help="Output PNG path",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="Cap samples for speed")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    ds = load_pde_dataset("ks", data_dir=Path(args.data_dir))
    Xg = ds.grids.get("X_grid")
    if Xg is None:
        raise KeyError("KS pickle missing X_grid")
    x = np.asarray(Xg[:, 0], dtype=np.float64)

    n = len(ds)
    idx = np.arange(n, dtype=np.int64)
    if args.max_samples is not None:
        idx = idx[: int(args.max_samples)]

    specs: list[np.ndarray] = []
    centroids: list[float] = []
    k = np.array([], dtype=np.float64)
    for i in idx:
        k, e = _time_mean_one_spectrum(ds.u[int(i)], x)
        specs.append(e)
        centroids.append(_spectral_centroid(k, e))

    spec_stack = np.stack(specs, axis=0)
    spec_m = np.median(spec_stack, axis=0)
    spec_lo = np.percentile(spec_stack, 10, axis=0)
    spec_hi = np.percentile(spec_stack, 90, axis=0)

    a = ds.params[idx, ds.param_names.index("alpha")]
    b = ds.params[idx, ds.param_names.index("beta")]
    g = ds.params[idx, ds.param_names.index("gamma")]
    c = np.asarray(centroids, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.fill_between(k, spec_lo, spec_hi, alpha=0.25, label="10–90% over samples")
    ax.plot(k, spec_m, color="C0", lw=1.5, label="Median")
    ax.set_xlabel(r"$k$ (rad / length)")
    ax.set_ylabel(r"time-mean $E(k)$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Energy spectrum (one-sided FFT)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    sc = axes[1].scatter(a, b, c=g, cmap="viridis", s=18, alpha=0.85, edgecolors="none")
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.85)
    cb.set_label(r"$\gamma$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\beta$")
    axes[1].set_title(r"Parameter plane $(\alpha,\beta)$")
    axes[1].grid(True, ls=":", alpha=0.5)

    sc2 = axes[2].scatter(a, g, c=c, cmap="magma", s=18, alpha=0.85, edgecolors="none")
    cb2 = fig.colorbar(sc2, ax=axes[2], shrink=0.85)
    cb2.set_label(r"median $|k|$ weight")
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"$\gamma$")
    axes[2].set_title(r"$(\alpha,\gamma)$ + spectral centroid")
    axes[2].grid(True, ls=":", alpha=0.5)

    fig.suptitle(
        "KS dataset validation (spectra + params). "
        r"Literature chaotic KS: $\lambda_{\max}\approx 0.6$–$0.8$ (not estimated here).",
        fontsize=10,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
