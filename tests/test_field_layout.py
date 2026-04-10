"""Tests for field layout inference, plotting prep, and vectorization."""

from __future__ import annotations

import numpy as np
import pytest

from eff_physics_learn_dataset.field_layout import (
    FieldLayout,
    default_similarity_slice_axis,
    infer_field_layout,
    prepare_batch_for_plot_grid,
    prepare_batch_for_split_rows,
    validate_u_against_layout,
)
from eff_physics_learn_dataset.similarity import vectorize_solutions


def _grid_xt(nx: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    return X, T


def test_infer_xt_from_grids() -> None:
    nx, nt = 5, 7
    X, T = _grid_xt(nx, nt)
    u = np.zeros((3, nx, nt))
    layout = infer_field_layout(
        equation="burgers",
        u=u,
        grids={"X_grid": X, "T_grid": T},
        metadata={},
    )
    assert layout.layout_id == "xt"
    assert layout.u_axes == ("x", "t")


def test_infer_xyz_from_grids() -> None:
    nx, ny, nz = 4, 5, 6
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u = np.zeros((2, nx, ny, nz))
    layout = infer_field_layout(
        equation="helmholtz3D",
        u=u,
        grids={"X_grid": X, "Y_grid": Y, "Z_grid": Z},
        metadata={},
    )
    assert layout.layout_id == "xyz"


def test_validate_xyz_3d_meshgrid_no_warning() -> None:
    nx, ny, nz = 3, 4, 5
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u = np.zeros((1, nx, ny, nz))
    layout = FieldLayout("xyz", ("x", "y", "z"))
    validate_u_against_layout(u, layout, {"X_grid": X, "Y_grid": Y, "Z_grid": Z}, strict=False)


def test_infer_xy_from_grids() -> None:
    nx, ny = 4, 6
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.zeros((2, nx, ny))
    layout = infer_field_layout(
        equation="helmholtz2D",
        u=u,
        grids={"X_grid": X, "Y_grid": Y},
        metadata={},
    )
    assert layout.layout_id == "xy"


def test_infer_txy_prefers_t_grid() -> None:
    nx, ny, nt = 3, 4, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")
    _, T = np.meshgrid(x, t, indexing="ij")  # shape (nx, nt) for T component
    Z, _ = np.meshgrid(x, y, indexing="ij")
    u = np.zeros((2, nt, nx, ny))
    layout = infer_field_layout(
        equation="wave2d1",
        u=u,
        grids={"X_grid": X, "Y_grid": Y, "T_grid": T, "Z_grid": Z},
        metadata={},
    )
    assert layout.layout_id == "txy"


def test_metadata_overrides_inference() -> None:
    u = np.zeros((1, 8, 8))
    layout = infer_field_layout(
        equation="weird",
        u=u,
        grids={"X_grid": np.zeros((8, 8)), "Y_grid": np.zeros((8, 8))},
        metadata={"field_layout": {"layout_id": "xy"}},
    )
    assert layout.layout_id == "xy"


def test_prepare_xt_transpose_plot_grid() -> None:
    layout = FieldLayout("xt", ("x", "t"))
    u = np.zeros((2, 3, 4))
    out, si, sa = prepare_batch_for_plot_grid(u, layout, slice_index=None, slice_axis=1)
    assert out.shape == (2, 4, 3)
    assert si is None


def test_prepare_txy_moveaxis_slice_axis_2() -> None:
    layout = FieldLayout("txy", ("t", "x", "y"))
    u = np.zeros((1, 5, 3, 4))
    out, _, sa = prepare_batch_for_plot_grid(u, layout, slice_index=2, slice_axis=2)
    assert sa == 1
    assert out.shape[1] == 3


def test_vectorize_auto_uses_layout_default_for_xyz() -> None:
    layout = FieldLayout("xyz", ("x", "y", "z"))
    u = np.zeros((2, 4, 5, 6))
    u[:, 2, :, :] = 1.0
    v = vectorize_solutions(u, field_layout=layout, featurize="auto")
    assert v.shape == (2, 5 * 6)
    assert np.allclose(v[0].sum(), 5 * 6)


def test_vectorize_flatten_4d() -> None:
    u = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    v = vectorize_solutions(u, featurize="flatten")
    assert v.shape == (2, 3 * 4 * 5)


def test_validate_xt_warns_on_mismatch() -> None:
    nx, nt = 3, 4
    X, T = _grid_xt(nx, nt)
    u = np.zeros((1, nx + 1, nt))
    layout = FieldLayout("xt", ("x", "t"))
    with pytest.warns(UserWarning, match="u.shape"):
        validate_u_against_layout(u, layout, {"X_grid": X, "T_grid": T}, strict=False)


def test_prepare_split_rows_xt() -> None:
    layout = FieldLayout("xt", ("x", "t"))
    u = np.zeros((2, 3, 4))
    assert prepare_batch_for_split_rows(u, layout).shape == (2, 4, 3)


def test_default_similarity_slice_axis() -> None:
    assert default_similarity_slice_axis(FieldLayout("txy", ("t", "x", "y"))) == 1
    assert default_similarity_slice_axis(FieldLayout("xyz", ("x", "y", "z"))) == 1
