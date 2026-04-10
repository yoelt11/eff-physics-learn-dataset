"""Declarative axis semantics for PDE solution tensors (plotting, similarity, validation).

Storage convention after batch dimension ``N``:

- ``xt``: ``(n_x, n_t)`` — 1D + time, ``meshgrid(x, t, indexing='ij')``
- ``xy``: ``(n_x, n_y)`` — 2D elliptic
- ``xyz``: ``(n_x, n_y, n_z)`` — 3D elliptic
- ``txy``: ``(n_t, n_x, n_y)`` — time-dependent 2D (time first)

Optional pickle metadata override::

    metadata[\"field_layout\"] = {
        \"layout_id\": \"txy\",
        \"u_axes\": [\"t\", \"x\", \"y\"],  # optional if default matches layout_id
    }
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_DEFAULT_U_AXES: dict[str, tuple[str, ...]] = {
    "xt": ("x", "t"),
    "xy": ("x", "y"),
    "xyz": ("x", "y", "z"),
    "txy": ("t", "x", "y"),
}


@dataclass(frozen=True)
class FieldLayout:
    """Semantic layout for ``u`` with shape ``(N, *u_axes)``."""

    layout_id: str
    u_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.layout_id not in _DEFAULT_U_AXES:
            raise ValueError(f"Unknown layout_id={self.layout_id!r}; expected one of {sorted(_DEFAULT_U_AXES)}")
        expected = _DEFAULT_U_AXES[self.layout_id]
        if self.u_axes != expected:
            # Allow custom ordering only if same multiset (future); for now require exact match.
            if set(self.u_axes) != set(expected) or len(self.u_axes) != len(expected):
                raise ValueError(f"u_axes {self.u_axes} incompatible with layout_id {self.layout_id!r} (expected {expected})")


def axis_index(layout: FieldLayout, name: str) -> int:
    """Index into ``u_axes`` (0-based), not counting batch dimension."""
    try:
        return layout.u_axes.index(name)
    except ValueError as e:
        raise KeyError(f"Axis {name!r} not in layout {layout.u_axes}") from e


def default_plot_slice_axis(layout: FieldLayout) -> int:
    """Batch-axis index (1..ndim-1) used to reduce 4D fields to 2D for default smoke plots."""
    if layout.layout_id == "txy":
        return axis_index(layout, "t") + 1
    if layout.layout_id == "xyz":
        return axis_index(layout, "x") + 1
    raise ValueError(f"No default plot slice axis for layout {layout.layout_id}")


def default_similarity_slice_axis(layout: FieldLayout) -> int:
    """Same as default plot slice for volumetric / time stacks."""
    return default_plot_slice_axis(layout)


def infer_field_layout(
    *,
    equation: str,
    u: np.ndarray,
    grids: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> FieldLayout:
    """Infer layout from metadata, grids, array shape, and equation name."""

    if u.ndim < 3:
        raise ValueError(f"Expected batched solutions u.ndim >= 3, got shape {u.shape}")
    spatial_ndim = u.ndim - 1

    meta_fl = metadata.get("field_layout") if isinstance(metadata, dict) else None
    if isinstance(meta_fl, dict) and meta_fl.get("layout_id"):
        lid = str(meta_fl["layout_id"])
        if lid not in _DEFAULT_U_AXES:
            raise ValueError(f"metadata field_layout.layout_id={lid!r} is not supported")
        axes = meta_fl.get("u_axes")
        if axes is not None:
            return FieldLayout(lid, tuple(str(a) for a in axes))
        return FieldLayout(lid, _DEFAULT_U_AXES[lid])

    g = set(grids.keys())
    eq_l = equation.lower()

    has_xt = "X_grid" in g and "T_grid" in g and "Y_grid" not in g
    has_xy = "X_grid" in g and "Y_grid" in g and "T_grid" not in g and "Z_grid" not in g and "Z" not in g
    has_xyz = ("Z_grid" in g or "Z" in g) and "X_grid" in g and "Y_grid" in g
    has_txy = "T_grid" in g and "X_grid" in g and "Y_grid" in g

    if spatial_ndim == 2 and has_xt:
        return FieldLayout("xt", _DEFAULT_U_AXES["xt"])
    if spatial_ndim == 2 and has_xy:
        return FieldLayout("xy", _DEFAULT_U_AXES["xy"])
    if spatial_ndim == 3 and has_txy:
        return FieldLayout("txy", _DEFAULT_U_AXES["txy"])
    if spatial_ndim == 3 and has_xyz:
        return FieldLayout("xyz", _DEFAULT_U_AXES["xyz"])

    # Name fallbacks (weak)
    if spatial_ndim == 3 and ("wave" in eq_l or "wave2d" in eq_l):
        return FieldLayout("txy", _DEFAULT_U_AXES["txy"])
    if spatial_ndim == 3 and ("helmholtz3" in eq_l or "helmholtz3d" in eq_l):
        return FieldLayout("xyz", _DEFAULT_U_AXES["xyz"])

    raise ValueError(
        f"Cannot infer field layout for equation={equation!r} u.shape={u.shape} grid_keys={sorted(g)}. "
        "Set metadata['field_layout'] = {'layout_id': 'xt'|'xy'|'xyz'|'txy'} in the pickle."
    )


def validate_u_against_layout(
    u: np.ndarray,
    layout: FieldLayout,
    grids: Mapping[str, np.ndarray],
    *,
    strict: bool = False,
) -> None:
    """Check that leading spatial sizes match coordinate grids (best-effort)."""

    msg: list[str] = []

    def _len_from_grid(key: str, axis_2d: int) -> int | None:
        if key not in grids:
            return None
        arr = np.asarray(grids[key])
        if arr.ndim == 2:
            return int(arr.shape[axis_2d])
        if arr.ndim == 1:
            return int(arr.size)
        return None

    if layout.layout_id == "xt":
        nx = _len_from_grid("X_grid", 0)
        nt = _len_from_grid("T_grid", 1)
        if nx is not None and u.shape[1] != nx:
            msg.append(f"u.shape[1]={u.shape[1]} != nx from X_grid ({nx})")
        if nt is not None and u.shape[2] != nt:
            msg.append(f"u.shape[2]={u.shape[2]} != nt from T_grid ({nt})")
    elif layout.layout_id == "xy":
        nx = _len_from_grid("X_grid", 0)
        ny = _len_from_grid("Y_grid", 1)
        if nx is not None and u.shape[1] != nx:
            msg.append(f"u.shape[1]={u.shape[1]} != nx from X_grid ({nx})")
        if ny is not None and u.shape[2] != ny:
            msg.append(f"u.shape[2]={u.shape[2]} != ny from Y_grid ({ny})")
    elif layout.layout_id == "txy":
        nt = _len_from_grid("T_grid", 1)
        nx = _len_from_grid("X_grid", 0)
        ny = _len_from_grid("Y_grid", 1)
        if nt is not None and u.shape[1] != nt:
            msg.append(f"u.shape[1]={u.shape[1]} != nt from T_grid ({nt})")
        if nx is not None and u.shape[2] != nx:
            msg.append(f"u.shape[2]={u.shape[2]} != nx from X_grid ({nx})")
        if ny is not None and u.shape[3] != ny:
            msg.append(f"u.shape[3]={u.shape[3]} != ny from Y_grid ({ny})")
    elif layout.layout_id == "xyz":
        nx = _len_from_grid("X_grid", 0)
        ny = _len_from_grid("Y_grid", 1)
        nz = None
        if "Z_grid" in grids:
            Zg = np.asarray(grids["Z_grid"])
            if Zg.ndim == 3:
                nz = int(Zg.shape[2])
            else:
                nz = _len_from_grid("Z_grid", 1)
        if nz is None and "Z" in grids:
            zz = np.asarray(grids["Z"])
            if zz.ndim == 2:
                nz = int(zz.shape[1])
            elif zz.ndim == 1:
                nz = int(zz.size)
        if nx is not None and u.shape[1] != nx:
            msg.append(f"u.shape[1]={u.shape[1]} != nx from X_grid ({nx})")
        if ny is not None and u.shape[2] != ny:
            msg.append(f"u.shape[2]={u.shape[2]} != ny from Y_grid ({ny})")
        if nz is not None and u.shape[3] != nz:
            msg.append(f"u.shape[3]={u.shape[3]} != nz from Z/Z_grid ({nz})")

    if not msg:
        return
    text = "; ".join(msg)
    if strict:
        raise ValueError(f"Field layout validation failed ({layout.layout_id}): {text}")
    warnings.warn(f"Field layout validation ({layout.layout_id}): {text}", UserWarning, stacklevel=2)



def prepare_batch_for_plot_grid(
    u_batch: np.ndarray,
    layout: FieldLayout,
    *,
    slice_index: int | None,
    slice_axis: int,
) -> tuple[np.ndarray, int | None, int]:
    """Return ``(array, slice_index, slice_axis)`` for :func:`plotting.plot_solution_grid`.

    For ``xt``, returns time–x transposed ``(N, n_t, n_x)``; slice args are ignored (3D batch).
    For ``xy``, returns the batch unchanged.
    For ``txy`` / ``xyz``, :func:`plotting.plot_solution_grid` only slices along axes ``1`` or ``3``;
    if another axis is requested, the array is moved so that axis becomes ``1``.
    """

    u_batch = np.asarray(u_batch)
    if layout.layout_id == "xt":
        return np.transpose(u_batch, (0, 2, 1)), slice_index, slice_axis
    if layout.layout_id == "xy":
        return u_batch, slice_index, slice_axis
    if layout.layout_id in ("txy", "xyz"):
        ax = int(slice_axis)
        if ax < 0:
            ax += u_batch.ndim
        if ax not in (1, 3):
            u_out = np.moveaxis(u_batch, ax, 1)
            return u_out, slice_index, 1
        return u_batch, slice_index, ax
    raise ValueError(f"Unsupported layout for plot grid: {layout.layout_id}")


def extent_for_split_rows(
    layout: FieldLayout,
    grids: Mapping[str, np.ndarray],
) -> tuple[float, float, float, float] | None:
    """``extent`` for :func:`plotting.plot_solution_rows` (horizontal x, vertical t or y)."""

    if layout.layout_id == "xt":
        if "X_grid" not in grids or "T_grid" not in grids:
            return None
        X = np.asarray(grids["X_grid"])
        T = np.asarray(grids["T_grid"])
        return (float(X.min()), float(X.max()), float(T.min()), float(T.max()))
    if layout.layout_id == "txy":
        if "X_grid" not in grids or "Y_grid" not in grids:
            return None
        X = np.asarray(grids["X_grid"])
        Y = np.asarray(grids["Y_grid"])
        return (float(X.min()), float(X.max()), float(Y.min()), float(Y.max()))
    if layout.layout_id == "xyz":
        if "X_grid" not in grids or "Y_grid" not in grids:
            return None
        X = np.asarray(grids["X_grid"])
        Y = np.asarray(grids["Y_grid"])
        return (float(X.min()), float(X.max()), float(Y.min()), float(Y.max()))
    return None


def prepare_batch_for_split_rows(
    u_batch: np.ndarray,
    layout: FieldLayout,
) -> np.ndarray:
    """Transpose ``xt`` batches for row plots; leave other layouts unchanged."""

    u_batch = np.asarray(u_batch)
    if layout.layout_id == "xt":
        return np.transpose(u_batch, (0, 2, 1))
    return u_batch
