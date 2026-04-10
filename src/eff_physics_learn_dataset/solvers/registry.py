from __future__ import annotations

from typing import Dict

from .base import Solver

_SOLVERS: Dict[str, Solver] = {}


def register_solver(name: str, solver: Solver) -> None:
    """Register a solver implementation under a string name."""

    _SOLVERS[name] = solver


def get_solver(name: str) -> Solver:
    """Return a registered solver by name."""

    try:
        return _SOLVERS[name]
    except KeyError as exc:  # pragma: no cover - trivial
        raise KeyError(f"Unknown solver: {name!r}. Available: {list(_SOLVERS.keys())}") from exc


def list_solvers() -> Dict[str, Solver]:
    """Return a shallow copy of the solver registry."""

    return dict(_SOLVERS)

