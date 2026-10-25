"""Utility θᵀφ, best options and regret."""

from __future__ import annotations

import numpy as np

from flight_prefs.paper_data import Menu

TIE_TOL = 1e-9


def utilities(theta: np.ndarray, menu: Menu) -> np.ndarray:
    return menu.phi @ theta


def best_options(theta: np.ndarray, menu: Menu) -> frozenset[int]:
    u = utilities(theta, menu)
    return frozenset(int(i) for i in np.flatnonzero(u >= u.max() - TIE_TOL))


def regret(theta: np.ndarray, menu: Menu, pick: int | None) -> float:
    """U(best) − U(pick); an unparsed pick (None) counts as the worst option."""
    u = utilities(theta, menu)
    chosen = u.min() if pick is None else u[pick]
    return float(u.max() - chosen)
