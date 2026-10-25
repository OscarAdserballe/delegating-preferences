from __future__ import annotations

import numpy as np

from flight_prefs.flight_utility import best_options, regret, utilities
from flight_prefs.paper_data import Flight, Menu, PaperData


def menu(*features: tuple[float, ...]) -> Menu:
    a, b, c = (Flight(f"Flight {i + 1}: x", f) for i, f in enumerate(features))
    return Menu((a, b, c))


def test_utilities_and_regret() -> None:
    m = menu((0.0, 0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 0.9))
    theta = np.array([0, 0, 0, -1.0])
    assert np.allclose(utilities(theta, m), [-0.1, -0.5, -0.9])
    assert best_options(theta, m) == frozenset({0})
    assert regret(theta, m, 0) == 0.0
    assert np.isclose(regret(theta, m, 2), 0.8)
    assert np.isclose(regret(theta, m, None), 0.8)


def test_ties_are_all_best() -> None:
    m = menu((0.5, 0.0, 0.0, 0.1), (0.5, 1.0, 0.0, 0.1), (0.0, 0.0, 0.0, 0.1))
    assert best_options(np.array([1.0, 0, 0, 0]), m) == frozenset({0, 1})


def test_paper_labels_are_best_options(data: PaperData) -> None:
    for u in data.users.values():
        for m, c in zip(u.rounds, u.choices, strict=True):
            assert best_options(u.theta, m) == frozenset({c})
        for m, c in zip(data.heldout, u.heldout_choices, strict=True):
            assert c in best_options(u.theta, m)


def test_data_shape(data: PaperData) -> None:
    assert len(data.users) == 624 and len(data.heldout) == 100
    assert data.thetas.shape == (624, 4)
    assert not np.any(np.all(data.thetas == 0, axis=1))
