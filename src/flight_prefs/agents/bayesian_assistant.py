"""Exact Bayesian Assistant: uniform prior over the 624 reward functions, keep the θ consistent with every choice."""

from __future__ import annotations

import numpy as np

from flight_prefs.agents.base import Decision, DialogueState
from flight_prefs.agents.tiebreak import argmax_random_tiebreak
from flight_prefs.flight_utility import TIE_TOL
from flight_prefs.paper_data import Menu


def consistent(thetas: np.ndarray, menu: Menu, user_pick: int) -> np.ndarray:
    """1.0 for every θ under which `user_pick` is a best option of `menu`, else 0.0."""
    u = thetas @ menu.phi.T  # (n, 3)
    return (u[:, user_pick] >= u.max(axis=1) - TIE_TOL).astype(float)


class BayesianAssistant:
    name = "bayes"

    def __init__(self, thetas: np.ndarray, seed: int = 0):
        self.thetas = thetas
        self.seed = seed
        self._cache: dict[DialogueState, np.ndarray] = {}

    def posterior(self, state: DialogueState) -> np.ndarray:
        if state in self._cache:
            return self._cache[state]
        if not state.past:
            post = np.full(len(self.thetas), 1.0 / len(self.thetas))
        else:
            last = state.past[-1]
            post = self.posterior(DialogueState(state.past[:-1])) * consistent(self.thetas, last.menu, last.user_pick)
            post = post / post.sum()
        self._cache[state] = post
        return post

    def pick(self, state: DialogueState, menu: Menu) -> int:
        mean_theta = self.posterior(state) @ self.thetas
        return argmax_random_tiebreak(menu.phi @ mean_theta, self.seed, state, menu)

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        return Decision(pick=self.pick(state, menu))
