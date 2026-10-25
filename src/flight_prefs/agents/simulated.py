"""Local agents that check the pipeline: random floor, oracle ceiling, position bias, surface copying."""

from __future__ import annotations

import numpy as np

from flight_prefs.agents.base import Decision, DialogueState
from flight_prefs.agents.tiebreak import argmax_random_tiebreak, rng_for
from flight_prefs.paper_data import Menu, User


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 1):
        self.seed = seed

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        return Decision(pick=int(rng_for(self.seed, state, menu).integers(len(menu.flights))))


class OracleAgent:
    """Knows the user: their θ and, where the data records it, how they break ties."""

    name = "oracle"

    def __init__(self, user: User, heldout: tuple[Menu, ...]):
        self.theta = user.theta
        self.labels: dict[Menu, int] = {}
        for menus, picks in ((user.rounds, user.choices), (heldout, user.heldout_choices)):
            self.labels.update(zip(menus, picks, strict=True))

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        if menu in self.labels:
            return Decision(pick=self.labels[menu])
        return Decision(pick=argmax_random_tiebreak(menu.phi @ self.theta, 3, state, menu))


class AlwaysFirstAgent:
    name = "always_first"

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        return Decision(pick=0)


class CopyLastChoiceAgent:
    """Picks the flight nearest (Euclidean, on the 0–1 features) to the user's most recent choice."""

    name = "copy_last"

    def __init__(self, seed: int = 2):
        self.seed = seed

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        if not state.past:
            return Decision(pick=int(rng_for(self.seed, state, menu).integers(len(menu.flights))))
        last = state.past[-1]
        target = last.menu.phi[last.user_pick]
        return Decision(pick=int(np.argmin(np.linalg.norm(menu.phi - target, axis=1))))
