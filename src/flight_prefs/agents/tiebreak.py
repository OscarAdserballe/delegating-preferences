"""Seeded randomness that depends only on (seed, state, menu), so results don't depend on call order."""

from __future__ import annotations

import hashlib

import numpy as np

from flight_prefs.agents.base import DialogueState
from flight_prefs.flight_utility import TIE_TOL
from flight_prefs.paper_data import Menu


def rng_for(seed: int, state: DialogueState, menu: Menu) -> np.random.Generator:
    parts = [str(seed)]
    for p in state.past:
        parts += [f.text for f in p.menu.flights] + [str(p.agent_pick), str(p.user_pick)]
    parts += [f.text for f in menu.flights]
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def argmax_random_tiebreak(values: np.ndarray, seed: int, state: DialogueState, menu: Menu) -> int:
    best = np.flatnonzero(values >= values.max() - TIE_TOL)
    return int(best[0]) if len(best) == 1 else int(rng_for(seed, state, menu).choice(best))
