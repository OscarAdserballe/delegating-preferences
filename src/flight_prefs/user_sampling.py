"""Seeded user order, stratified by preference strength ‖θ‖, so any prefix has the population's mix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flight_prefs.paper_data import User

# Upper edges on ‖θ‖² of the plan's five groups (88 / 144 / 88 / 96 / 208 users).
NORM_SQ_EDGES = (1.0, 1.5, 2.0, 2.25)


def stratum(theta: np.ndarray) -> int:
    return int(np.searchsorted(NORM_SQ_EDGES, float(theta @ theta) - 1e-9))


def stratified_order(users: dict[int, User], seed: int) -> list[int]:
    """All user ids in a fixed order; each prefix allocates to strata by largest remainder."""
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = {}
    for uid in sorted(users):
        groups.setdefault(stratum(users[uid].theta), []).append(uid)
    keys = sorted(groups)
    queues = {g: list(rng.permutation(groups[g])) for g in keys}
    sizes = {g: len(groups[g]) for g in keys}
    taken = dict.fromkeys(keys, 0)
    n = len(users)
    order: list[int] = []
    for k in range(1, n + 1):
        g = max((g for g in keys if queues[g]), key=lambda g: (k * sizes[g] / n - taken[g], -g))
        order.append(int(queues[g].pop(0)))
        taken[g] += 1
    return order


def save(order: list[int], users: dict[int, User], seed: int, path: Path) -> None:
    payload = {
        "seed": seed,
        "stratified_by": "||theta||^2 groups with upper edges " + str(list(NORM_SQ_EDGES)),
        "order": order,
        "stratum": [stratum(users[u].theta) for u in order],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")


def load(path: Path, n: int | None = None) -> list[int]:
    order: list[int] = json.loads(path.read_text())["order"]
    return order if n is None else order[:n]
