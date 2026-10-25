"""Load the paper's Zenodo evaluation data: 624 users × 5 rounds, and 100 shared held-out menus."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

FEATURES = ("departure_time", "duration", "number_of_stops", "price")
INTERACTION_FILE = Path("data/eval/interaction/flight.jsonl")
HELDOUT_FILE = Path("data/eval/heldout/flight.json")


@dataclass(frozen=True)
class Flight:
    text: str  # "Flight 1: departure time: 02:00 PM, …" (verbatim from the paper)
    features: tuple[float, ...]  # (4,), in [0, 1]: departure, duration, stops, price


@dataclass(frozen=True)
class Menu:
    flights: tuple[Flight, Flight, Flight]

    @cached_property
    def phi(self) -> np.ndarray:
        """(3, 4) feature matrix."""
        return np.array([f.features for f in self.flights])


@dataclass(frozen=True, eq=False)
class User:
    user_id: int
    theta: np.ndarray  # (4,), entries in {-1, -0.5, 0, 0.5, 1}
    rounds: tuple[Menu, ...]  # the 5 interaction menus
    choices: tuple[int, ...]  # the user's pick in each round (0-based)
    heldout_choices: tuple[int, ...]  # 100: the user's pick in each held-out menu


@dataclass(frozen=True, eq=False)
class PaperData:
    users: dict[int, User]  # 624
    heldout: tuple[Menu, ...]  # 100, shared by all users
    thetas: np.ndarray = field(repr=False)  # (624, 4), row i = users[i].theta


def _attribute_values(text: str) -> list[str]:
    """'Flight 1: departure time: 02:00 PM, duration: 30 min, …' → ['02:00 PM', '30 min', …]."""
    body = text.split(": ", 1)[1]
    return [part.split(": ", 1)[1] for part in body.split(", ")]


def build_feature_lookup(heldout_raw: dict) -> list[dict[str, float]]:
    """Per feature, map the displayed value (e.g. '$370') to its 0–1 value, from the held-out file."""
    lookup: list[dict[str, float]] = [{} for _ in FEATURES]
    for texts, numbers in zip(heldout_raw["all_options"], heldout_raw["all_options_numpy"], strict=True):
        for text, vec in zip(texts, numbers, strict=True):
            for j, (shown, value) in enumerate(zip(_attribute_values(text), vec, strict=True)):
                if lookup[j].setdefault(shown, value) != value:
                    raise ValueError(f"conflicting value for {FEATURES[j]} {shown!r}")
    return lookup


def flight_from_text(text: str, lookup: list[dict[str, float]]) -> Flight:
    values = _attribute_values(text)
    if len(values) != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} attributes in {text!r}")
    return Flight(text=text, features=tuple(lookup[j][v] for j, v in enumerate(values)))


def _menu(texts: list[str], lookup: list[dict[str, float]]) -> Menu:
    a, b, c = (flight_from_text(t, lookup) for t in texts)
    return Menu((a, b, c))


def load(paper_dir: Path) -> PaperData:
    heldout_raw = json.loads((paper_dir / HELDOUT_FILE).read_text())
    if tuple(heldout_raw["features"]) != FEATURES:
        raise ValueError(f"unexpected feature order {heldout_raw['features']}")
    lookup = build_feature_lookup(heldout_raw)
    heldout = tuple(_menu(m, lookup) for m in heldout_raw["all_options"])

    users: dict[int, User] = {}
    with (paper_dir / INTERACTION_FILE).open() as fh:
        for line in fh:
            raw = json.loads(line)
            uid = int(raw["idx"])
            rounds = tuple(_menu(r["options"], lookup) for r in raw["rounds"])
            if "rounds_numpy" in raw and not np.allclose([m.phi for m in rounds], raw["rounds_numpy"]):
                raise ValueError(f"text→feature lookup disagrees with rounds_numpy for user {uid}")
            held = heldout_raw["user_idxs"][uid]
            if held["reward_fn"] != raw["reward_fn"]:
                raise ValueError(f"reward_fn mismatch between files for user {uid}")
            users[uid] = User(
                user_id=uid,
                theta=np.array(raw["reward_fn"], dtype=float),
                rounds=rounds,
                choices=tuple(int(r["user_idx"]) for r in raw["rounds"]),
                heldout_choices=tuple(int(i) for i in held["idxs"]),
            )
    if sorted(users) != list(range(len(users))):
        raise ValueError("user ids are not 0..n-1")
    thetas = np.stack([users[i].theta for i in range(len(users))])
    return PaperData(users=users, heldout=heldout, thetas=thetas)
