from __future__ import annotations

import asyncio

import numpy as np

from flight_prefs import metrics
from flight_prefs.agents.base import Decision, DialogueState
from flight_prefs.agents.simulated import CopyLastChoiceAgent, OracleAgent, RandomAgent
from flight_prefs.interaction_loop import run_user
from flight_prefs.paper_data import Menu, PaperData


class Recorder:
    """Picks Flight 2 always and remembers the history length of every call."""

    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[tuple[int, tuple[int | None, ...]]] = []

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        self.seen.append((len(state.past), tuple(p.agent_pick for p in state.past)))
        return Decision(pick=1)


def test_loop_shape_and_branching(data: PaperData) -> None:
    user = data.users[378]
    agent = Recorder()
    recs = asyncio.run(run_user(agent, user, data.heldout))
    assert len(recs) == 605
    assert sum(r.kind == "dialogue" for r in recs) == 5
    held = [r for r in recs if r.kind == "heldout"]
    assert sorted({r.checkpoint for r in held}) == list(range(6))
    assert all(sum(r.checkpoint == c for r in held) == 100 for c in range(6))
    # held-out branches never enter the dialogue: history length k always has exactly k past picks of 1
    assert all(picks == (1,) * k for k, picks in agent.seen)
    assert sum(k == 5 for k, _ in agent.seen) == 100


def test_unparsed_counts_as_wrong_with_worst_regret(data: PaperData) -> None:
    class Mute:
        name = "mute"

        async def choose(self, state: DialogueState, menu: Menu) -> Decision:
            return Decision(pick=None, raw_text="I need more information.")

    user = data.users[0]
    recs = asyncio.run(run_user(Mute(), user, data.heldout))
    assert not any(r.correct for r in recs)
    worst = np.mean([(m.phi @ user.theta).max() - (m.phi @ user.theta).min() for m in data.heldout])
    assert np.isclose(np.mean([r.regret for r in recs if r.kind == "heldout" and r.checkpoint == 0]), worst)


def test_simulated_agents_and_metrics(data: PaperData) -> None:
    users = [data.users[u] for u in (378, 381, 568)]

    async def go() -> list:
        jobs = [
            run_user(a, u, data.heldout)
            for u in users
            for a in (OracleAgent(u, data.heldout), RandomAgent(), CopyLastChoiceAgent())
        ]
        return [r for rs in await asyncio.gather(*jobs) for r in rs]

    df = metrics.to_frame(asyncio.run(go()))
    acc = metrics.wide(metrics.accuracy_by_checkpoint(df, n_boot=200))
    assert (acc.loc["oracle"] == 1.0).all()
    assert (metrics.wide(metrics.regret_by_checkpoint(df, n_boot=200)).loc["oracle"] == 0).all()
    assert 0.2 < float(acc.loc["random"].mean()) < 0.45
    d = metrics.paired_difference(df, "oracle", "random", 5, n_boot=200)
    assert d.lo <= d.mean <= d.hi and d.n_users == 3


def test_bootstrap_ci_covers_mean() -> None:
    e = metrics.bootstrap_users(np.array([0.2, 0.4, 0.6, 0.8]), n_boot=1000)
    assert e.lo < e.mean < e.hi and np.isclose(e.mean, 0.5)
