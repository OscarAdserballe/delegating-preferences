from __future__ import annotations

import asyncio

import numpy as np

from flight_prefs.agents.base import DialogueState, PastRound
from flight_prefs.agents.bayesian_assistant import BayesianAssistant
from flight_prefs.flight_utility import best_options
from flight_prefs.paper_data import PaperData
from flight_prefs.user_sampling import stratified_order, stratum


def test_true_theta_survives_and_posterior_is_consistent(data: PaperData) -> None:
    bayes = BayesianAssistant(data.thetas)
    for uid in (0, 100, 378):
        u = data.users[uid]
        state = DialogueState()
        for r in range(5):
            state = state.after(PastRound(u.rounds[r], None, u.choices[r]))
            post = bayes.posterior(state)
            assert np.isclose(post.sum(), 1.0) and post[uid] > 0
            for k in np.flatnonzero(post):
                assert all(p.user_pick in best_options(data.thetas[k], p.menu) for p in state.past)


def test_round_zero_is_uniform_and_ties_random(data: PaperData) -> None:
    bayes = BayesianAssistant(data.thetas)
    picks = [asyncio.run(bayes.choose(DialogueState(), m)).pick for m in data.heldout]
    assert set(picks) == {0, 1, 2}
    assert picks == [bayes.pick(DialogueState(), m) for m in data.heldout]


def test_stratified_prefix_matches_population(data: PaperData) -> None:
    order = stratified_order(data.users, seed=1)
    assert sorted(order) == list(range(624))
    counts = np.bincount([stratum(data.users[u].theta) for u in order[:50]], minlength=5)
    assert counts.tolist() == [7, 11, 7, 8, 17]
    assert order == stratified_order(data.users, seed=1)
