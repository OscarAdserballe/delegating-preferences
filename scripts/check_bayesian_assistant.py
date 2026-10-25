"""S0 gate: the exact Bayesian Assistant on all 624 users reproduces the prototype (and the paper's ~80%)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from flight_prefs import paper_data, user_sampling
from flight_prefs.agents.base import DialogueState, PastRound
from flight_prefs.agents.bayesian_assistant import BayesianAssistant
from flight_prefs.flight_utility import regret

ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = np.array([33.4, 56.8, 66.8, 73.0, 77.4, 80.3])
TOLERANCE_PTS = 0.5


def main() -> int:
    data = paper_data.load(ROOT / "data" / "paper")
    bayes = BayesianAssistant(data.thetas)
    n = len(data.users)
    acc = np.zeros((n, 6))
    reg = np.zeros((n, 6))
    for uid, user in data.users.items():
        state = DialogueState()
        for r in range(6):
            picks = [bayes.pick(state, m) for m in data.heldout]
            acc[uid, r] = np.mean(np.array(picks) == np.array(user.heldout_choices))
            reg[uid, r] = np.mean([regret(user.theta, m, p) for m, p in zip(data.heldout, picks, strict=True)])
            if r < 5:
                state = state.after(PastRound(user.rounds[r], None, user.choices[r]))

    random_regret = np.mean(
        [[np.mean([regret(u.theta, m, p) for p in range(3)]) for m in data.heldout] for u in data.users.values()]
    )
    mean = acc.mean(axis=0) * 100
    print(f"users: {n}, held-out menus: {len(data.heldout)}")
    print("checkpoint            " + "".join(f"{r:>8d}" for r in range(6)))
    print("Bayes accuracy (%)    " + "".join(f"{x:8.1f}" for x in mean))
    print("prototype (%)         " + "".join(f"{x:8.1f}" for x in PROTOTYPE))
    print("Bayes mean regret     " + "".join(f"{x:8.3f}" for x in reg.mean(axis=0)))
    print(f"random-pick regret (expected): {random_regret:.3f}")
    print(
        f"between-user SD at checkpoint 5: {acc[:, 5].std(ddof=1) * 100:.1f} pts; "
        f"SD of (cp5 - cp1): {(acc[:, 5] - acc[:, 1]).std(ddof=1) * 100:.1f} pts"
    )
    strata = np.array([user_sampling.stratum(data.users[u].theta) for u in range(n)])
    for g in sorted(set(strata)):
        print(
            f"  stratum {g} (n={np.sum(strata == g):3d}): checkpoint-5 accuracy {acc[strata == g, 5].mean() * 100:.1f}%"
        )
    ok = bool(np.all(np.abs(mean - PROTOTYPE) <= TOLERANCE_PTS))
    print(f"GATE S0: {'PASS' if ok else 'FAIL'} (|Bayes − prototype| ≤ {TOLERANCE_PTS} pts at every checkpoint)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
