"""One user, one agent: 5 feedback rounds, plus 100 branched held-out picks at each of 6 checkpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from flight_prefs.agents.base import Agent, Decision, DialogueState, PastRound
from flight_prefs.flight_utility import best_options, regret
from flight_prefs.paper_data import Menu, User

N_ROUNDS = 5


@dataclass(frozen=True)
class DecisionRecord:
    arm: str
    user_id: int
    checkpoint: int  # 0..5 = rounds of history seen
    kind: Literal["dialogue", "heldout"]
    menu_index: int  # round index or held-out index
    pick: int | None
    label: int  # the paper's labelled user choice
    correct: bool  # pick == label (the paper's metric)
    correct_tie_aware: bool  # pick is one of the best options under θ
    regret: float
    retried: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    finish_reason: str | None = None


def _record(
    arm: str,
    user: User,
    checkpoint: int,
    kind: Literal["dialogue", "heldout"],
    index: int,
    menu: Menu,
    label: int,
    d: Decision,
) -> DecisionRecord:
    u = d.usage
    return DecisionRecord(
        arm=arm,
        user_id=user.user_id,
        checkpoint=checkpoint,
        kind=kind,
        menu_index=index,
        pick=d.pick,
        label=label,
        correct=d.pick == label,
        correct_tie_aware=d.pick is not None and d.pick in best_options(user.theta, menu),
        regret=regret(user.theta, menu, d.pick),
        retried=d.retried,
        prompt_tokens=u.prompt_tokens if u else 0,
        completion_tokens=u.completion_tokens if u else 0,
        reasoning_tokens=u.reasoning_tokens if u else 0,
        cost=u.cost if u else 0.0,
        finish_reason=u.finish_reason if u else None,
    )


async def run_user(agent: Agent, user: User, heldout: tuple[Menu, ...], arm: str | None = None) -> list[DecisionRecord]:
    arm = arm or agent.name

    async def score(state: DialogueState, checkpoint: int, i: int, menu: Menu) -> DecisionRecord:
        d = await agent.choose(state, menu)  # a branch: no feedback, never enters the dialogue
        return _record(arm, user, checkpoint, "heldout", i, menu, user.heldout_choices[i], d)

    state = DialogueState()
    tasks: list[asyncio.Task[DecisionRecord]] = []
    dialogue: list[DecisionRecord] = []
    try:
        for r in range(N_ROUNDS + 1):
            tasks += [asyncio.create_task(score(state, r, i, m)) for i, m in enumerate(heldout)]
            if r == N_ROUNDS:
                break
            menu = user.rounds[r]
            d = await agent.choose(state, menu)
            dialogue.append(_record(arm, user, r, "dialogue", r, menu, user.choices[r], d))
            state = state.after(PastRound(menu, d.pick, user.choices[r]))
        return [*dialogue, *await asyncio.gather(*tasks)]
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
