"""The Agent protocol and the structured dialogue state every agent reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flight_prefs.paper_data import Menu


@dataclass(frozen=True)
class PastRound:
    menu: Menu
    agent_pick: int | None
    user_pick: int
    agent_reply: str | None = None  # kept only when agent_pick is None


@dataclass(frozen=True)
class DialogueState:
    past: tuple[PastRound, ...] = ()  # empty at round 0

    def after(self, r: PastRound) -> DialogueState:
        return DialogueState(past=(*self.past, r))


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    provider: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    cached: bool = False  # served from the call log, not the network


@dataclass(frozen=True)
class Decision:
    pick: int | None  # None = unparsed, scored as wrong
    raw_text: str | None = None
    thinking: str | None = None
    usage: Usage | None = None
    retried: bool = False


class Agent(Protocol):
    name: str

    async def choose(self, state: DialogueState, menu: Menu) -> Decision: ...
