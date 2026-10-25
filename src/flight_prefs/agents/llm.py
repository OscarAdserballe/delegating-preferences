"""LLMAgent: structured state → the paper's prompt → model → parsed pick (one retry with a format reminder)."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from flight_prefs import answer_parsing, prompts
from flight_prefs.agents.base import Decision, DialogueState, Usage
from flight_prefs.model_client import Completion, ModelSpec
from flight_prefs.paper_data import Menu

__all__ = ["LLMAgent", "ModelSpec"]


class Client(Protocol):
    async def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> Completion: ...


def _sum(a: Usage, b: Usage) -> Usage:
    return replace(
        b,
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
        cost=a.cost + b.cost,
    )


class LLMAgent:
    def __init__(self, spec: ModelSpec, client: Client):
        self.spec = spec
        self.client = client
        self.name = spec.arm

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        messages = prompts.build_messages(state, menu)
        reply = await self.client.complete(self.spec, messages)
        pick = answer_parsing.parse_pick(reply.text)
        if pick is not None:
            return Decision(pick, reply.text, reply.thinking, reply.usage)
        retry = await self.client.complete(self.spec, prompts.with_format_reminder(messages))
        pick = answer_parsing.parse_pick(retry.text)
        return Decision(pick, retry.text, retry.thinking, _sum(reply.usage, retry.usage), retried=True)
