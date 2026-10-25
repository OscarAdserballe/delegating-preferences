"""LLMAgent: structured state → the paper's prompt → model → parsed pick.

No pick: if thinking was cut off at max_tokens (and the arm forces answers), one no-thinking call answers from the
partial thinking; otherwise one retry with a format reminder.
"""

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

    def _cut_off(self, reply: Completion) -> bool:
        return self.spec.force_answer and reply.usage.finish_reason == "length"

    async def _force(self, messages: list[dict[str, str]], reply: Completion, usage: Usage, retried: bool) -> Decision:
        spec = replace(self.spec, thinking=False, thinking_budget=None, max_tokens=16, force_answer=False)
        partial = reply.thinking or reply.text or ""
        answer = await self.client.complete(spec, prompts.with_forced_answer(messages, partial))
        pick = answer_parsing.parse_pick(answer.text)
        return Decision(pick, answer.text, reply.thinking, _sum(usage, answer.usage), retried=retried, forced=True)

    async def choose(self, state: DialogueState, menu: Menu) -> Decision:
        messages = prompts.build_messages(state, menu)
        reply = await self.client.complete(self.spec, messages)
        pick = answer_parsing.parse_pick(reply.text)
        if pick is not None:
            return Decision(pick, reply.text, reply.thinking, reply.usage)
        if self._cut_off(reply):
            return await self._force(messages, reply, reply.usage, retried=False)
        reminded = prompts.with_format_reminder(messages)
        retry = await self.client.complete(self.spec, reminded)
        usage = _sum(reply.usage, retry.usage)
        pick = answer_parsing.parse_pick(retry.text)
        if pick is None and self._cut_off(retry):
            return await self._force(reminded, retry, usage, retried=True)
        return Decision(pick, retry.text, retry.thinking, usage, retried=True)
