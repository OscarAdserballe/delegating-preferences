from __future__ import annotations

import asyncio
from pathlib import Path

from flight_prefs import prompts
from flight_prefs.agents.base import DialogueState, PastRound, Usage
from flight_prefs.agents.llm import LLMAgent
from flight_prefs.call_log import CachedClient, CallLog, call_key
from flight_prefs.interaction_loop import run_user
from flight_prefs.model_client import Completion, ModelSpec
from flight_prefs.paper_data import PaperData

SPEC = ModelSpec(arm="fake", model_id="fake/model", provider="fake")


class FakeServer:
    """Verbose (unparseable) on the first try of every round-0 prompt; always 'Flight 2' when reminded."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> Completion:
        self.calls += 1
        await asyncio.sleep(0)
        if prompts.FORMAT_REMINDER in messages[-1]["content"]:
            text = "The best option is Flight 2."
        elif len(messages) == 1:
            text = "Based on your preferences, let's analyze each flight:"
        else:
            text = "The best option is Flight 1."
        return Completion(text=text, thinking=None, usage=Usage(prompt_tokens=10, completion_tokens=5, cost=1e-6))


def test_messages_follow_table_h3(data: PaperData) -> None:
    u = data.users[0]
    state = DialogueState().after(PastRound(u.rounds[0], 0, 1)).after(PastRound(u.rounds[1], 2, 2))
    msgs = prompts.build_messages(state, u.rounds[2])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"].startswith(prompts.INSTRUCTION + "\n\nWhich flight is the best option?\n\nFlight 1:\n")
    assert msgs[1]["content"] == "The best option is Flight 1."
    assert msgs[2]["content"].startswith("Your option Flight 1 is incorrect. I prefer Flight 2.\n\nWhich flight")
    assert msgs[3]["content"] == "The best option is Flight 3."
    assert msgs[4]["content"].startswith("Your option Flight 3 is correct.\n\nWhich flight is the best option?")
    assert msgs[4]["content"].count("\nFlight ") == 3 and msgs[4]["content"].endswith(u.rounds[2].flights[2].text[-5:])


def test_retry_then_resume_without_network(tmp_path: Path, data: PaperData) -> None:
    user = data.users[378]
    server = FakeServer()
    client = CachedClient(server, CallLog(tmp_path), "fake", user.user_id)  # type: ignore[arg-type]
    recs = asyncio.run(run_user(LLMAgent(SPEC, client), user, data.heldout))
    assert len(recs) == 605 and all(r.pick is not None for r in recs)
    round0 = [r for r in recs if r.checkpoint == 0]
    assert all(r.retried and r.pick == 1 for r in round0)
    assert not any(r.retried for r in recs if r.checkpoint > 0)
    first_run_calls = server.calls
    assert first_run_calls == 605 + 101  # every round-0 prompt needed the reminder

    fresh = FakeServer()
    log = CallLog(tmp_path)
    assert log.loaded == first_run_calls
    again = asyncio.run(run_user(LLMAgent(SPEC, CachedClient(fresh, log, "fake", 378)), user, data.heldout))  # type: ignore[arg-type]
    assert fresh.calls == 0
    assert [r.pick for r in again] == [r.pick for r in recs]


def test_call_key_ignores_arm_but_not_params() -> None:
    msgs = [{"role": "user", "content": "x"}]
    a = ModelSpec(arm="a", model_id="m", provider="p")
    assert call_key(a, msgs) == call_key(ModelSpec(arm="b", model_id="m", provider="p"), msgs)
    assert call_key(a, msgs) != call_key(ModelSpec(arm="a", model_id="m", provider="p", thinking=True), msgs)
