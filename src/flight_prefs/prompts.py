"""The paper's wording (Supp. Table H3) and chat layout (as in its Zenodo training data), canonical history."""

from __future__ import annotations

from flight_prefs.agents.base import DialogueState, PastRound
from flight_prefs.paper_data import Menu

INSTRUCTION = (
    "Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a "
    "flight, and these preferences remain the same. You need to figure out my preferences and select the best "
    "flights for me. Use your best judgment if you are unsure. Do not say you need more information."
)
QUESTION = "Which flight is the best option?"
FORMAT_REMINDER = "Answer with exactly one sentence of the form 'The best option is Flight N.'"
FORCE_ANSWER = f"Stop thinking and answer now. {FORMAT_REMINDER}"


def render_menu(menu: Menu) -> str:
    """'Flight 1: departure time: …' → 'Flight 1:\\ndeparture time: …', one flight per line, as in the paper's data."""
    return "\n".join(f.text.replace(": ", ":\n", 1) for f in menu.flights)


def ask(menu: Menu) -> str:
    return f"{QUESTION}\n\n{render_menu(menu)}"


def canonical_answer(pick: int) -> str:
    return f"The best option is Flight {pick + 1}."


def feedback_line(past: PastRound) -> str:
    if past.agent_pick is None:
        return f"I prefer Flight {past.user_pick + 1}."
    if past.agent_pick == past.user_pick:
        return f"Your option Flight {past.agent_pick + 1} is correct."
    return f"Your option Flight {past.agent_pick + 1} is incorrect. I prefer Flight {past.user_pick + 1}."


def build_messages(state: DialogueState, menu: Menu) -> list[dict[str, str]]:
    """Canonical history: past assistant turns are exactly 'The best option is Flight N.'.

    A past round the agent failed to answer (pick None) is re-sent as its raw reply.
    """
    messages: list[dict[str, str]] = []
    prefix = INSTRUCTION
    for past in state.past:
        messages.append({"role": "user", "content": f"{prefix}\n\n{ask(past.menu)}"})
        if past.agent_pick is not None:
            answer = canonical_answer(past.agent_pick)
        else:
            answer = (past.agent_reply or "").strip() or "(no answer)"
        messages.append({"role": "assistant", "content": answer})
        prefix = feedback_line(past)
    messages.append({"role": "user", "content": f"{prefix}\n\n{ask(menu)}"})
    return messages


def with_format_reminder(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    *head, last = messages
    return [*head, {"role": last["role"], "content": f"{last['content']}\n\n{FORMAT_REMINDER}"}]


def with_forced_answer(messages: list[dict[str, str]], partial_thinking: str) -> list[dict[str, str]]:
    """Budget forcing: the cut-off thinking goes back as the assistant's turn, then the user asks for the answer."""
    return [
        *messages,
        {"role": "assistant", "content": partial_thinking.strip()},
        {"role": "user", "content": FORCE_ANSWER},
    ]


def transcript(messages: list[dict[str, str]]) -> str:
    """Human-readable 'User: … / Model: …' view, as in Table H3."""
    names = {"user": "User", "assistant": "Model"}
    return "\n\n".join(f"{names[m['role']]}: {m['content']}" for m in messages)
