from __future__ import annotations

import pytest

from flight_prefs.answer_parsing import parse_pick


@pytest.mark.parametrize(
    ("text", "pick"),
    [
        ("The best option is Flight 1.", 0),
        ("The best option is Flight 3", 2),
        ("The best option is **Flight 2**. It has a departure time of", 1),
        ("Flight 1 is the best option.", 0),
        ("**Flight 3** is the best option for you.", 2),
        ("Flight 2", 1),
        ("I would recommend Flight 3.", 2),
        ("Flight 1 looks cheap, but the best option is Flight 2.", 1),
        ("The best option is Flight 1.\n\nActually, the best option is Flight 3.", 2),
    ],
)
def test_parses(text: str, pick: int) -> None:
    assert parse_pick(text) == pick


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "Based on your preferences, it seems you value a combination of price, duration,",
        "Based on your preferences, let's analyze each flight:\n\n1. **Flight 1**: Departure at 06:48 PM, duration of",
        "Let's break down the options:\n\n- **Flight 1**: Departure at 02:00 PM\n- **Flight 2**:",
        "Flight 4 is the best option.",
        "I need more information.",
    ],
)
def test_refuses_to_guess(text: str | None) -> None:
    assert parse_pick(text) is None
