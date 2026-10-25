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
        ("Based on the information provided, Flight 3 appears to be the best option.", 2),
        ("Given your preference for Flight 1, the best option is indeed Flight 1", 0),
        ("Based on your preferences, the best option appears to be Flight 1. It", 0),
        ("The best option based on your preference for Flight 1 is Flight 3.", 2),
        ("**Flight 2**", 1),
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
        "Given your preference for Flight 1, the best option based on your criteria is",
        "Given your preference for Flight 1, the best option that aligns with your",
        "Based on the preferences for a direct flight and a reasonable price, Flight 1",
        "Given your preference for Flight 1, it seems you value a later departure time",
    ],
)
def test_refuses_to_guess(text: str | None) -> None:
    assert parse_pick(text) is None
