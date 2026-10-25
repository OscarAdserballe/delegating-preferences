"""Reply text → 0-based pick, or None when the reply doesn't commit to one flight."""

from __future__ import annotations

import re

_FLIGHT = r"\**\s*Flight\s*\**\s*([123])\b"
_DECLARATIONS = (
    re.compile(r"best\s+(?:option|choice|flight)\s*(?:for\s+you\s+)?(?:is|would\s+be|:)\s*" + _FLIGHT, re.I),
    re.compile(_FLIGHT + r"\**\s*(?:is|would\s+be|seems\s+to\s+be)\s+(?:the\s+)?(?:best|optimal)", re.I),
    re.compile(r"(?:I\s+(?:would\s+)?(?:recommend|choose|pick|suggest|select))\s*" + _FLIGHT, re.I),
)
_ANY = re.compile(r"Flight\s*\**\s*([123])\b", re.I)
SHORT_REPLY = 80


def parse_pick(text: str | None) -> int | None:
    """The last explicit declaration ('The best option is Flight N.'); else a short reply naming one flight."""
    if not text:
        return None
    found = [(m.start(), m.group(1)) for pat in _DECLARATIONS for m in pat.finditer(text)]
    if found:
        return int(max(found)[1]) - 1
    named = set(_ANY.findall(text))
    if len(text.strip()) <= SHORT_REPLY and len(named) == 1:
        return int(named.pop()) - 1
    return None
