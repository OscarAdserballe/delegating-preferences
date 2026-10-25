"""Reply text → 0-based pick, or None when the reply doesn't commit to one flight."""

from __future__ import annotations

import re

_FLIGHT = r"\**\s*Flight\s*\**\s*([123])\b"
_VERB = r"(?:is|would\s+be|(?:appears|seems)\s+to\s+be|is\s+indeed|remains|:)"
_DECLARATIONS = (
    re.compile(r"best\s+(?:option|choice|flight)\b[^.\n]{0,60}?" + _VERB + r"\s*" + _FLIGHT, re.I),
    re.compile(_FLIGHT + r"\**,?\s*" + _VERB + r"\s+(?:the\s+)?(?:best|optimal)", re.I),
    re.compile(r"\bI\s+(?:would\s+)?(?:recommend|choose|pick|suggest|select)\s*" + _FLIGHT, re.I),
)
_BARE = re.compile(r"^\W*" + _FLIGHT + r"\W*$", re.I)


def parse_pick(text: str | None) -> int | None:
    """The last explicit declaration ('The best option is Flight N.', 'Flight N is the best option'), or a bare
    'Flight N'. A flight merely mentioned (e.g. in a truncated analysis) is not a pick."""
    if not text:
        return None
    found = [(m.start(), m.group(1)) for pat in _DECLARATIONS for m in pat.finditer(text)]
    if found:
        return int(max(found)[1]) - 1
    bare = _BARE.match(text)
    return int(bare.group(1)) - 1 if bare else None
