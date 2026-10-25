"""Mid-run look at one arm's held-out accuracy, read from its call logs (approximate: last pick per prompt)."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from flight_prefs import paper_data, prompts
from flight_prefs.answer_parsing import parse_pick
from flight_prefs.call_log import _read_lines

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run_dir, arm = Path(sys.argv[1]), sys.argv[2]
    data = paper_data.load(ROOT / "data" / "paper")
    menu_index = {prompts.ask(m): i for i, m in enumerate(data.heldout)}
    picks: dict[tuple[int, int, int], int | None] = {}
    for path in (run_dir / "calls" / arm).glob("*.jsonl.gz"):
        uid = int(path.name.split(".")[0])
        for e in _read_lines(path):
            last = e["request"]["last_user"].removesuffix("\n\n" + prompts.FORMAT_REMINDER)
            i = next((j for q, j in menu_index.items() if last.endswith(q)), None)
            if i is None:
                continue
            cp = (e["request"]["n_messages"] - 1) // 2
            key = (uid, cp, i)
            if picks.get(key) is None:
                picks[key] = parse_pick(e["text"])
    # checkpoint-0 prompts are shared by every user, so each is logged once under whichever user asked first
    seen_users = {uid for uid, cp, _ in picks if cp > 0}
    shared = {i: p for (_, cp, i), p in picks.items() if cp == 0}
    picks = {k: p for k, p in picks.items() if k[1] > 0}
    picks.update({(uid, 0, i): p for uid in seen_users for i, p in shared.items()})
    acc: dict[int, list[float]] = defaultdict(list)
    users: dict[int, set[int]] = defaultdict(set)
    for (uid, cp, i), p in picks.items():
        acc[cp].append(float(p == data.users[uid].heldout_choices[i]))
        users[cp].add(uid)
    for cp in sorted(acc):
        print(
            f"checkpoint {cp}: {100 * sum(acc[cp]) / len(acc[cp]):5.1f}%  "
            f"({len(acc[cp])} picks, {len(users[cp])} users)"
        )


if __name__ == "__main__":
    main()
