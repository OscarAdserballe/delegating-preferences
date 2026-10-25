"""S2 gate: one LLM, one user, end to end. 605 parsed decisions, 0 duplicate calls, transcript as in Table H3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from flight_prefs import paper_data, prompts
from flight_prefs.agents.base import DialogueState, PastRound
from flight_prefs.experiment import log_stats

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "results" / "s2-one-llm")
    ap.add_argument("--arm", default="qwen25-7b")
    args = ap.parse_args()

    data = paper_data.load(ROOT / "data" / "paper")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    df = pd.read_parquet(args.run_dir / "decisions.parquet")
    q = df[df["arm"] == args.arm]
    uid = int(q["user_id"].iloc[0])
    user = data.users[uid]
    dialogue = q[q["kind"] == "dialogue"].sort_values("checkpoint")

    state = DialogueState()
    for _, row in dialogue.head(2).iterrows():
        pick = None if pd.isna(row["pick"]) else int(row["pick"])
        r = int(row["checkpoint"])
        state = state.after(
            PastRound(user.rounds[r], pick, user.choices[r], None if pick is not None else row["raw_text"])
        )
    msgs = prompts.build_messages(state, user.rounds[2])
    print(f"user {uid}, arm {args.arm}: the round-3 prompt (two rounds of history), cf. Supp. Table H3\n")
    print(prompts.transcript(msgs))

    stats = log_stats(args.run_dir)
    held = q[q["kind"] == "heldout"]
    print("\naccuracy by checkpoint (%):", (held.groupby("checkpoint")["correct"].mean() * 100).round(1).to_dict())
    print("needed the format reminder, by checkpoint:", q.groupby("checkpoint")["retried"].mean().round(3).to_dict())
    print("served:", manifest.get("served"), "| cost so far: $", manifest.get("cost_total_usd"))
    print("log:", stats)
    checks = {
        f"605 decisions ({len(q)})": len(q) == 605,
        f"all parsed ({int(q['pick'].notna().sum())}/605)": bool(q["pick"].notna().all()),
        f"0 duplicate calls ({stats['duplicate_calls']})": stats["duplicate_calls"] == 0,
        "one served provider/model": len(manifest.get("served", {})) == 1 and manifest["status"] == "complete",
        "history re-sent canonically": all(
            m["content"].startswith("The best option is Flight ") for m in msgs if m["role"] == "assistant"
        ),
    }
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"GATE S2: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
