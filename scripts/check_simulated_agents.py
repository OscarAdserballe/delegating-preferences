"""S1 gate: simulated agents and Bayes through the same async loop the LLMs use, scored and plotted."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

from flight_prefs import figures, metrics, paper_data, user_sampling
from flight_prefs.agents.base import Agent
from flight_prefs.agents.bayesian_assistant import BayesianAssistant
from flight_prefs.agents.simulated import AlwaysFirstAgent, CopyLastChoiceAgent, OracleAgent, RandomAgent
from flight_prefs.interaction_loop import DecisionRecord, run_user

ROOT = Path(__file__).resolve().parents[1]


async def run_all(data: paper_data.PaperData, user_ids: list[int]) -> list[DecisionRecord]:
    shared: list[Agent] = [BayesianAssistant(data.thetas), RandomAgent(), AlwaysFirstAgent(), CopyLastChoiceAgent()]
    jobs = []
    for uid in user_ids:
        user = data.users[uid]
        for agent in [*shared, OracleAgent(user, data.heldout)]:
            jobs.append(run_user(agent, user, data.heldout))
    return [rec for recs in await asyncio.gather(*jobs) for rec in recs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "simulated")
    args = ap.parse_args()

    data = paper_data.load(ROOT / "data" / "paper")
    user_ids = user_sampling.load(ROOT / "data" / "user_sample.json", args.n_users)
    df = metrics.to_frame(asyncio.run(run_all(data, user_ids)))

    acc = metrics.accuracy_by_checkpoint(df)
    acc_tie = metrics.accuracy_by_checkpoint(df, tie_aware=True)
    reg = metrics.regret_by_checkpoint(df)
    pd.set_option("display.width", 160)
    for name, t, scale, fmt in (
        ("accuracy (%)", acc, 100, "{:.1f}"),
        ("tie-aware accuracy (%)", acc_tie, 100, "{:.1f}"),
        ("mean regret", reg, 1, "{:.3f}"),
    ):
        print(f"\n{name}, {len(user_ids)} users\n{(metrics.wide(t) * scale).map(fmt.format)}")

    per_user = df.groupby(["arm", "user_id"]).size()
    m = metrics.wide(acc)
    rand_all = float(df[(df["arm"] == "random") & (df["kind"] == "heldout")]["correct"].mean())
    first, bayes0, bayes5 = float(m.at["always_first", 5]), float(m.at["bayes", 0]), float(m.at["bayes", 5])
    oracle = df[df["arm"] == "oracle"]
    checks = {
        "605 decisions per (arm, user)": bool((per_user == 605).all()),
        f"random ≈ 33% over all checkpoints ({rand_all * 100:.1f}%, within 30–37)": 0.30 <= rand_all <= 0.37,
        "oracle 100% accuracy at every checkpoint, incl. dialogue": bool(oracle["correct"].all()),
        "oracle regret 0 everywhere": bool((oracle["regret"] <= 1e-9).all()),
        f"always-first ≈ 33% ({first * 100:.1f}%)": 0.25 <= first <= 0.42,
        f"Bayes rises (cp0 {bayes0 * 100:.1f}% → cp5 {bayes5 * 100:.1f}%)": bayes5 > 0.7,
    }
    print()
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    gain = metrics.learning_gain(df, "bayes")
    diff = metrics.paired_difference(df, "bayes", "copy_last", 5)
    print(f"\nBayes gain cp1→cp5: {gain.mean * 100:+.1f} pts [{gain.lo * 100:+.1f}, {gain.hi * 100:+.1f}]")
    print(f"Bayes − copy_last at cp5 (paired): {diff.mean * 100:+.1f} pts [{diff.lo * 100:+.1f}, {diff.hi * 100:+.1f}]")
    print(metrics.agreement(df, "copy_last", "bayes").round(3).to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "decisions.parquet", index=False)
    fig = figures.results_figure(acc, reg, args.out / "results.png", f"Simulated agents + Bayes, {len(user_ids)} users")
    print(f"\nwrote {args.out / 'decisions.parquet'} and {fig}")
    ok = all(checks.values())
    print(f"GATE S1: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
