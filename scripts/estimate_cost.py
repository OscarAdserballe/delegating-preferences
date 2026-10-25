"""Dry run: project calls, tokens, $ and wall time of a config from a pilot's measured decisions and call log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from flight_prefs.call_log import _read_lines
from flight_prefs.experiment import RunConfig
from flight_prefs.interaction_loop import N_ROUNDS

ROOT = Path(__file__).resolve().parents[1]
N_HELDOUT = 100


def sustained_throughput(pilot_dirs: list[Path]) -> dict[tuple[str, str], float]:
    """Calls per second per (arm, provider): the median over minutes in which that arm logged any call."""
    rows = [
        (e["arm"], str(e["usage"].get("provider")).lower(), e["time"])
        for d in pilot_dirs
        for p in (d / "calls").glob("*/*.jsonl.gz")
        for e in _read_lines(p)
    ]
    df = pd.DataFrame(rows, columns=["arm", "provider", "time"])
    df["minute"] = df["time"] // 60
    per_min = df.groupby(["arm", "provider", "minute"]).size()
    return {(str(a), str(p)): float(v) / 60 for (a, p), v in per_min.groupby(level=[0, 1]).median().items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=ROOT / "results" / "pilot")
    ap.add_argument("--add", type=Path, nargs="*", default=[], help="more pilot dirs (e.g. a provider probe)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "main.toml")
    ap.add_argument("--budget", type=float, default=20.0)
    args = ap.parse_args()

    cfg = RunConfig.from_toml(args.config)
    df = pd.read_parquet(args.pilot / "decisions.parquet")
    for extra in args.add:
        other = pd.read_parquet(extra / "decisions.parquet")
        df = pd.concat([df[~df["arm"].isin(set(other["arm"]))], other], ignore_index=True)
    n_users = cfg.n_users
    throughput = sustained_throughput([args.pilot, *args.add])
    rows = []
    for spec in cfg.arms:
        d = df[df["arm"] == spec.arm]
        if d.empty:
            print(f"no pilot decisions for {spec.arm}", file=sys.stderr)
            return 1
        d = d.assign(calls=1 + d["retried"].astype(int))
        per = d.groupby(["kind", "checkpoint"])[
            ["cost", "calls", "prompt_tokens", "completion_tokens", "reasoning_tokens"]
        ].mean()
        total = pd.Series(0.0, index=per.columns)
        for cp in range(N_ROUNDS + 1):
            # checkpoint-0 held-out prompts carry no history, so they are identical for every user: paid once
            reps = N_HELDOUT * (1 if cp == 0 else n_users)
            total += per.loc[("heldout", cp)] * reps
            if cp < N_ROUNDS:
                total += per.loc[("dialogue", cp)] * n_users
        rate = throughput.get((spec.arm, spec.provider.lower()), float("nan"))
        hours = total["calls"] / rate / 3600
        rows.append(
            {
                "arm": spec.arm,
                "provider": spec.provider,
                "calls": total["calls"],
                "input_tokens_M": total["prompt_tokens"] / 1e6,
                "output_tokens_M": total["completion_tokens"] / 1e6,
                "thinking_tokens_M": total["reasoning_tokens"] / 1e6,
                "cost_usd": total["cost"],
                "pilot_calls_per_s": rate,
                "hours_if_alone": hours,
            }
        )
    est = pd.DataFrame(rows).set_index("arm")
    pd.set_option("display.width", 200)
    print(
        f"Projection for {args.config.name}: {n_users} users × {len(cfg.arms)} LLM arms, from {args.pilot.name} "
        f"({df['user_id'].nunique()} users)\n"
    )
    print(est.round(3).to_string())
    cost = float(est["cost_usd"].sum())
    by_provider = est.groupby("provider")["hours_if_alone"].sum()
    print(
        f"\nTotal: {est['calls'].sum():,.0f} calls, ${cost:.2f} (×1.25 margin: ${cost * 1.25:.2f}); "
        f"wall time ≈ {by_provider.max():.1f} h (arms sharing a provider run back to back on its rate limit: "
        f"{by_provider.round(1).to_dict()})"
    )
    ok = cost * 1.25 < args.budget
    print(f"GATE (projected main run with 25% margin under ${args.budget:.0f}): {'PASS' if ok else 'FAIL'}")
    (args.pilot / "analysis").mkdir(exist_ok=True)
    (args.pilot / "analysis" / "cost_projection.json").write_text(
        json.dumps(
            {
                "config": args.config.name,
                "arms": est.reset_index().to_dict("records"),
                "total_usd": cost,
                "throughput_calls_per_s": {f"{a}@{p}": v for (a, p), v in throughput.items()},
            },
            indent=1,
            default=float,
        )
        + "\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
