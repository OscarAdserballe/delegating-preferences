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


def provider_throughput(pilot_dir: Path) -> dict[str, float]:
    """Logged calls per second per provider over each provider's busiest stretch (from log timestamps)."""
    rows = [
        (e["usage"].get("provider"), e["time"])
        for p in (pilot_dir / "calls").glob("*/*.jsonl.gz")
        for e in _read_lines(p)
    ]
    df = pd.DataFrame(rows, columns=["provider", "time"]).dropna()
    out = {}
    for prov, g in df.groupby("provider"):
        per_min = g["time"].floordiv(60).value_counts()
        out[str(prov).lower()] = float(per_min.max() / 60)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=ROOT / "results" / "pilot")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "main.toml")
    ap.add_argument("--budget", type=float, default=20.0)
    args = ap.parse_args()

    cfg = RunConfig.from_toml(args.config)
    df = pd.read_parquet(args.pilot / "decisions.parquet")
    n_users = cfg.n_users
    throughput = provider_throughput(args.pilot)
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
        hours = total["calls"] / throughput.get(spec.provider.lower(), float("nan")) / 3600
        rows.append(
            {
                "arm": spec.arm,
                "provider": spec.provider,
                "calls": total["calls"],
                "input_tokens_M": total["prompt_tokens"] / 1e6,
                "output_tokens_M": total["completion_tokens"] / 1e6,
                "thinking_tokens_M": total["reasoning_tokens"] / 1e6,
                "cost_usd": total["cost"],
                "pilot_calls_per_s": throughput.get(spec.provider.lower()),
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
                "throughput_calls_per_s": throughput,
            },
            indent=1,
            default=float,
        )
        + "\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
