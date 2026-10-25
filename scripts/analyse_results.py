"""Tables, figure and summary.json for one run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flight_prefs import figures, metrics
from flight_prefs.call_log import _read_lines

ROOT = Path(__file__).resolve().parents[1]
LLM_ORDER = ("qwen25-7b", "gemma4-31b", "gemma4-31b-think")
PAPER_QWEN_FINAL = 0.37


def call_stats(run_dirs: list[Path], budgets: dict[str, int | None]) -> pd.DataFrame:
    """Per arm, per network call (log entry): tokens, latency, finish reasons, how often the thinking cap binds."""
    rows = []
    for path in (p for d in run_dirs for p in (d / "calls").glob("*/*.jsonl.gz")):
        for e in _read_lines(path):
            u = e["usage"]
            rows.append(
                {
                    "arm": e["arm"],
                    "reminder": "Answer with exactly one sentence" in e["request"]["last_user"],
                    **{
                        k: u.get(k)
                        for k in (
                            "prompt_tokens",
                            "completion_tokens",
                            "reasoning_tokens",
                            "cost",
                            "latency_s",
                            "finish_reason",
                            "provider",
                        )
                    },
                    "text_chars": len(e.get("text") or ""),
                    "empty_text": not (e.get("text") or "").strip(),
                }
            )
    calls = pd.DataFrame(rows)
    if calls.empty:
        return calls
    out = []
    for (arm, provider), g in calls.groupby(["arm", "provider"]):
        budget = budgets.get(str(arm))
        r = g["reasoning_tokens"].astype(float)
        out.append(
            {
                "arm": arm,
                "provider": provider,
                "calls": len(g),
                "reminder_calls": int(g["reminder"].sum()),
                "prompt_tokens_mean": g["prompt_tokens"].mean(),
                "completion_tokens_mean": g["completion_tokens"].mean(),
                "reasoning_tokens_mean": r.mean(),
                "reasoning_tokens_p50": r.median(),
                "reasoning_tokens_p95": r.quantile(0.95),
                "reasoning_tokens_max": r.max(),
                "thinking_cap": budget,
                "cap_binds_share": float((r >= 0.98 * budget).mean()) if budget else None,
                "finish_length_share": float((g["finish_reason"] == "length").mean()),
                "empty_text_share": float(g["empty_text"].mean()),
                "latency_p50_s": g["latency_s"].median(),
                "latency_p95_s": g["latency_s"].quantile(0.95),
                "cost_usd": g["cost"].sum(),
                "providers": sorted(g["provider"].dropna().unique().tolist()),
            }
        )
    return pd.DataFrame(out)


def fmt_wide(t: pd.DataFrame, scale: float, digits: int) -> pd.DataFrame:
    w = metrics.wide(t) * scale
    lo = t.pivot(index="arm", columns="checkpoint", values="lo") * scale
    hi = t.pivot(index="arm", columns="checkpoint", values="hi") * scale
    return w.round(digits).astype(str) + " [" + lo.round(digits).astype(str) + ", " + hi.round(digits).astype(str) + "]"


def est(e: metrics.Estimate, scale: float = 100) -> dict[str, float]:
    return {
        "mean": round(e.mean * scale, 3),
        "lo": round(e.lo * scale, 3),
        "hi": round(e.hi * scale, 3),
        "n_users": e.n_users,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="results/<run_id>; outputs go here")
    ap.add_argument("--add", type=Path, nargs="*", default=[], help="more run dirs whose arms are added (same users)")
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args()
    run_dir: Path = args.run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    df = pd.read_parquet(run_dir / "decisions.parquet")
    for extra in args.add:
        other = pd.read_parquet(extra / "decisions.parquet")
        if set(other["user_id"]) != set(df["user_id"]):
            raise SystemExit(f"{extra} has different users")
        df = pd.concat([df[~df["arm"].isin(set(other["arm"]))], other], ignore_index=True)
        m = json.loads((extra / "manifest.json").read_text())
        manifest["models"].update(m["models"])
        manifest.setdefault("added_runs", []).append(
            {"run_id": m["run_id"], "served": m.get("served"), "cost_total_usd": m.get("cost_total_usd")}
        )
    arms = set(df["arm"])
    llm_arms = [a for a in LLM_ORDER if a in arms] + sorted(a for a in manifest["models"] if a not in LLM_ORDER)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    acc = metrics.accuracy_by_checkpoint(df, n_boot=args.n_boot)
    acc_tie = metrics.accuracy_by_checkpoint(df, tie_aware=True, n_boot=args.n_boot)
    reg = metrics.regret_by_checkpoint(df, n_boot=args.n_boot)
    zero = metrics.by_checkpoint(df, "zero_regret", args.n_boot)
    for name, t in (("accuracy", acc), ("accuracy_tie_aware", acc_tie), ("regret", reg), ("zero_regret_share", zero)):
        t.to_csv(out / f"{name}.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    n_users = df["user_id"].nunique()
    print(f"run {manifest['run_id']}: {n_users} users, arms {sorted(arms)}, status {manifest['status']}")
    print(f"\nHeld-out accuracy (%), mean [95% CI over users]\n{fmt_wide(acc, 100, 1)}")
    print(f"\nTie-aware accuracy (%)\n{(metrics.wide(acc_tie) * 100).round(1)}")
    print(f"\nMean utility regret (θᵀφ)\n{fmt_wide(reg, 1, 3)}")
    print(f"\nShare of picks with zero regret (%)\n{(metrics.wide(zero) * 100).round(1)}")

    random_regret = float(metrics.wide(reg).loc["random"].mean()) if "random" in arms else None
    summary: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "n_users": n_users,
        "status": manifest["status"],
        "accuracy": acc.to_dict("records"),
        "accuracy_tie_aware": acc_tie.to_dict("records"),
        "regret": reg.to_dict("records"),
        "zero_regret_share": zero.to_dict("records"),
        "random_regret_mean": random_regret,
        "comparisons": {},
        "diagnostics": {},
    }

    print("\nComparisons at checkpoint 5 (paired by user; pts for accuracy, θᵀφ for regret)")
    pairs = [(a, "bayes") for a in llm_arms if "bayes" in arms]
    if {"gemma4-31b-think", "gemma4-31b"} <= arms:
        pairs.append(("gemma4-31b-think", "gemma4-31b"))
    for a, b in pairs:
        d_acc = metrics.paired_difference(df, a, b, 5, n_boot=args.n_boot)
        d_reg = metrics.paired_difference(df, a, b, 5, value="regret", n_boot=args.n_boot)
        summary["comparisons"][f"{a} - {b} @cp5"] = {"accuracy_pts": est(d_acc), "regret": est(d_reg, 1)}
        print(
            f"  {a} − {b}: accuracy {d_acc.mean * 100:+.1f} [{d_acc.lo * 100:+.1f}, {d_acc.hi * 100:+.1f}] · "
            f"regret {d_reg.mean:+.3f} [{d_reg.lo:+.3f}, {d_reg.hi:+.3f}]"
        )
    print("\nLearning: checkpoint 5 − checkpoint 1, and 1 − 0 (paired by user, pts)")
    for a in [*llm_arms, "bayes", "copy_last"]:
        if a in arms:
            g51, g10 = (
                metrics.learning_gain(df, a, 1, 5, n_boot=args.n_boot),
                metrics.learning_gain(df, a, 0, 1, n_boot=args.n_boot),
            )
            summary["comparisons"][f"{a} gain cp1->cp5"] = est(g51)
            summary["comparisons"][f"{a} gain cp0->cp1"] = est(g10)
            print(
                f"  {a:18s} 1→5 {g51.mean * 100:+.1f} [{g51.lo * 100:+.1f}, {g51.hi * 100:+.1f}]   "
                f"0→1 {g10.mean * 100:+.1f} [{g10.lo * 100:+.1f}, {g10.hi * 100:+.1f}]"
            )
    if "qwen25-7b" in arms:
        q5 = metrics.bootstrap_users(
            metrics._user_means(df, "correct")[("qwen25-7b", 5)].dropna().to_numpy(), args.n_boot
        )
        summary["comparisons"]["qwen25-7b cp5 vs paper 37%"] = {**est(q5), "paper": 37.0}
        print(
            f"\nQwen 2.5 7B checkpoint-5 accuracy {q5.mean * 100:.1f}% [{q5.lo * 100:.1f}, {q5.hi * 100:.1f}] "
            f"vs paper 37% (replication band ±3 pts)"
        )

    print("\nDiagnostics: held-out agreement with reference agents, mean over checkpoints 1–5 (%)")
    agree_rows = []
    for a in llm_arms:
        for ref in ("bayes", "always_first", "copy_last"):
            if ref in arms:
                ag = metrics.agreement(df, a, ref)
                agree_rows.append(ag)
                print(
                    f"  {a:18s} vs {ref:13s} {ag[ag['checkpoint'] >= 1]['agreement'].mean() * 100:5.1f}   "
                    f"by checkpoint {(ag.set_index('checkpoint')['agreement'] * 100).round(0).to_dict()}"
                )
    if agree_rows:
        pd.concat(agree_rows).to_csv(out / "agreement.csv", index=False)
        summary["diagnostics"]["agreement"] = pd.concat(agree_rows).to_dict("records")

    llm = df[df["arm"].isin(llm_arms)]
    if not llm.empty:
        picks = llm[llm["kind"] == "heldout"].groupby("arm")["pick"].value_counts(normalize=True).unstack()
        oracle = df[(df["arm"] == "oracle") & (df["kind"] == "heldout")]["pick"]
        best_pos = (oracle.value_counts(normalize=True).sort_index() * 100).round(1).tolist()
        print(f"\nPick position shares, held-out (%; the user's choice is Flight 1/2/3 in {best_pos}%)")
        print((picks * 100).round(1))
        per = llm.groupby("arm").agg(
            decisions=("pick", "size"),
            unparsed=("unparsed", "sum"),
            retried=("retried", "mean"),
            cost_usd=("cost", "sum"),
        )
        per["retried"] = (per["retried"] * 100).round(1)
        print(f"\nPer arm: decisions, unparsed (scored wrong), % needing the format reminder, cost\n{per}")
        retry_cp = (llm.pivot_table(index="arm", columns="checkpoint", values="retried", aggfunc="mean") * 100).round(1)
        print(f"\n% of decisions needing the format reminder, by checkpoint\n{retry_cp}")
        unparsed = llm[llm["unparsed"]]
        if len(unparsed):
            print("\nUnparsed after the retry (scored wrong), by arm × user × checkpoint:")
            print(
                unparsed.pivot_table(
                    index=["arm", "user_id"], columns="checkpoint", values="pick", aggfunc="size", fill_value=0
                )
            )
            print("raw replies:", unparsed["raw_text"].fillna("<none>").str[:60].value_counts().head(5).to_dict())
        summary["diagnostics"]["per_arm"] = per.reset_index().to_dict("records")
        summary["diagnostics"]["retry_by_checkpoint"] = {
            str(a): {str(c): float(v) for c, v in row.items()} for a, row in retry_cp.iterrows()
        }
        summary["diagnostics"]["pick_shares"] = {
            str(a): {str(k): float(v) for k, v in row.items()} for a, row in picks.fillna(0).iterrows()
        }
        summary["diagnostics"]["unparsed"] = unparsed[
            ["arm", "user_id", "checkpoint", "kind", "menu_index", "raw_text"]
        ].to_dict("records")

    budgets = {a: m["request_params"].get("reasoning", {}).get("max_tokens") for a, m in manifest["models"].items()}
    calls = call_stats([run_dir, *args.add], budgets)
    if not calls.empty:
        calls.to_csv(out / "call_stats.csv", index=False)
        table = calls.drop(columns=["providers"]).round(3).T
        print(f"\nPer network call (all logged calls), by arm and served provider\n{table}")
        summary["diagnostics"]["calls"] = calls.replace({np.nan: None}).to_dict("records")

    fig = figures.results_figure(acc, reg, out / "results.png", f"{manifest['run_id']}: {n_users} users")
    summary["manifest"] = {
        k: manifest.get(k)
        for k in (
            "served",
            "cost_total_usd",
            "git",
            "started",
            "finished",
            "concurrency",
            "duplicate_calls",
            "added_runs",
        )
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"\nwrote {out}/*.csv, {fig}, {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
