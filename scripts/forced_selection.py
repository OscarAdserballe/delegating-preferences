"""Is the forced-vs-unforced accuracy gap selection? Slice every arm by whether the thinking arm was forced on the
same (user, checkpoint, held-out menu), and compare gaps (difference-in-differences, CIs by resampling users)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THINK = "gemma4-31b-think"
ARMS = ("gemma4-31b-think", "gemma4-31b", "qwen25-7b", "bayes", "copy_last", "random")
KEY = ["user_id", "checkpoint", "menu_index"]


def gaps_by_user(df: pd.DataFrame, cps: list[int]) -> pd.DataFrame:
    """users × arm: unforced − forced accuracy within each checkpoint, then averaged over that user's checkpoints
    (forced items cluster at late checkpoints, so pooling across checkpoints would confound the gap)."""
    d = df[df["checkpoint"].isin(cps)]
    acc = d.groupby(["user_id", "checkpoint", "arm", "think_forced"])["correct"].mean().unstack("think_forced")
    gap = (acc[False] - acc[True]).dropna()
    return gap.groupby(["user_id", "arm"]).mean().unstack("arm")


def boot(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "main"
    df = pd.read_parquet(run_dir / "decisions.parquet")
    held = df[(df["kind"] == "heldout") & df["arm"].isin(ARMS)]
    forced = held[held["arm"] == THINK].set_index(KEY)["forced"].rename("think_forced")
    held = held.join(forced, on=KEY)

    print("Held-out accuracy (%) on items where the thinking arm was / wasn't forced (same user, checkpoint, menu)\n")
    tab = held.groupby(["checkpoint", "think_forced", "arm"])["correct"].mean().unstack("arm")[list(ARMS)] * 100
    share = held[held["arm"] == THINK].groupby("checkpoint")["think_forced"].mean() * 100
    print(tab.round(1).to_string())
    print(f"\nforced share by checkpoint (%): {share.round(1).to_dict()}")

    cps = [1, 2, 3, 4, 5]
    g = gaps_by_user(held, cps)
    print("\nGap = unforced − forced accuracy within checkpoint (1–5), mean over users [95% CI] (pts)")
    for arm in ARMS:
        m, lo, hi = boot(g[arm].to_numpy() * 100)
        print(f"  {arm:18s} {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]   (users: {g[arm].notna().sum()})")
    print(
        "\nDifference-in-differences: thinking's gap − another arm's gap (positive = forcing hurts thinking beyond "
        "what item difficulty explains)"
    )
    for ref in ARMS[1:]:
        m, lo, hi = boot((g[THINK] - g[ref]).to_numpy() * 100)
        print(f"  thinking − {ref:12s} {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
    rel = (
        held[held["arm"].isin([THINK, "gemma4-31b"])].pivot_table(
            index=["think_forced"], columns="arm", values="correct", aggfunc="mean"
        )
        * 100
    )
    print(f"\nThinking vs no-thinking on the same items, pooled checkpoints 0–5 (%):\n{rel.round(1)}")


if __name__ == "__main__":
    main()
