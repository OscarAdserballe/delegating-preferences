"""Learning curve of user 343 for the walkthrough."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT / "results" / "main" / "decisions.parquet")
h = df[(df["user_id"] == 343) & (df["kind"] == "heldout")].groupby(["arm", "checkpoint"])["correct"].mean().unstack() * 100
arms = {"bayes": ("Bayesian Assistant", "#2a78d6"), "gemma4-31b-think": ("Gemma 4 · thinking", "#6250d6"),
        "gemma4-31b": ("Gemma 4", "#1baf7a"), "qwen25-7b": ("Qwen 2.5 7B", "#eb6834")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#e3e7ea",
                     "axes.labelcolor": "#5b6773", "xtick.color": "#5b6773", "ytick.color": "#5b6773"})
fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=170, constrained_layout=True)
for a, (lab, c) in arms.items():
    y = h.loc[a].to_numpy()
    ax.plot(range(6), y, color=c, lw=2.2, marker="o", ms=4)
    ax.annotate(f"{lab} {y[-1]:.0f}%", (5.1, y[-1]), color=c, va="center", fontsize=8.5, fontweight="bold")
ax.set_xlim(-0.2, 7.4)
ax.set_ylim(0, 100)
ax.set_xticks(range(6))
ax.set_xlabel("Rounds of feedback seen")
ax.set_ylabel("Held-out accuracy (%)")
ax.grid(axis="y", color="#e3e7ea")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(Path(__file__).resolve().parent / "figures" / "4_user343.png")
