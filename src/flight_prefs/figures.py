"""Results figure: accuracy and regret by checkpoint, one line per arm, 95% CI bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Color follows the arm, never its rank; reference agents are recessive gray.
ARM_STYLE: dict[str, dict[str, str]] = {
    "bayes": {"color": "#2a78d6", "label": "Bayesian Assistant"},
    "qwen25-7b": {"color": "#eb6834", "label": "Qwen 2.5 7B"},
    "gemma4-31b": {"color": "#1baf7a", "label": "Gemma 4 31B"},
    "gemma4-31b-think": {"color": "#6250d6", "label": "Gemma 4 31B · thinking"},
    "oracle": {"color": "#8a8984", "label": "Oracle", "ls": ":"},
    "random": {"color": "#8a8984", "label": "Random", "ls": "--"},
    "always_first": {"color": "#b5b4ae", "label": "Always Flight 1", "ls": "--"},
    "copy_last": {"color": "#52514e", "label": "Copy last choice", "ls": "-."},
}
PAPER_QWEN_FINAL = 37.0
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _panel(ax: plt.Axes, table: pd.DataFrame, scale: float, ylabel: str, title: str) -> None:
    for arm, g in table.groupby("arm"):
        style = ARM_STYLE.get(str(arm), {"color": MUTED, "label": str(arm)})
        g = g.sort_values("checkpoint")
        ax.fill_between(g["checkpoint"], g["lo"] * scale, g["hi"] * scale, color=style["color"], alpha=0.15, lw=0)
        ax.plot(
            g["checkpoint"],
            g["mean"] * scale,
            color=style["color"],
            lw=2,
            ls=style.get("ls", "-"),
            marker="o",
            ms=4,
            label=style["label"],
        )
    ax.set_xticks(range(6))
    ax.set_xlabel("Rounds of feedback seen (checkpoint)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def results_figure(accuracy: pd.DataFrame, regret: pd.DataFrame, path: Path, title: str = "") -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    _panel(a, accuracy, 100, "Held-out accuracy (%)", "Accuracy (paper's labels)")
    a.axhline(PAPER_QWEN_FINAL, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    a.annotate("paper: Qwen 2.5 7B, final round", (5, PAPER_QWEN_FINAL - 4), color=MUTED, fontsize=8, ha="right")
    a.set_ylim(0, 102)
    _panel(b, regret, 1, "Mean utility regret (θᵀφ units)", "Utility regret")
    b.set_ylim(bottom=0)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False, labelcolor=INK)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
