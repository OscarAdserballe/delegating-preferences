"""Delegation figures: replication, unusual vs conventional principals, and net welfare under a cost per example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from flight_prefs import paper_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
ARMS = ["bayes", "gemma4-31b-think", "gemma4-31b", "qwen25-7b"]
LABEL = {
    "bayes": "Bayesian Assistant",
    "gemma4-31b-think": "Gemma 4 31B · thinking",
    "gemma4-31b": "Gemma 4 31B",
    "qwen25-7b": "Qwen 2.5 7B",
    "random": "Random",
}
COLOR = {
    "bayes": "#2a78d6",
    "gemma4-31b-think": "#6250d6",
    "gemma4-31b": "#1baf7a",
    "qwen25-7b": "#eb6834",
    "random": "#8a949e",
}
INK, MUTED, GRID = "#14202b", "#5b6773", "#e3e7ea"
COSTS = [0, 1, 2, 4, 8]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "figure.dpi": 160,
    }
)


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def boot(per_user: pd.DataFrame, n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = per_user.to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return x.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def main() -> None:
    data = paper_data.load(ROOT / "data" / "paper")
    df = pd.read_parquet(ROOT / "results" / "main" / "decisions.parquet")
    held = df[df["kind"] == "heldout"].copy()
    H = np.stack([m.phi for m in data.heldout])
    users = sorted(held["user_id"].unique())

    # principal type: unusual = wants pricier / longer / more stops on >= 2 of the 3; conventional = on none
    contra = {u: int((data.users[u].theta[1:] > 0).sum()) for u in users}
    group = {u: "unusual" if c >= 2 else "conventional" if c == 0 else "mixed" for u, c in contra.items()}
    held["group"] = held["user_id"].map(group)

    # welfare = 1 - regret / (best - worst utility on that menu); 100% = always the user's best flight
    span = {(u, i): float(np.ptp(H[i] @ data.users[u].theta)) for u in users for i in range(100)}
    held["span"] = [span[(u, i)] for u, i in zip(held["user_id"], held["menu_index"], strict=False)]
    held = held[held["span"] > 0]
    held["welfare"] = (1 - held["regret"] / held["span"]) * 100
    held["acc"] = held["correct"] * 100.0

    def curves(value: str, arms: list[str], grp: str | None = None) -> dict:
        out = {}
        h = held if grp is None else held[held["group"] == grp]
        for a in arms:
            t = h[h["arm"] == a].groupby(["user_id", "checkpoint"])[value].mean().unstack()
            m, lo, hi = boot(t)
            out[a] = {"mean": m.round(2).tolist(), "lo": lo.round(2).tolist(), "hi": hi.round(2).tolist(), "n": len(t)}
        return out

    summary: dict = {
        "groups": {g: sum(1 for u in users if group[u] == g) for g in ("conventional", "mixed", "unusual")}
    }
    x = np.arange(6)

    # Figure 1: replication + new models
    acc = curves("acc", [*ARMS, "random"])
    summary["accuracy_all"] = acc
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for a in [*ARMS, "random"]:
        c = acc[a]
        ax.fill_between(x, c["lo"], c["hi"], color=COLOR[a], alpha=0.13, lw=0)
        ax.plot(x, c["mean"], color=COLOR[a], lw=2.2, marker="o", ms=4, ls="--" if a == "random" else "-")
        ax.annotate(
            f"{LABEL[a]}  {c['mean'][-1]:.0f}%",
            (5.08, c["mean"][-1]),
            color=COLOR[a] if a != "random" else MUTED,
            va="center",
            fontsize=9.5,
            fontweight="bold" if a != "random" else "normal",
        )
    for y in (37, 80):
        ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (1, 3)))
    ax.set_xlim(-0.15, 6.9)
    ax.set_xticks(x)
    ax.set_ylim(25, 92)
    ax.set_xlabel(
        "Rounds of feedback seen\n\nDotted lines: the paper's final-round values "
        "(Bayesian Assistant ~80%; Qwen 2.5 7B and Gemma 2 9B 37%).\nBands: 95% CI over 50 users.",
        fontsize=9,
    )
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Replication holds; Gemma 4 learns, Qwen 2.5 does not")
    style(ax)
    fig.savefig(OUT / "1_replication.png")
    plt.close(fig)

    # Figure 2: unusual vs conventional principals (welfare)
    w = {g: curves("welfare", ARMS, g) for g in ("conventional", "unusual")}
    summary["welfare_by_group"] = w
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True, constrained_layout=True)
    for ax, g in zip(axs, ("conventional", "unusual"), strict=False):
        for a in ARMS:
            c = w[g][a]
            ax.fill_between(x, c["lo"], c["hi"], color=COLOR[a], alpha=0.13, lw=0)
            ax.plot(x, c["mean"], color=COLOR[a], lw=2.2, marker="o", ms=4, label=LABEL[a])
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (1, 3)))
        ax.text(5, 51.2, "random picking", color=MUTED, fontsize=8.5, ha="right")
        n = summary["groups"][g]
        ax.set_title(
            {"conventional": f"Conventional principals (n = {n})", "unusual": f"Unusual principals (n = {n})"}[g]
        )
        ax.set_xticks(x)
        ax.set_xlabel("Rounds of feedback seen")
        style(ax)
    axs[0].set_ylabel("Share of achievable welfare (%)")
    axs[0].set_ylim(25, 100)
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=4, fontsize=9.5)
    fig.suptitle(
        "The default serves conventional principals; unusual ones must teach the agent",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(OUT / "2_unusual_vs_conventional.png")
    plt.close(fig)

    # Figure 3: net welfare = W(n) - c*n, and the best number of examples
    best = {
        g: {a: [int(np.argmax(np.array(w[g][a]["mean"]) - c * x)) for c in COSTS] for a in ARMS}
        for g in ("conventional", "unusual")
    }
    summary["best_n"] = {"costs_pct_per_example": COSTS, **best}
    c_show = 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True, constrained_layout=True)
    for ax, g in zip(axs, ("conventional", "unusual"), strict=False):
        for a in ARMS:
            net = np.array(w[g][a]["mean"]) - c_show * x
            k = int(np.argmax(net))
            ax.plot(x, net, color=COLOR[a], lw=2.2, marker="o", ms=4, label=LABEL[a])
            ax.plot([k], [net[k]], marker="o", ms=11, mfc="none", mec=COLOR[a], mew=2)
        n = summary["groups"][g]
        ax.set_title(
            {"conventional": f"Conventional principals (n = {n})", "unusual": f"Unusual principals (n = {n})"}[g]
        )
        ax.set_xticks(x)
        ax.set_xlabel("Examples given (rounds of feedback)")
        style(ax)
    axs[0].set_ylabel(f"Net welfare (%), cost {c_show} pts per example")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=4, fontsize=9.5)
    fig.suptitle(
        "Net welfare peaks earlier for conventional principals (ringed = best number of examples)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(OUT / "3_net_welfare.png")
    plt.close(fig)


    # Figure 5: round 0 = the agent's default. Accuracy with no history, by how many contrarian tastes the user has
    r0 = held[held["checkpoint"] == 0].assign(contra=lambda d: d["user_id"].map(contra))
    tab = r0.groupby(["arm", "contra"])["acc"].mean().unstack()
    ncontra = pd.Series(contra).value_counts().sort_index()
    summary["round0_acc_by_contra"] = {a: tab.loc[a].round(1).tolist() for a in ARMS}
    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    bw = 0.2
    for j, a in enumerate(ARMS):
        ax.bar(np.arange(4) + (j - 1.5) * bw, tab.loc[a].to_numpy(), bw * 0.92, color=COLOR[a], label=LABEL[a])
    ax.axhline(100 / 3, color=MUTED, lw=0.8, ls=(0, (1, 3)))
    ax.text(3.45, 35, "random", color=MUTED, fontsize=8.5, ha="right")
    ax.set_xticks(range(4), [f"{k}\n(n = {ncontra.get(k, 0)})" for k in range(4)])
    ax.set_xlabel("Contrarian tastes the user has (wants pricier, longer or more stops)")
    ax.set_ylabel("Accuracy with no history (%)")
    ax.set_title("Before any feedback, the LLMs serve a cheap, short, direct traveller")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right", bbox_to_anchor=(1, 1.0))
    ax.set_ylim(0, 75)
    style(ax)
    fig.savefig(OUT / "5_round0_default.png")
    plt.close(fig)

    # Figure 6: the basic hump. Net welfare over all 50 users, LLMs vs the Bayesian learner
    wall = curves("welfare", ARMS)
    summary["welfare_all"] = wall
    summary["best_n_all"] = {a: [int(np.argmax(np.array(wall[a]["mean"]) - c * x)) for c in COSTS] for a in ARMS}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for a in ARMS:
        m = np.array(wall[a]["mean"])
        axs[0].plot(x, m, color=COLOR[a], lw=2.2, marker="o", ms=4, label=LABEL[a])
        net = m - c_show * x
        k = int(np.argmax(net))
        axs[1].plot(x, net, color=COLOR[a], lw=2.2, marker="o", ms=4)
        axs[1].plot([k], [net[k]], marker="o", ms=11, mfc="none", mec=COLOR[a], mew=2)
    axs[0].set_title("Welfare W(n), all 50 users")
    axs[0].set_ylabel("Share of achievable welfare (%)")
    axs[1].set_title(f"Net welfare W(n) − {c_show}n (ringed = best n)")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xlabel("Examples given (rounds of feedback)")
        style(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=4, fontsize=9.5)
    fig.savefig(OUT / "6_hump_all.png")
    plt.close(fig)

    # Figure 7: LLMs only, conventional (dashed) vs unusual (solid) principals, net welfare at c_show
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for a in ARMS[1:]:
        for g, ls in (("conventional", "--"), ("unusual", "-")):
            net = np.array(w[g][a]["mean"]) - c_show * x
            k = int(np.argmax(net))
            ax.plot(x, net, color=COLOR[a], lw=2.2, ls=ls, marker="o", ms=4)
            ax.plot([k], [net[k]], marker="o", ms=11, mfc="none", mec=COLOR[a], mew=2)
        ax.annotate(LABEL[a], (5.1, np.array(w["unusual"][a]["mean"])[-1] - c_show * 5), color=COLOR[a], fontsize=9, va="center", fontweight="bold")
    ax.plot([], [], color=MUTED, ls="--", label="conventional principals")
    ax.plot([], [], color=MUTED, ls="-", label="unusual principals")
    fig.legend(frameon=False, fontsize=9.5, loc="outside lower center", ncol=2)
    ax.set_xlim(-0.15, 6.6)
    ax.set_xticks(x)
    ax.set_xlabel("Examples given (rounds of feedback)")
    ax.set_ylabel(f"Net welfare (%), cost {c_show} pts per example")
    ax.set_title("Conventional principals should say nothing; unusual ones must teach")
    style(ax)
    fig.savefig(OUT / "7_llm_split.png")
    plt.close(fig)

    (Path(__file__).resolve().parent / "summary.json").write_text(json.dumps(summary, indent=1))
    for g in ("conventional", "unusual"):
        print(g, {a: (w[g][a]["mean"][0], w[g][a]["mean"][-1]) for a in ARMS})
    print("best n by cost", COSTS, json.dumps(best))


if __name__ == "__main__":
    main()
