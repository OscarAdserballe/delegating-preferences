"""Accuracy, regret and agreement per checkpoint, with CIs from resampling users."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from flight_prefs.interaction_loop import DecisionRecord

N_BOOT = 10_000


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n_users: int


def to_frame(records: list[DecisionRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    df["pick"] = df["pick"].astype("Int64")
    df["zero_regret"] = df["regret"] <= 1e-9
    df["unparsed"] = df["pick"].isna()
    return df


def _user_means(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """users × (arm, checkpoint) table of per-user held-out means."""
    held = df[df["kind"] == "heldout"]
    return held.groupby(["user_id", "arm", "checkpoint"])[value].mean().unstack(["arm", "checkpoint"])


def bootstrap_users(per_user: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> Estimate:
    """Mean over users with a percentile 95% CI from resampling users."""
    x = np.asarray(per_user, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Estimate(float(x.mean()), float(lo), float(hi), len(x))


def by_checkpoint(df: pd.DataFrame, value: str, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """arm × checkpoint: mean of `value` over users, with a 95% CI from resampling users."""
    table = _user_means(df, value)
    rows = []
    for col in table.columns:
        arm, checkpoint = col
        est = bootstrap_users(table[col].dropna().to_numpy(), n_boot, seed)
        rows.append({"arm": arm, "checkpoint": checkpoint, **asdict(est)})
    return pd.DataFrame(rows).sort_values(["arm", "checkpoint"]).reset_index(drop=True)


def wide(table: pd.DataFrame) -> pd.DataFrame:
    """arm × checkpoint matrix of means from a by_checkpoint table."""
    return table.pivot(index="arm", columns="checkpoint", values="mean").astype(float)


def accuracy_by_checkpoint(df: pd.DataFrame, tie_aware: bool = False, n_boot: int = N_BOOT) -> pd.DataFrame:
    return by_checkpoint(df, "correct_tie_aware" if tie_aware else "correct", n_boot)


def regret_by_checkpoint(df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    return by_checkpoint(df, "regret", n_boot)


def paired_difference(
    df: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    checkpoint: int,
    value: str = "correct",
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Estimate:
    """a − b on the same users; resamples users, so between-user variation cancels."""
    table = _user_means(df, value)
    both = table[[(arm_a, checkpoint), (arm_b, checkpoint)]].dropna()
    return bootstrap_users((both.iloc[:, 0] - both.iloc[:, 1]).to_numpy(), n_boot, seed)


def learning_gain(
    df: pd.DataFrame, arm: str, from_cp: int = 1, to_cp: int = 5, value: str = "correct", n_boot: int = N_BOOT
) -> Estimate:
    """Within-arm change from checkpoint `from_cp` to `to_cp`, paired by user."""
    table = _user_means(df, value)
    both = table[[(arm, to_cp), (arm, from_cp)]].dropna()
    return bootstrap_users((both.iloc[:, 0] - both.iloc[:, 1]).to_numpy(), n_boot)


def agreement(df: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    """Per checkpoint: share of held-out menus where the two arms made the same pick."""
    held = df[df["kind"] == "heldout"]
    key = ["user_id", "checkpoint", "menu_index"]
    a = held[held["arm"] == arm_a].set_index(key)["pick"]
    b = held[held["arm"] == arm_b].set_index(key)["pick"]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    joined["same"] = (joined["a"] == joined["b"]).fillna(False).astype(bool)
    out = joined.groupby(level="checkpoint")["same"].mean().rename("agreement").reset_index()
    out.insert(0, "arm_b", arm_b)
    out.insert(0, "arm_a", arm_a)
    return out
