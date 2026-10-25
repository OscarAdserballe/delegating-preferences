"""Draw the seeded, stratified user order once and commit it (data/user_sample.json)."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from flight_prefs import paper_data, user_sampling

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261025


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "user_sample.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; the sample is drawn once (use --force only before any API call)")
    data = paper_data.load(ROOT / "data" / "paper")
    order = user_sampling.stratified_order(data.users, SEED)
    user_sampling.save(order, data.users, SEED, args.out)
    for n in (2, 50):
        counts = collections.Counter(user_sampling.stratum(data.users[u].theta) for u in order[:n])
        print(f"first {n}: {order[:n] if n <= 2 else '…'}  strata {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
