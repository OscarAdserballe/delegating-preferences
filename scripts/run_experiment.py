"""Run a config: every (arm, user) dialogue concurrently; resumable from the call log; --max-cost ceiling."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from flight_prefs.experiment import RunConfig, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", help="override run_id (results/<run_id>)")
    ap.add_argument("--max-cost", type=float, help="stop starting new calls once this many USD are spent")
    ap.add_argument("--n-users", type=int)
    ap.add_argument("--users", type=int, nargs="+", help="explicit user ids instead of the sampled prefix")
    ap.add_argument("--arms", nargs="+", help="only these arms (LLM or local)")
    ap.add_argument("--max-in-flight", type=int)
    ap.add_argument("--push-every", type=float, help="commit + push call logs every N seconds")
    ap.add_argument("--progress-every", type=float, default=30.0)
    args = ap.parse_args()

    cfg = RunConfig.from_toml(
        args.config,
        run_id=args.run_id,
        max_cost=args.max_cost,
        n_users=args.n_users,
        users=args.users,
        only_arms=args.arms,
        max_in_flight=args.max_in_flight,
        push_every_s=args.push_every,
    )
    run_dir = asyncio.run(run(cfg, progress_every=args.progress_every))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    keys = (
        "status",
        "decisions",
        "network_calls_this_session",
        "log_hits_this_session",
        "log_entries",
        "unique_keys",
        "duplicate_calls",
        "cost_this_session_usd",
        "cost_total_usd",
        "served",
        "concurrency",
    )
    print(json.dumps({k: manifest.get(k) for k in keys}, indent=1))


if __name__ == "__main__":
    main()
