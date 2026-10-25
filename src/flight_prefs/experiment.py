"""Config → every (arm, user) dialogue at once → decisions.parquet, manifest.json, call logs."""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import time
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from flight_prefs import metrics, paper_data, user_sampling
from flight_prefs.agents.base import Agent
from flight_prefs.agents.bayesian_assistant import BayesianAssistant
from flight_prefs.agents.llm import LLMAgent
from flight_prefs.agents.simulated import AlwaysFirstAgent, CopyLastChoiceAgent, OracleAgent, RandomAgent
from flight_prefs.call_log import CachedClient, CallLog
from flight_prefs.interaction_loop import DecisionRecord, run_user
from flight_prefs.model_client import ModelSpec, OpenRouterClient

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ARMS = ("bayes", "random", "oracle", "always_first", "copy_last")


@dataclass
class RunConfig:
    run_id: str
    n_users: int
    arms: list[ModelSpec]
    local_arms: list[str] = field(default_factory=lambda: list(LOCAL_ARMS))
    users: list[int] | None = None  # explicit override of the sampled prefix
    user_sample: str = "data/user_sample.json"
    paper_dir: str = "data/paper"
    results_dir: str = "results"
    max_in_flight: int = 64
    max_cost: float | None = None
    push_every_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path, **overrides: Any) -> RunConfig:
        raw = tomllib.loads(path.read_text())
        raw.update({k: v for k, v in overrides.items() if v is not None})
        arms = [ModelSpec(**a) for a in raw.pop("arms", [])]
        if "only_arms" in raw:
            keep = set(raw.pop("only_arms"))
            arms = [a for a in arms if a.arm in keep]
            raw["local_arms"] = [a for a in raw.get("local_arms", LOCAL_ARMS) if a in keep]
        cfg = cls(arms=arms, **raw)
        cfg.raw = {**raw, "arms": [asdict(a) for a in arms]}
        return cfg

    @property
    def run_dir(self) -> Path:
        return ROOT / self.results_dir / self.run_id

    def user_ids(self) -> list[int]:
        return self.users if self.users is not None else user_sampling.load(ROOT / self.user_sample, self.n_users)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def git_state() -> dict[str, Any]:
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


async def push_calls(run_dir: Path, message: str) -> str:
    """Commit and push the run's call logs so a fresh session can resume without paying twice."""
    rel = run_dir.relative_to(Path(_git("rev-parse", "--show-toplevel")))
    script = (
        f'git add "{rel}" && (git diff --cached --quiet || git commit -q -m "{message}") && '
        "for i in 2 4 8 16; do git push -q origin HEAD && exit 0; sleep $i; done; exit 1"
    )
    proc = await asyncio.create_subprocess_shell(
        script,
        cwd=_git("rev-parse", "--show-toplevel"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return f"exit {proc.returncode} {out.decode().strip()[-300:]}"


def log_stats(run_dir: Path) -> dict[str, int]:
    """Lines vs unique keys over all call logs: any difference is a duplicate (paid-twice) call."""
    from flight_prefs.call_log import _read_lines

    lines, keys = 0, set()
    for path in (run_dir / "calls").glob("*/*.jsonl.gz"):
        for e in _read_lines(path):
            lines += 1
            keys.add(e["key"])
    return {"log_entries": lines, "unique_keys": len(keys), "duplicate_calls": lines - len(keys)}


class Progress:
    def __init__(self, run_dir: Path, client: OpenRouterClient, clients: list[CachedClient], total: int):
        self.run_dir, self.client, self.clients, self.total = run_dir, client, clients, total
        self.done = 0
        self.t0 = time.monotonic()

    def line(self) -> str:
        net = sum(c.network_calls for c in self.clients)
        hits = sum(c.hits for c in self.clients)
        lim = " ".join(
            f"{p}:{int(lim.limit)}/{lim.in_flight} 429s={lim.rate_limited}" for p, lim in self.client.limits.items()
        )
        return (
            f"[{time.monotonic() - self.t0:6.0f}s] dialogues {self.done}/{self.total} · network {net} · "
            f"log hits {hits} · ${self.client.spend.total:.4f} · limit/in-flight {lim}"
        )

    async def report(self, every: float) -> None:
        while True:
            await asyncio.sleep(every)
            print(self.line(), file=sys.stderr, flush=True)


def build_jobs(
    cfg: RunConfig, data: paper_data.PaperData, client: OpenRouterClient, log: CallLog
) -> tuple[list[tuple[str, int, Callable[[], Awaitable[list[DecisionRecord]]]]], list[CachedClient]]:
    shared: dict[str, Agent] = {
        "bayes": BayesianAssistant(data.thetas),
        "random": RandomAgent(),
        "always_first": AlwaysFirstAgent(),
        "copy_last": CopyLastChoiceAgent(),
    }
    jobs: list[tuple[str, int, Callable[[], Awaitable[list[DecisionRecord]]]]] = []
    clients: list[CachedClient] = []
    for uid in cfg.user_ids():
        user = data.users[uid]
        for spec in cfg.arms:
            cc = CachedClient(client, log, spec.arm, uid)
            clients.append(cc)
            agent: Agent = LLMAgent(spec, cc)
            jobs.append((spec.arm, uid, partial(run_user, agent, user, data.heldout)))
        for name in cfg.local_arms:
            local = OracleAgent(user, data.heldout) if name == "oracle" else shared[name]
            jobs.append((name, uid, partial(run_user, local, user, data.heldout)))
    return jobs, clients


async def run(cfg: RunConfig, progress_every: float = 30.0) -> Path:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    data = paper_data.load(ROOT / cfg.paper_dir)
    log = CallLog(run_dir)
    client = OpenRouterClient(max_in_flight=cfg.max_in_flight, max_cost=cfg.max_cost)
    jobs, clients = build_jobs(cfg, data, client, log)
    manifest: dict[str, Any] = {
        "run_id": cfg.run_id,
        "config": cfg.raw,
        "git": git_state(),
        "users": cfg.user_ids(),
        "models": {
            a.arm: {
                "model_id": a.model_id,
                "provider": a.provider,
                "quantization": a.quantization,
                "request_params": a.request_params(),
            }
            for a in cfg.arms
        },
        "python": platform.python_version(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "log_entries_at_start": log.loaded,
        "status": "running",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"run {cfg.run_id}: {len(jobs)} dialogues, {len(cfg.arms)} LLM arms, {log.loaded} logged calls to reuse",
        file=sys.stderr,
        flush=True,
    )

    progress = Progress(run_dir, client, clients, len(jobs))
    background: list[asyncio.Task[Any]] = [asyncio.create_task(progress.report(progress_every))]
    if cfg.push_every_s:

        async def pusher() -> None:
            while True:
                await asyncio.sleep(cfg.push_every_s or 300)
                status = await push_calls(run_dir, f"{cfg.run_id}: checkpoint call logs ({progress.line()[:60]})")
                print(f"push: {status}", file=sys.stderr, flush=True)

        background.append(asyncio.create_task(pusher()))

    async def one(job: Callable[[], Awaitable[list[DecisionRecord]]]) -> list[DecisionRecord]:
        recs = await job()
        progress.done += 1
        return recs

    error: BaseException | None = None
    records: list[DecisionRecord] = []
    tasks = [asyncio.create_task(one(job)) for _, _, job in jobs]
    try:
        for recs in await asyncio.gather(*tasks):
            records += recs
    except BaseException as e:
        error = e
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for t in background:
            t.cancel()
        await client.aclose()

    stats = log_stats(run_dir)
    manifest.update(
        {
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "complete" if error is None else f"aborted: {type(error).__name__}: {error}",
            "served": {arm: {"provider": p, "model": m} for arm, (p, m) in client.served.items()},
            "cost_this_session_usd": round(client.spend.total, 6),
            "network_calls_this_session": sum(c.network_calls for c in clients),
            "log_hits_this_session": sum(c.hits for c in clients),
            "concurrency": {
                p: {
                    "final_limit": lim.limit,
                    "peak_in_flight": lim.peak,
                    "halvings_on_429": lim.halvings,
                    "rate_limited_responses": lim.rate_limited,
                }
                for p, lim in client.limits.items()
            },
            "errors": dict(client.errors),
            **stats,
        }
    )
    if records:
        df = metrics.to_frame(records)
        df.to_parquet(run_dir / "decisions.parquet", index=False)
        manifest["cost_total_usd"] = round(float(df["cost"].sum()), 6)
        manifest["decisions"] = len(df)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(progress.line(), file=sys.stderr, flush=True)
    if cfg.push_every_s:
        print(f"push: {await push_calls(run_dir, f'{cfg.run_id}: final call logs and manifest')}", file=sys.stderr)
    if error is not None:
        raise error
    return run_dir
