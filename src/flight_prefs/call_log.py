"""Every model call goes through here: a log hit never touches the network, so a resumed run never pays twice."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flight_prefs.agents.base import Usage
from flight_prefs.model_client import Completion, ModelSpec, OpenRouterClient


def call_key(spec: ModelSpec, messages: list[dict[str, str]]) -> str:
    """sha256 of the canonical JSON of (model, provider, messages, params); the arm name is not part of it."""
    payload = {**spec.request_params(), "messages": messages}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CallLog:
    """calls/<arm>/<user>.jsonl.gz, one gzip member per line, so appends never rewrite earlier bytes."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir / "calls"
        self.entries: dict[str, dict[str, Any]] = {}
        self.loaded = 0
        for path in sorted(self.dir.glob("*/*.jsonl.gz")):
            for line in _read_lines(path):
                self.entries[line["key"]] = line
                self.loaded += 1

    def get(self, key: str) -> Completion | None:
        e = self.entries.get(key)
        if e is None:
            return None
        usage = Usage(**{**e["usage"], "cached": True})
        return Completion(text=e["text"], thinking=e.get("thinking"), usage=usage, generation_id=e.get("generation_id"))

    def append(
        self, arm: str, user_id: int, key: str, c: Completion, spec: ModelSpec, messages: list[dict[str, str]]
    ) -> None:
        entry = {
            "key": key,
            "arm": arm,
            "user_id": user_id,
            "time": time.time(),
            "request": {**spec.request_params(), "n_messages": len(messages), "last_user": messages[-1]["content"]},
            "text": c.text,
            "thinking": c.thinking,
            "usage": asdict(c.usage),
            "generation_id": c.generation_id,
        }
        path = self.dir / arm / f"{user_id}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "ab") as fh:
            fh.write((json.dumps(entry, ensure_ascii=False) + "\n").encode())
        self.entries[key] = entry


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """Reads every complete line; a torn final gzip member (process killed mid-write) is skipped."""
    out: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except (EOFError, gzip.BadGzipFile, OSError):
        pass
    return out


class CachedClient:
    """What an LLMAgent holds: one per (arm, user)."""

    def __init__(self, inner: OpenRouterClient, log: CallLog, arm: str, user_id: int):
        self.inner, self.log, self.arm, self.user_id = inner, log, arm, user_id
        self.hits = 0
        self.network_calls = 0
        self._in_flight: dict[str, asyncio.Future[Completion]] = {}

    async def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> Completion:
        key = call_key(spec, messages)
        hit = self.log.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        if key in self._in_flight:
            return await asyncio.shield(self._in_flight[key])
        fut: asyncio.Future[Completion] = asyncio.get_running_loop().create_future()
        self._in_flight[key] = fut
        try:
            self.network_calls += 1
            c = await self.inner.complete(spec, messages)
            self.log.append(self.arm, self.user_id, key, c, spec, messages)
            fut.set_result(c)
            return c
        except BaseException as e:
            if not fut.done():
                fut.set_exception(e)
                fut.exception()  # mark retrieved so an unawaited future doesn't warn
            raise
        finally:
            del self._in_flight[key]
