"""OpenRouter chat completions: pinned provider, no fallbacks, retries, adaptive concurrency, cost tracking."""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from flight_prefs.agents.base import Usage

URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


@dataclass(frozen=True)
class ModelSpec:
    arm: str  # "qwen25-7b", "gemma4-31b", "gemma4-31b-think"
    model_id: str  # "google/gemma-4-31b-it"
    provider: str  # pinned, e.g. "crusoe"
    quantization: str | None = None  # e.g. "bf16"; None = don't filter (provider states none)
    temperature: float = 0.0
    max_tokens: int = 16
    thinking: bool | None = None  # None = don't send a reasoning parameter at all
    thinking_budget: int | None = None  # reasoning.max_tokens when thinking (a hint providers may ignore)
    force_answer: bool = False  # cut off at max_tokens without an answer → one no-thinking call to answer now

    def request_params(self) -> dict[str, Any]:
        """Everything except messages that goes into the request body (and the call key); force_answer is
        agent behaviour, not a request parameter."""
        provider: dict[str, Any] = {"only": [self.provider], "allow_fallbacks": False}
        if self.quantization:
            provider["quantizations"] = [self.quantization]
        params: dict[str, Any] = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": provider,
            "usage": {"include": True},
        }
        if self.thinking is not None:
            reasoning: dict[str, Any] = {"enabled": self.thinking}
            if self.thinking and self.thinking_budget:
                reasoning = {"max_tokens": self.thinking_budget}
            params["reasoning"] = reasoning
        return params


@dataclass(frozen=True)
class Completion:
    text: str
    thinking: str | None
    usage: Usage
    generation_id: str | None = None


class CostCeilingReached(RuntimeError):
    pass


class ServedModelChanged(RuntimeError):
    pass


class RequestFailed(RuntimeError):
    pass


class OutOfCredits(RequestFailed):
    """OpenRouter 402: top up, then rerun the same config; logged calls are not paid again."""


@dataclass
class AdaptiveLimit:
    """AIMD concurrency: halve on a 429 (at most once per `cooldown_s`), +1 after `limit` successes in a row.

    Waiters queue FIFO and are woken one per free slot, so tens of thousands of queued calls cost nothing.
    """

    limit: float = 64
    floor: int = 8  # 429s are free, and a shared upstream pool recovers in bursts
    ceiling: int = 512
    cooldown_s: float = 2.0
    in_flight: int = 0
    successes: int = 0
    peak: int = 0
    halvings: int = 0
    rate_limited: int = 0
    last_halving: float = -1e9
    _waiters: deque[asyncio.Future[None]] = field(default_factory=deque)

    def _wake(self) -> None:
        while self._waiters and self.in_flight < int(self.limit):
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_flight += 1
                fut.set_result(None)

    async def __aenter__(self) -> AdaptiveLimit:
        if self.in_flight < int(self.limit) and not self._waiters:
            self.in_flight += 1
        else:
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await fut
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():
                    self.in_flight -= 1  # granted a slot just as we were cancelled
                    self._wake()
                raise
        self.peak = max(self.peak, self.in_flight)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.in_flight -= 1
        self._wake()

    async def on_success(self) -> None:
        self.successes += 1
        if self.successes >= self.limit:
            self.limit = min(self.ceiling, self.limit + 1)
            self.successes = 0
            self._wake()

    async def on_rate_limited(self) -> None:
        """A burst of 429s from requests already in flight is one signal, not many."""
        self.rate_limited += 1
        self.successes = 0
        now = time.monotonic()
        if now - self.last_halving >= self.cooldown_s:
            self.limit = max(self.floor, self.limit / 2)
            self.halvings += 1
            self.last_halving = now


@dataclass
class Spend:
    total: float = 0.0
    calls: int = 0
    max_cost: float | None = None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_in_flight: int = 64,
        max_cost: float | None = None,
        max_attempts: int = 8,
        timeout_s: float = 300.0,
        rate_limit_patience_s: float = 1800.0,
    ):
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.start_limit = max_in_flight
        self.limits: dict[str, AdaptiveLimit] = {}
        self.spend = Spend(max_cost=max_cost)
        self.max_attempts = max_attempts
        self.rate_limit_patience_s = rate_limit_patience_s
        self.errors: Counter[str] = Counter()
        self.served: dict[str, tuple[str, str]] = {}  # arm → (provider, model) first served
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
            headers={"Authorization": f"Bearer {self.api_key}", "X-Title": "bayesian-teaching-replication"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def limit_for(self, provider: str) -> AdaptiveLimit:
        if provider not in self.limits:
            self.limits[provider] = AdaptiveLimit(limit=self.start_limit)
        return self.limits[provider]

    def _check_served(self, spec: ModelSpec, provider: str, model: str) -> None:
        first = self.served.setdefault(spec.arm, (provider, model))
        if first != (provider, model):
            raise ServedModelChanged(f"{spec.arm}: served {provider}/{model}, earlier {first[0]}/{first[1]}")
        if provider.lower().split("/")[0] != spec.provider.lower():
            raise ServedModelChanged(f"{spec.arm}: served by {provider}, pinned {spec.provider}")

    async def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> Completion:
        if self.spend.max_cost is not None and self.spend.total >= self.spend.max_cost:
            raise CostCeilingReached(f"spent ${self.spend.total:.4f} ≥ --max-cost ${self.spend.max_cost:.2f}")
        body = {**spec.request_params(), "messages": messages}
        limit = self.limit_for(spec.provider)
        last_error = ""
        failures, rate_limited = 0, 0
        first_429: float | None = None  # patience counts from the first 429, not from time spent queued
        while True:
            async with limit:
                t0 = time.monotonic()
                try:
                    resp: httpx.Response | None = await self._http.post(URL, json=body)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = f"{type(e).__name__}: {e}"
                    resp = None
                latency = time.monotonic() - t0
            is_429 = False
            if resp is not None:
                data = _json(resp)
                error = data.get("error")
                if resp.status_code == 200 and not error and data.get("choices"):
                    completion = self._parse(spec, data, latency)
                    await limit.on_success()
                    return completion
                code = error.get("code", resp.status_code) if isinstance(error, dict) else resp.status_code
                last_error = f"HTTP {resp.status_code} {str(error or resp.text)[:300]}"
                is_429 = resp.status_code == 429 or code == 429
                if resp.status_code == 402 or code == 402:
                    self.errors["402 out of credits"] += 1
                    raise OutOfCredits(f"{spec.arm}: {last_error}")
                if not is_429 and resp.status_code not in RETRYABLE_STATUS and resp.status_code != 200:
                    self.errors[f"{spec.arm} fatal {resp.status_code}"] += 1
                    raise RequestFailed(f"{spec.arm}: {last_error}")
            if is_429:
                self.errors[f"{spec.provider} 429"] += 1
                await limit.on_rate_limited()
                rate_limited += 1
                first_429 = first_429 or time.monotonic()
                if time.monotonic() - first_429 > self.rate_limit_patience_s:
                    raise RequestFailed(f"{spec.arm}: still rate-limited after {self.rate_limit_patience_s:.0f}s")
                await asyncio.sleep(min(30.0, 2 ** min(rate_limited, 5)) * (0.5 + random.random()))
                continue
            failures += 1
            self.errors[f"{spec.arm} {last_error[:40]}"] += 1
            if failures >= self.max_attempts:
                raise RequestFailed(f"{spec.arm}: gave up after {failures} failures; last: {last_error}")
            await asyncio.sleep(min(60.0, 2**failures) * (0.5 + random.random()))

    def _parse(self, spec: ModelSpec, data: dict[str, Any], latency: float) -> Completion:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        u = data.get("usage") or {}
        provider, model = str(data.get("provider")), str(data.get("model"))
        self._check_served(spec, provider, model)
        cost = float(u.get("cost") or 0.0)
        self.spend.total += cost
        self.spend.calls += 1
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            cost=cost,
            provider=provider,
            model=model,
            finish_reason=choice.get("finish_reason"),
            latency_s=round(latency, 3),
        )
        return Completion(
            text=message.get("content") or "",
            thinking=message.get("reasoning"),
            usage=usage,
            generation_id=data.get("id"),
        )


def _json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
