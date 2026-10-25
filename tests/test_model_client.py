from __future__ import annotations

import asyncio

import httpx
import pytest

from flight_prefs.model_client import ModelSpec, OpenRouterClient, OutOfCredits

SPEC = ModelSpec(arm="a", model_id="m", provider="p")


def client_with(handler: httpx.MockTransport) -> OpenRouterClient:
    c = OpenRouterClient(api_key="x")
    c._http = httpx.AsyncClient(transport=handler)
    return c


def test_out_of_credits_stops_without_retrying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})

    with pytest.raises(OutOfCredits):
        asyncio.run(client_with(httpx.MockTransport(handler)).complete(SPEC, [{"role": "user", "content": "x"}]))
    assert calls == 1


def test_rate_limit_then_success() -> None:
    replies = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(replies)
        if status == 429:
            return httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}})
        return httpx.Response(
            200,
            json={
                "provider": "P",
                "model": "m",
                "choices": [{"message": {"content": "The best option is Flight 1."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 1e-6},
            },
        )

    c = client_with(httpx.MockTransport(handler))
    out = asyncio.run(c.complete(SPEC, [{"role": "user", "content": "x"}]))
    assert out.text == "The best option is Flight 1." and c.limits["p"].rate_limited == 1


def test_limiter_caps_in_flight_and_serves_fifo() -> None:
    from flight_prefs.model_client import AdaptiveLimit

    async def go() -> tuple[int, list[int]]:
        lim = AdaptiveLimit(limit=3, floor=1)
        peak, order = 0, []

        async def job(i: int) -> None:
            nonlocal peak
            async with lim:
                peak = max(peak, lim.in_flight)
                order.append(i)
                await asyncio.sleep(0.001)

        await asyncio.gather(*(job(i) for i in range(50)))
        assert lim.in_flight == 0
        return peak, order

    peak, order = asyncio.run(go())
    assert peak == 3 and order == list(range(50))
