"""Regression tests for ReplicateProvider's bounded-polling fallback.

Replicate's ``Prefer: wait`` usually returns a terminal prediction directly, but
when it comes back ``starting``/``processing`` the provider polls. These tests
exercise that loop with a fake client (no network): the success-on-the-final-poll
boundary must be honored, and a never-finishing prediction must raise cleanly.
"""

from __future__ import annotations

import pytest

from app.core.providers import replicate_provider as rp
from app.core.providers.base import ProviderError


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = ""

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Returns a queued sequence of prediction payloads on each GET."""

    def __init__(self, payloads: list[dict]):
        self._payloads = list(payloads)
        self.get_calls = 0

    async def get(self, url, headers=None):  # noqa: ANN001 - test stub
        self.get_calls += 1
        # Repeat the last payload if polled beyond the queue.
        payload = self._payloads.pop(0) if self._payloads else {"status": "processing"}
        return _FakeResponse(payload)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_seconds):
        return None

    monkeypatch.setattr(rp.asyncio, "sleep", _instant)


async def test_succeeds_on_final_poll_is_honored():
    provider = rp.ReplicateProvider(api_key="x")
    # processing for (MAX_POLLS - 1) polls, then succeeded on the very last one.
    payloads = [{"status": "processing", "urls": {"get": "http://poll"}}] * (rp._MAX_POLLS - 1)
    payloads.append({"status": "succeeded", "output": ["http://img/result.png"]})
    client = _FakeClient(payloads)

    initial = {"status": "starting", "urls": {"get": "http://poll"}}
    result = await provider._await_terminal(client, initial)

    assert result["status"] == "succeeded"
    assert client.get_calls == rp._MAX_POLLS  # used the entire budget, kept the win


async def test_never_finishes_raises_after_budget():
    provider = rp.ReplicateProvider(api_key="x")
    client = _FakeClient([{"status": "processing", "urls": {"get": "http://poll"}}])
    initial = {"status": "starting", "urls": {"get": "http://poll"}}

    with pytest.raises(ProviderError):
        await provider._await_terminal(client, initial)
    assert client.get_calls == rp._MAX_POLLS


async def test_already_terminal_returns_without_polling():
    provider = rp.ReplicateProvider(api_key="x")
    client = _FakeClient([])
    done = {"status": "succeeded", "output": ["http://img/result.png"]}

    result = await provider._await_terminal(client, done)

    assert result is done
    assert client.get_calls == 0
