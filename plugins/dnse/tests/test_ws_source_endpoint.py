"""#160: the WS sources must honour a configured endpoint — RED-FIRST pin.

`WSTickSource` today takes no `ws_url` and constructs the vendored `TradingClient` without a
`base_url`, so the vendored default (the production host) always wins and the sub-minute tick
path can only ever dial production. `ws_order_source.py:95-107` already carries the fix and the
comment recording why: pointed at a non-production endpoint the feed still dialled production,
failed auth, and degraded to poll-only, "which is why the sandbox had NEVER exercised the WS
order path (measured 2026-09-16)". The order source was repaired; the tick source was not.

These pins FAIL against the current tree. That is the point: a pin that only ever ran against
the fixed code has shown nothing.

The second pin is the one that matters for #157. A configured endpoint is not enough on its own,
because the vendored connection passes an SSL context unconditionally
(`_vendor/dnse/websocket/connection.py:69-74`) and websockets 17.1 raises
`ValueError: ssl argument is incompatible with a ws:// URI` — measured 2026-09-18. So a local
fake serving plain `ws://` is unreachable by the real client however it is addressed, and the
sources need an injectable CLIENT FACTORY, not just a URL.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))

from pynecore_dnse import tick_source as tick_source_module      # noqa: E402


class _RecordingClient:
    """Stands in for the vendored TradingClient, recording how it was constructed."""

    def __init__(self, api_key, api_secret, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs

    def on(self, *a, **k):
        pass


def __test_the_tick_source_passes_a_configured_ws_url_to_its_client__(monkeypatch):
    """A configured endpoint must reach the client as base_url. Without this the tick feed
    dials production no matter what the plugin is pointed at."""
    captured = {}

    def _factory(api_key, api_secret, **kwargs):
        client = _RecordingClient(api_key, api_secret, **kwargs)
        captured.update(kwargs)
        return client

    monkeypatch.setattr(tick_source_module, "TradingClient", _factory)

    tick_source_module.WSTickSource("k", "s", "41I1G9000",
                                    ws_url="ws://127.0.0.1:8899")

    assert captured.get("base_url") == "ws://127.0.0.1:8899", (
        "a configured ws_url must be passed to the client as base_url; without it the vendored "
        "default (production) wins")


def __test_the_tick_source_omits_base_url_when_none_is_configured__(monkeypatch):
    """The discriminating half. A source that ALWAYS passed a base_url would satisfy the pin
    above while changing production behaviour; the vendored constant must stay the single
    definition of the production host, exactly as ws_order_source.py:104-106 keeps it."""
    captured = {}

    def _factory(api_key, api_secret, **kwargs):
        captured.update(kwargs)
        return _RecordingClient(api_key, api_secret, **kwargs)

    monkeypatch.setattr(tick_source_module, "TradingClient", _factory)

    tick_source_module.WSTickSource("k", "s", "41I1G9000")

    assert "base_url" not in captured, (
        "with no ws_url configured the source must not pass base_url at all")


def __test_the_tick_source_accepts_an_injected_client_factory__(monkeypatch):
    """A URL alone cannot reach a local fake: the vendored connection passes an SSL context
    unconditionally and websockets refuses `ssl` with a ws:// URI (measured 2026-09-18), so the
    real client can never connect to a plain local WS server. An injectable factory is what
    makes the WS path testable off production at all."""
    built = {}

    def _factory(api_key, api_secret, **kwargs):
        built["used"] = True
        return _RecordingClient(api_key, api_secret, **kwargs)

    source = tick_source_module.WSTickSource("k", "s", "41I1G9000",
                                             client_factory=_factory)

    assert built.get("used") is True, "the injected factory must build the client"
    assert source is not None
