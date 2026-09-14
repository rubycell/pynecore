"""#121 dual-transport WS order feed — additive failsafe over the REST poll.

Covers the three hard requirements:
* the PROD path subscribes the BROKER channels (``subscribe_broker_order_event``),
  never the SANDBOX ``subscribe_order_event`` every prior prod probe wrongly used;
* the shared ``_last_seen`` cumulative watermark dedups the SAME fill seen on both
  transports (one delta), and sizes partial deltas correctly (9/30, 80/100, and a
  poll that jumps 0->30);
* WS failure/absence degrades to poll-only — the poll keeps detecting fills, no
  crash, no gap.

Fill DELTAS are the plugin's contract to the engine: ``record_fill`` consumes a
delta and the engine arms/extends protection against the resulting
``position.size``, so a correct per-transport delta IS correct arm sizing.
"""
from __future__ import annotations

import asyncio

import pytest

from pynecore_dnse import broker
from pynecore_dnse import ws_order_source as ws_mod
from pynecore.core.broker.models import LegType


# --- helpers (local mirror of test_broker_state's) --------------------------

@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def _fast_sleep(_delay, result=None):
        return result
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


def _broker(fake_client, *, symbol="VN30F1M", account_no="ACC1", **responses):
    cfg = broker.DNSEBrokerConfig(api_key="k", api_secret="s", account_no=account_no)
    instance = broker.DNSEBroker(symbol=symbol, timeframe="5", config=cfg)
    instance._client = fake_client(**responses)
    return instance


def _order_row(order_id, status, *, symbol="C1", side="NB", qty=10.0, fill=0.0,
               avg_price=100.0):
    return {"id": order_id, "symbol": symbol, "side": side, "quantity": qty,
            "fillQuantity": fill, "orderStatus": status, "averagePrice": avg_price}


def _books(normal=None, stop=None):
    def get_orders(account, market_type, order_category=None, page_index=0,
                   page_size=100):
        body = normal if order_category == "NORMAL" else stop
        return body if body is not None else (200, {"orders": []})
    return get_orders


# === 1. PROD path uses the BROKER channel, not the sandbox one ==============

class _FakeTradingClient:
    """Records subscribe calls; no network."""
    def __init__(self, *a, **k):
        self.calls: list[tuple] = []

    async def connect(self):
        self.calls.append(("connect",))

    async def subscribe_order_event(self, *a, **k):        # SANDBOX channel
        self.calls.append(("subscribe_order_event", a, k))

    async def subscribe_broker_order_event(self, investor_id, market_type,
                                           on_order_event=None, encoding="json"):
        self.calls.append(("broker_order", investor_id, market_type))

    async def subscribe_broker_position_event(self, investor_id, market_type,
                                              on_position_event=None, encoding="json"):
        self.calls.append(("broker_position", investor_id, market_type))

    async def disconnect(self):
        self.calls.append(("disconnect",))


def __test_ws_order_source_subscribes_broker_channels_not_sandbox__(monkeypatch):
    monkeypatch.setattr(ws_mod, "TradingClient", _FakeTradingClient)
    src = ws_mod.WSOrderSource("k", "s", "1000005917", "DERIVATIVE")
    asyncio.run(src.start())
    kinds = [c[0] for c in src._client.calls]
    assert "broker_order" in kinds, (
        "#121 prod path MUST use subscribe_broker_order_event — the sandbox "
        "subscribe_order_event is why no prod frame was ever captured")
    assert "broker_position" in kinds
    assert "subscribe_order_event" not in kinds, (
        "the SANDBOX channel must never be used on the prod path")
    broker_call = next(c for c in src._client.calls if c[0] == "broker_order")
    assert broker_call[1] == "1000005917" and broker_call[2] == "DERIVATIVE"


# === 2. Shared watermark: same fill on both transports -> one delta =========

def __test_shared_watermark_dedups_same_fill_across_transports__(fake_client):
    """A partial(cum=9) seen by BOTH transports advances the watermark ONCE and
    emits ONE delta=9; the later Filled(cum=30) emits delta=21 (measured 9/30)."""
    async def run():
        b = _broker(fake_client)
        b._identity["O1"] = ("pineA", None, LegType.ENTRY)
        # transport #1 (say WS) sees cum=9:
        e_first = await b._scan_row(
            _order_row("O1", "PartiallyFilled", fill=9.0, qty=30.0))
        # transport #2 (the poll) sees the IDENTICAL cum=9 — a cross-transport
        # duplicate:
        e_dup = await b._scan_row(
            _order_row("O1", "PartiallyFilled", fill=9.0, qty=30.0))
        # the remainder lands (cum=30):
        e_rest = await b._scan_row(
            _order_row("O1", "Filled", fill=30.0, qty=30.0))
        return e_first, e_dup, e_rest, b._last_seen["O1"]

    e_first, e_dup, e_rest, watermark = asyncio.run(run())
    assert sum(ev.fill_qty for ev in e_first) == 9.0, "first sighting -> delta 9"
    assert e_dup == [], "the same cum on the other transport must dedup to nothing"
    assert sum(ev.fill_qty for ev in e_rest) == 21.0, "remainder -> delta 30-9=21"
    assert watermark == (30.0, "Filled"), "watermark advanced once, to cumulative 30"


def __test_partial_then_full_80_100_emits_matching_deltas__(fake_client):
    """The other measured shape: 80/100 -> deltas 80 then 20 (never 100-while-80,
    never the last 20 lost)."""
    async def run():
        b = _broker(fake_client)
        b._identity["O1"] = ("pineA", None, LegType.ENTRY)
        e1 = await b._scan_row(_order_row("O1", "PartiallyFilled", fill=80.0, qty=100.0))
        e2 = await b._scan_row(_order_row("O1", "Filled", fill=100.0, qty=100.0))
        return e1, e2

    e1, e2 = asyncio.run(run())
    assert sum(ev.fill_qty for ev in e1) == 80.0
    assert sum(ev.fill_qty for ev in e2) == 20.0


def __test_poll_jump_zero_to_thirty_emits_full_delta__(fake_client):
    """If the poll first observes the order only AFTER both WS ticks (a 0->30
    jump), it must still arm for the FULL 30 — one delta=30, not a lost fill."""
    async def run():
        b = _broker(fake_client)
        b._identity["O1"] = ("pineA", None, LegType.ENTRY)
        e = await b._scan_row(_order_row("O1", "Filled", fill=30.0, qty=30.0))
        return e, b._last_seen["O1"]

    events, watermark = asyncio.run(run())
    assert sum(ev.fill_qty for ev in events) == 30.0, "0->30 jump -> full delta 30"
    assert watermark == (30.0, "Filled")


# === 3. Failsafe: WS failure/absence -> poll still protects, no crash =======

class _RaisingWSSource:
    async def collect(self, timeout):
        raise ConnectionError("WS dropped mid-stream (server 1000, #92)")

    async def stop(self):
        pass


class _ScriptedWSSource:
    """Yields a fixed batch of raw rows once, then nothing (drained)."""
    def __init__(self, rows):
        self._rows = rows

    async def collect(self, timeout):
        rows, self._rows = self._rows, []
        return rows

    async def stop(self):
        pass


def __test_collect_ws_order_events_swallows_ws_failure_returns_empty__(fake_client):
    """A WS ``collect`` that raises must be swallowed to [] — the poll floor is
    untouched, the watch loop never dies."""
    async def run():
        b = _broker(fake_client)
        b._ws_order_source = _RaisingWSSource()       # already "started"
        return await b._collect_ws_order_events(0.0)

    assert asyncio.run(run()) == [], "a WS failure must degrade to poll-only, not raise"


def __test_ws_frame_drives_fill_then_poll_dedups_it__(fake_client, collect):
    """End-to-end dual transport: a WS order frame drives the fill through
    ``watch_orders`` (fast path), and the poll — reading the SAME order at the
    SAME cumulative — deduces it via the shared watermark, so the fill is
    protected exactly once."""
    ws_row = _order_row("O1", "Filled", fill=10.0, qty=10.0, avg_price=101.0)
    poll_row = _order_row("O1", "Filled", fill=10.0, qty=10.0, avg_price=101.0)
    b = _broker(fake_client, get_orders=_books((200, {"orders": [poll_row]}), None))
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    b._ws_order_source = _ScriptedWSSource([ws_row])   # inject a ready WS source

    events = collect(b.watch_orders(), 3, timeout=0.3)
    fills = [e for e in events if e.event_type == "filled"]
    assert len(fills) == 1, (
        "the fill must be detected EXACTLY ONCE across the WS + poll transports; "
        f"got {len(fills)} ({[ (e.order.id, e.fill_qty) for e in fills ]})")
    assert fills[0].fill_qty == 10.0


def __test_ws_drop_leaves_poll_to_detect_the_fill__(fake_client, collect):
    """FAILSAFE: the WS source raises on every collect, yet the poll still
    detects and yields the fill — WS failure degrades latency, never protection,
    and never crashes the run."""
    poll_row = _order_row("O1", "Filled", fill=10.0, qty=10.0, avg_price=101.0)
    b = _broker(fake_client, get_orders=_books((200, {"orders": [poll_row]}), None))
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    b._ws_order_source = _RaisingWSSource()            # WS is broken

    events = collect(b.watch_orders(), 1, timeout=0.3)
    fills = [e for e in events if e.event_type == "filled"]
    assert len(fills) == 1, "the poll floor must still detect the fill when WS is down"
    assert fills[0].fill_qty == 10.0
