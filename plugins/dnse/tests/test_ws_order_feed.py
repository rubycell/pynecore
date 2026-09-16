"""#121 dual-transport WS order feed — additive failsafe over the REST poll.

Covers the three hard requirements:
* the PROD path subscribes BOTH order channels — the SHORT ``order.{MT}.json``
  (``subscribe_order_event``, the only one ever MEASURED delivering prod frames:
  4 ``do`` frames on 2026-09-15) AND the BROKER ``order.broker.{MT}.{investor}``
  (``subscribe_broker_order_event``, never captured, kept as a failsafe) — with
  the market type UPPERCASED (a lowercase channel is accepted and streams
  nothing) and one channel's failure never killing the other;
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
from types import SimpleNamespace

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


# === 1. PROD path subscribes BOTH order channels ============================

class _FakeTradingClient:
    """Records subscribe/handler calls; no network.

    ``fail_on`` names the subscribe method that must raise (one-channel-down).
    """
    def __init__(self, *a, fail_on=None, **k):
        self.calls: list[tuple] = []
        self.handlers: dict[str, list] = {}
        self._fail_on = fail_on
        #: #135/2 — the ctor args matter: the endpoint the client dials is
        #: decided here, and getting it wrong is silent (prod default wins).
        self.ctor_args = a
        self.ctor_kwargs = k

    def _maybe_fail(self, name):
        if self._fail_on == name:
            raise ConnectionError(f"{name} refused by the venue")

    async def connect(self):
        self.calls.append(("connect",))

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    async def subscribe_order_event(self, market_type="STOCK",
                                    on_order_event=None, encoding="json"):
        self._maybe_fail("short_order")
        self.calls.append(("short_order", market_type, on_order_event))

    async def subscribe_broker_order_event(self, investor_id, market_type="STOCK",
                                           on_order_event=None, encoding="json"):
        self._maybe_fail("broker_order")
        self.calls.append(("broker_order", investor_id, market_type, on_order_event))

    async def subscribe_broker_position_event(self, investor_id, market_type="STOCK",
                                              on_position_event=None, encoding="json"):
        self._maybe_fail("broker_position")
        self.calls.append(("broker_position", investor_id, market_type))

    async def disconnect(self):
        self.calls.append(("disconnect",))


def _started_source(monkeypatch, *, fail_on=None, market_type="DERIVATIVE"):
    def _factory(*a, **k):
        return _FakeTradingClient(*a, fail_on=fail_on, **k)
    monkeypatch.setattr(ws_mod, "TradingClient", _factory)
    src = ws_mod.WSOrderSource("k", "s", "1000005917", market_type)
    return src


def __test_ws_order_source_subscribes_both_order_channels__(monkeypatch):
    """#121 correction: the SHORT channel is the only one PROD was ever measured
    delivering on (2026-09-15, 4 ``do`` frames) — it must be subscribed; the
    BROKER channel stays as the never-captured failsafe."""
    src = _started_source(monkeypatch)
    asyncio.run(src.start())
    kinds = [c[0] for c in src._client.calls]
    assert "short_order" in kinds, (
        "the SHORT channel order.{MT}.json is the PROD-measured deliverer — "
        "subscribing only the broker channel is why no prod frame was captured")
    assert "broker_order" in kinds, "the broker channel stays as the failsafe"
    assert "broker_position" in kinds
    assert src._started is True


def __test_channel_names_use_uppercase_market_type__(monkeypatch):
    """A lowercase channel name is silently accepted and streams NOTHING — the
    subscribe args must carry the UPPERCASED market type."""
    src = _started_source(monkeypatch, market_type="derivative")
    asyncio.run(src.start())
    short = next(c for c in src._client.calls if c[0] == "short_order")
    broker_call = next(c for c in src._client.calls if c[0] == "broker_order")
    position = next(c for c in src._client.calls if c[0] == "broker_position")
    assert short[1] == "DERIVATIVE", f"short channel market_type not upper: {short[1]!r}"
    assert broker_call[2] == "DERIVATIVE"
    assert position[2] == "DERIVATIVE"
    assert broker_call[1] == "1000005917"


def __test_order_callback_registered_once_not_per_subscribe__(monkeypatch):
    """The vendored client dispatches by EVENT name, not channel: passing the
    callback to BOTH subscribes would append it twice and invoke ``_on_order``
    twice per single frame (double-enqueue + double log)."""
    src = _started_source(monkeypatch)
    asyncio.run(src.start())
    assert src._client.handlers["order_event"] == [src._on_order]
    assert src._client.handlers["position_event"] == [src._on_position]
    for call in src._client.calls:
        if call[0] == "short_order":
            assert call[2] is None, "callback must not be re-registered per subscribe"
        if call[0] == "broker_order":
            assert call[3] is None


def __test_one_channel_subscribe_failure_keeps_the_other_live__(monkeypatch):
    """A subscribe raising on ONE channel must not kill the source: the other
    channel stays subscribed, the source still starts, and it is logged."""
    warnings: list[str] = []
    monkeypatch.setattr(ws_mod.log, "broker_warning",
                        lambda msg, *a: warnings.append(msg % a if a else msg))
    src = _started_source(monkeypatch, fail_on="short_order")
    asyncio.run(src.start())
    kinds = [c[0] for c in src._client.calls]
    assert "short_order" not in kinds
    assert "broker_order" in kinds, "the surviving channel must still be subscribed"
    assert src._started is True, "one failed channel must not degrade to poll-only"
    assert any("order.DERIVATIVE.json" in w for w in warnings), (
        f"the failing channel must be named in a warning; got {warnings}")
    # the frames of the surviving channel still reach the queue
    assert src._client.handlers["order_event"] == [src._on_order]


def __test_both_order_channels_failing_raises_to_poll_only__(monkeypatch):
    """Nothing subscribed -> raise, so ``_ensure_ws_order_source`` swallows it and
    the run is poll-only (rather than a started source that never yields)."""
    class _AllFail(_FakeTradingClient):
        def _maybe_fail(self, name):
            raise ConnectionError(name)

    monkeypatch.setattr(ws_mod, "TradingClient", lambda *a, **k: _AllFail())
    src = ws_mod.WSOrderSource("k", "s", "1000005917", "DERIVATIVE")
    with pytest.raises(RuntimeError):
        asyncio.run(src.start())
    assert src._started is False


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


def __test_same_frame_on_both_ws_channels_emits_one_event__(fake_client, collect,
                                                            monkeypatch):
    """#121 dual-channel: the venue may publish the SAME order event on BOTH the
    short and the broker channel. Two identical frames reach the queue (one per
    channel) and the shared ``_last_seen`` watermark — keyed
    ``(cumulative, raw_status)`` per order id, transport-agnostic — must collapse
    them to EXACTLY ONE engine event."""
    src = _started_source(monkeypatch)
    asyncio.run(src.start())

    frame = SimpleNamespace(id="O1", symbol="C1", side="NB", quantity=10.0,
                            fillQuantity=10.0, leaveQuantity=0.0,
                            canceledQuantity=0.0, price=101.0, averagePrice=101.0,
                            orderStatus="Filled", marketType="DERIVATIVE",
                            accountNo="ACC1")
    # the client dispatches the same event object once per channel it arrived on
    src._on_order(frame)          # short channel delivery
    src._on_order(frame)          # broker channel delivery
    assert src._queue.qsize() == 2, "both channel copies are enqueued (no pre-filter)"

    b = _broker(fake_client, get_orders=_books(None, None))
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    rows = asyncio.run(src.collect(0.5))
    assert len(rows) == 2, "collect() must hand BOTH channel copies to the scanner"
    b._ws_order_source = _ScriptedWSSource(rows)

    events = collect(b.watch_orders(), 1, timeout=0.3)
    fills = [e for e in events if e.event_type == "filled"]
    assert len(fills) == 1, (
        "the same frame on BOTH channels must emit ONE event; got "
        f"{[(e.order.id, e.fill_qty) for e in fills]}")
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


# === _resolve_investor_id: the /accounts BODY parsing (#129) =================
# Measured live 2026-09-16: the resolver read ``accounts[0]["investorId"]``, but
# ``investorId`` is a TOP-LEVEL field of the /accounts body — docs
# dnse-get-accounts.md schema lists it as "» investorId" while the accounts[]
# members carry only "»» id / dealAccount / derivativeAccount / derivative".
# It therefore returned None on every real body, and because None is also the
# legitimate "degrade to poll-only" answer, EVERY live run silently ran
# poll-only and the broker channel was never once subscribed. Nothing tested
# this parsing (the sibling account-ID parsing IS covered in
# test_broker_lifecycle.py); these are that missing coverage.

def _accounts_body(investor_id="1000005917"):
    """The REAL /accounts shape, as documented AND as served live 2026-09-16:
    ``investorId`` at the top level, never inside an accounts[] member."""
    body = {"name": "N", "custodyCode": "C", "accounts": [
        {"id": "0001179019", "dealAccount": True, "derivativeAccount": True,
         "derivative": {"status": "ACTIVE"}}]}
    if investor_id is not None:
        body["investorId"] = investor_id
    return body


def __test_investor_id_is_read_from_the_body_top_level__(fake_client):
    """THE red-first anchor for #129: a real body resolves. Against the pre-fix
    resolver this returns None (it indexed accounts[0]), which is exactly the
    silent poll-only degrade measured live."""
    b = _broker(fake_client, get_accounts=(200, _accounts_body("1000005917")))

    assert b._resolve_investor_id() == "1000005917", \
        "investorId is a TOP-LEVEL field of the /accounts body, not accounts[0]'s"


def __test_investor_id_is_cached_after_the_first_read__(fake_client):
    b = _broker(fake_client, get_accounts=(200, _accounts_body("1000005917")))

    first, second = b._resolve_investor_id(), b._resolve_investor_id()

    assert first == second == "1000005917"
    assert b._client.count("get_accounts") == 1, "second call must come from the cache"


@pytest.mark.parametrize("status, body", [
    (200, _accounts_body(investor_id=None)),   # 200, but the field is absent
    (200, "not-a-dict"),
    (500, {"code": "REMOTE_SERVER_ERROR"}),
])
def __test_investor_id_returns_none_when_it_cannot_be_read__(fake_client, status, body):
    """The degrade path stays intact: None on any unreadable body, so the caller
    still falls back to poll-only instead of raising into the live order path."""
    b = _broker(fake_client, get_accounts=(status, body))

    assert b._resolve_investor_id() is None


# === #134: a subscribe ACK is not evidence a channel is live ================
# Measured 2026-09-16 (#131): the venue accepts a broker-channel subscription
# and then refuses it asynchronously with an ERROR CONTROL FRAME
# ({"action":"error","code":"SUBSCRIBE_FAILED"}) AFTER the subscribe coroutine
# has already returned cleanly. ``start()``'s ``_try`` only catches EXCEPTIONS,
# so it recorded that channel as subscribed. The log then claimed a live
# transport that would never deliver — the #50 trap again, one level up.
# Positive evidence is a FRAME ARRIVING; everything else is a request.

def _fake_order(order_id="O9", status="Filled", fill=1.0):
    return SimpleNamespace(id=order_id, orderStatus=status, fillQuantity=fill,
                           quantity=1.0, averagePrice=100.0, symbol="C1",
                           side="NB", accountNo="ACC1")


def __test_subscribe_log_does_not_claim_the_channel_is_live__(monkeypatch, caplog):
    """RED-FIRST for #134: the subscribe-time line must describe a REQUEST.

    Pre-fix it read 'WS order feed subscribed: ...', which is exactly the claim
    #131 proved can be false."""
    src = _started_source(monkeypatch)
    with caplog.at_level("INFO"):
        asyncio.run(src.start())

    text = " ".join(r.message for r in caplog.records)
    assert "subscribe requested" in text.lower(), \
        "the subscribe-time line must say REQUESTED — a venue ACK is not confirmation (#134)"
    assert "feed subscribed:" not in text.lower(), \
        "the old wording asserted a live channel on an ACK alone (#131 showed that is false)"


def __test_first_frame_emits_the_live_milestone_exactly_once__(monkeypatch, caplog):
    """The milestone records something that HAPPENED (a frame arrived), and it
    must not re-fire on every subsequent frame."""
    src = _started_source(monkeypatch)
    asyncio.run(src.start())
    with caplog.at_level("INFO"):
        src._on_order(_fake_order("O1"))
        src._on_order(_fake_order("O2"))

    milestones = [r.message for r in caplog.records
                  if "WS ORDER SOURCE FIRST LIVE FRAME" in r.message]
    assert len(milestones) == 1, \
        f"exactly one first-frame milestone expected, got {len(milestones)}"


def __test_server_error_frame_warns_and_never_emits_the_milestone__(monkeypatch,
                                                                    caplog):
    """The discriminating control: the #131 sequence (clean subscribe, THEN a
    server error frame, and NO frame ever delivered) must never look live."""
    src = _started_source(monkeypatch)
    asyncio.run(src.start())
    with caplog.at_level("INFO"):
        for handler in src._client.handlers.get("error", []):
            handler(Exception("internal error"))

    text = " ".join(r.message for r in caplog.records)
    assert "WS ORDER SOURCE FIRST LIVE FRAME" not in text, \
        "an error frame must NEVER be mistaken for delivery"
    assert "server error" in text.lower() or "refused" in text.lower(), \
        "a server-side subscription error must be reported, not swallowed"


# === the watermark must never walk BACKWARDS (live incident 2026-09-16) =====
# A real fill of ONE contract was counted TWICE on prod, and the engine's
# position belief went 1.0 -> 2.0. strategy.close("E") then closed what it
# believed it held — 2 — against a real long of 1, opening a NAKED SHORT.
#
# Mechanism (from the run log, not inferred): the REST poll observed the order
# already `Filled` (cum=1) and set the watermark. The WS queue THEN drained
# frames that PREDATE that observation — PendingNew and New, both cum=0 —
# and `_scan_row` ends with an UNCONDITIONAL
#     self._last_seen[order_id] = (cumulative, raw_status)
# which walked the watermark back to 0. The WS's own Filled(cum=1) then looked
# like a fresh 0->1 delta and emitted a SECOND fill.
#
# Note why every existing test missed it: they model both transports seeing the
# SAME cumulative. Nothing modelled a LOWER cumulative arriving AFTER a higher
# one, which is exactly what a queued transport does when it drains behind a
# poll that already saw the end state.

def __test_stale_frame_must_not_walk_the_watermark_backwards__(fake_client):
    """One venue fill of 1 must be counted ONCE, whatever order the frames
    arrive in. This is the live shape, replayed exactly."""
    async def run():
        b = _broker(fake_client)
        b._identity["O1"] = ("pineA", None, LegType.ENTRY)
        # the POLL gets there first and sees the END state:
        poll = await b._scan_row(_order_row("O1", "Filled", fill=1.0, qty=1.0))
        # the WS queue now drains frames that predate it (cum=0):
        stale_pending = await b._scan_row(
            _order_row("O1", "PendingNew", fill=0.0, qty=1.0))
        stale_new = await b._scan_row(_order_row("O1", "New", fill=0.0, qty=1.0))
        # ...followed by the WS's own copy of the fill it already counted:
        ws_filled = await b._scan_row(_order_row("O1", "Filled", fill=1.0, qty=1.0))
        return poll, stale_pending, stale_new, ws_filled, b._last_seen["O1"]

    poll, stale_pending, stale_new, ws_filled, watermark = asyncio.run(run())

    # non-fill events (CREATED) carry fill_qty=None — count only real fills
    counted = sum(event.fill_qty or 0.0 for group in
                  (poll, stale_pending, stale_new, ws_filled) for event in group)
    assert counted == 1.0, (
        f"ONE venue fill of 1 contract must be counted ONCE, got {counted} — "
        "a double count makes strategy.close() oversize and opens a reverse "
        "position (live 2026-09-16: long 1 -> close sell 2 -> naked short 1)")
    assert watermark[0] == 1.0, (
        f"the watermark must not regress below the highest cumulative seen; "
        f"got {watermark}")


# === #135/2: the WS order feed must dial the CONFIGURED endpoint ============
# Measured 2026-09-16: WSOrderSource built TradingClient(api_key, api_secret,
# auto_reconnect=True) and never passed base_url, so the vendored default
# ("wss://ws-openapi.dnse.com.vn" — PROD) always won. Pointed at the sandbox the
# WS order feed therefore dialled PROD with SANDBOX keys, failed auth
# ("invalid API key") and degraded to poll-only. Consequence: the sandbox has
# NEVER been able to exercise the WS order path, which is a large part of why
# the watermark double-count reached production untested.

def _captured_client(monkeypatch):
    """Patch in the fake and hand back the instance it built."""
    built = {}

    def _factory(*a, **k):
        client = _FakeTradingClient(*a, **k)
        built["client"] = client
        return client

    monkeypatch.setattr(ws_mod, "TradingClient", _factory)
    return built


def __test_ws_order_source_dials_the_configured_endpoint__(monkeypatch):
    """A configured ws_url (e.g. the sandbox) MUST reach the client."""
    built = _captured_client(monkeypatch)
    ws_mod.WSOrderSource("k", "s", "1000005917", "DERIVATIVE",
                         ws_url="wss://ws-sb-openapi.dnse.com.vn")

    assert built["client"].ctor_kwargs.get("base_url") == \
        "wss://ws-sb-openapi.dnse.com.vn", (
        "the configured ws_url must be passed to TradingClient — otherwise the "
        "vendored PROD default wins and a sandbox run silently dials prod")


def __test_ws_order_source_without_a_url_leaves_the_vendored_default__(monkeypatch):
    """No ws_url configured -> do NOT force one: the vendored default stays the
    single source of truth for the prod endpoint (no second hard-coded copy)."""
    built = _captured_client(monkeypatch)
    ws_mod.WSOrderSource("k", "s", "1000005917", "DERIVATIVE")

    assert "base_url" not in built["client"].ctor_kwargs, \
        "with no configured url the vendored default must apply, unduplicated"


def __test_equal_cumulative_status_transition_still_flows__(fake_client):
    """THE OVER-DROP GUARD for the staleness fix.

    A terminal status arriving at the SAME cumulative as a partial (e.g. the
    remainder is Canceled after a partial fill) must still be PROCESSED. If the
    staleness guard were written as `cumulative <= previous -> drop`, it would
    silently blind the cancel/expiry machinery (#124/#135) — this pins that it
    is `<`, not `<=`."""
    async def run():
        b = _broker(fake_client)
        b._identity["O1"] = ("pineA", None, LegType.ENTRY)
        partial = await b._scan_row(
            _order_row("O1", "PartiallyFilled", fill=9.0, qty=30.0))
        # same cumulative, TERMINAL status — the rest of the order was cancelled
        terminal = await b._scan_row(
            _order_row("O1", "Canceled", fill=9.0, qty=30.0))
        return partial, terminal, b._last_seen["O1"]

    partial, terminal, watermark = asyncio.run(run())
    assert sum(e.fill_qty or 0.0 for e in partial) == 9.0
    assert terminal, ("a terminal status at an UNCHANGED cumulative must still "
                      "produce an event — dropping it blinds cancel detection")
    assert watermark == (9.0, "Canceled"), \
        "the status must advance even when the cumulative does not"


def __test_restart_seeded_high_water_suppresses_replayed_fills__(fake_client):
    """A restart seeds the watermark from the journal. Frames replayed by a
    transport that reconnects and re-sends the order's earlier lifecycle must
    NOT re-emit fills the previous run already booked — the ratchet working as
    designed, not a lost fill."""
    async def run():
        b = _broker(fake_client)
        b._identity["O1"] = ("pineA", None, LegType.ENTRY)
        # as the journal seeding does at startup (broker.py ~504):
        b._last_seen["O1"] = (1.0, "Filled")
        replay_pending = await b._scan_row(
            _order_row("O1", "PendingNew", fill=0.0, qty=1.0))
        replay_new = await b._scan_row(_order_row("O1", "New", fill=0.0, qty=1.0))
        replay_filled = await b._scan_row(
            _order_row("O1", "Filled", fill=1.0, qty=1.0))
        return replay_pending, replay_new, replay_filled, b._last_seen["O1"]

    a, b_, c, watermark = asyncio.run(run())
    counted = sum(e.fill_qty or 0.0 for group in (a, b_, c) for e in group)
    assert counted == 0.0, (
        f"pre-restart fills must not be re-emitted, got {counted} — the "
        "previous run already booked them")
    assert watermark[0] == 1.0, \
        f"seeded high-water must not regress, got {watermark}"
