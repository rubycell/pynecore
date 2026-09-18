"""#100 — the LTF feed's tick source, on the vendored TradingClient.

The WS tick stream is the ONLY sanctioned sub-minute bar source: REST
``/trades/latest`` is a no-pagination latest-print sample that retains
~10% of prints at 1 Hz (panel-measured against the recorded corpus) — no
path may build a sub-minute bar from it (#100 adjudication, hard rule).
The vendored client is mandated for production WS paths (auth within
30 s, SDK-only connection path — the 2026-08-26 measured lessons) and
carries auto-reconnect with re-subscription.

``WSTickSource`` adapts the callback-driven client to an awaitable
per-tick pull with a bounded queue. A FULL queue marks overflow instead
of dropping silently — a dropped print is a wrong high/low, and the
consumer must know (the aggregator's cumulative-volume check will flag
the affected bar suspect anyway; the flag here makes the cause loud).
"""
import asyncio

from pynecore.lib import log

from ._vendor.dnse.websocket.client import TradingClient
from .tick_frames import parse_tick_time


class WSTickSource:
    """Per-print ticks for ONE wire symbol, board G1 (continuous — the
    T1 put-through board carries an independent volume counter and never
    feeds synthesis, the measured 2026-08-25 lesson)."""

    def __init__(self, api_key: str, api_secret: str, wire_symbol: str,
                 queue_max: int = 20_000, ws_url: "str | None" = None,
                 client_factory=None) -> None:
        # #160 — HONOUR THE CONFIGURED ENDPOINT, mirroring ws_order_source.py:95-107 rather than
        # inventing a second idiom. This used to call ``TradingClient(api_key, api_secret,
        # auto_reconnect=True)`` with no ``base_url``, letting the vendored production default
        # win, so the sub-minute tick feed could only ever dial PRODUCTION — the same defect
        # #135/2 fixed in the sibling and left unfixed here. ``base_url`` is passed ONLY when
        # configured, so the vendored constant stays the single definition of the prod host.
        #
        # ``client_factory`` exists because a URL alone is NOT enough. The vendored connection
        # passes an SSL context unconditionally (``_vendor/dnse/websocket/connection.py:69-74``)
        # and websockets 17.1 raises ``ssl argument is incompatible with a ws:// URI`` (measured
        # 2026-09-18), so the real client cannot reach a plain local WS server however it is
        # addressed. Injecting the client is what makes the WS path testable off production at
        # all — and it is why ``fake_dnse_ws.py`` has been unreachable since the SDK was vendored.
        client_kwargs = {"auto_reconnect": True}
        if ws_url:
            client_kwargs["base_url"] = ws_url
        factory = client_factory or TradingClient
        self._client = factory(api_key, api_secret, **client_kwargs)
        self._wire_symbol = wire_symbol
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self._overflowed = False
        self._started = False

    def _on_trade(self, trade) -> None:
        """Vendored-client callback (its dispatch worker thread/task)."""
        if getattr(trade, "boardId", "G1") != "G1":
            return
        ts = parse_tick_time(getattr(trade, "time", None))
        price = getattr(trade, "price", None)
        if ts is None or price is None:
            return
        qty = float(getattr(trade, "quantity", 0) or 0)
        cumulative = getattr(trade, "totalVolumeTraded", None)
        cumulative = float(cumulative) if cumulative is not None else None
        try:
            self._queue.put_nowait((ts, float(price), qty, cumulative))
        except asyncio.QueueFull:
            # Never drop silently: the consumer stalled badly. The
            # cumulative check flags the bars; this flag names the cause.
            self._overflowed = True

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    async def start(self) -> None:
        if self._started:
            return
        await self._client.connect()
        await self._client.subscribe_trades(
            [self._wire_symbol], on_trade=self._on_trade, board_id="G1")
        self._started = True
        log.broker_info(
            "LTF tick source subscribed: %s board=G1 (WS per-print, #100)",
            self._wire_symbol)

    async def next_tick(self, timeout: float):
        """The next ``(ts, price, qty, cumulative)`` or raises
        ``asyncio.TimeoutError`` after ``timeout`` seconds of silence —
        the CALLER decides whether silence is an outage (raise to the
        engine) or a quiet venue phase (keep waiting)."""
        return await asyncio.wait_for(self._queue.get(), timeout)

    async def stop(self) -> None:
        if self._started:
            try:
                await self._client.disconnect()
            except Exception:                                 # noqa: BLE001
                pass
            self._started = False
