"""#121 — the PROD WS order/position feed, on the vendored TradingClient.

A FAILSAFE, ADDITIVE speed path that runs ALONGSIDE the REST order-book poll in
``broker.watch_orders``. It subscribes BOTH order channels the venue exposes
(``_vendor/dnse/websocket/client.py``):

* ``order.{MARKET_TYPE}.json`` — the SHORT channel
  (``TradingClient.subscribe_order_event``). **Measured 2026-09-15 on PROD**:
  4 ``do`` frames captured during a real OCO place+cancel — the first prod order
  frames ever captured. The sandbox uses this same channel.
* ``order.broker.{MARKET_TYPE}.{investorId}.json`` — the BROKER channel
  (``subscribe_broker_order_event``), what this source used to subscribe
  EXCLUSIVELY. It has NEVER delivered a captured frame on prod; it is kept
  because subscribing it costs nothing and a venue that starts publishing there
  must not be missed.
* ``position.broker.{MARKET_TYPE}.{investorId}.json`` (position updates,
  observability only).

``market_type`` is upper-cased into the channel name: a lowercase channel name is
silently ACCEPTED by the venue (``status: active``) and then streams NOTHING —
that mistake produced a false "the trading WS is silent" verdict once already.

Subscribing both channels is safe because the callbacks are registered ONCE on
the client's ``order_event`` dispatch (see ``start``), and both channels feed the
same queue -> the same ``broker._scan_row`` watermark: a frame delivered on BOTH
channels is two identical raw rows, and the second dedups to zero events
(``_last_seen`` keys ``(cumulative, raw_status)`` per order id — transport
agnostic by construction). A subscribe failure on ONE channel leaves the other
live (logged); only "no channel at all" degrades to poll-only.

Design contract (why this can never make protection WORSE):

* The vendored client's dispatch runs its OWN tasks; its callbacks here only
  ``put_nowait`` a normalised raw-row dict onto an internal queue. ``_scan_row``
  — the single read-modify-write of the shared ``_last_seen`` watermark — is
  therefore called ONLY by the one ``watch_orders`` task that drains this queue,
  never concurrently. So cross-transport dedup is race-free with NO extra lock:
  a fill seen on WS advances the watermark; the later poll sees
  ``cumulative == previous`` and emits nothing (and vice-versa).
* Every failure is swallowed to poll-only: a connect/auth failure leaves
  ``started`` False; a mid-stream drop is logged and the queue simply stops
  filling, so ``broker.watch_orders`` behaves exactly like the poll-only floor.
  A WS exception NEVER propagates into the fill/protection path.
* The vendored ``Order.from_dict`` coerces ``float(price)`` etc., so a
  prod frame missing a field raises INSIDE the SDK dispatch (before our
  callback) — that degrades to poll-only too, it does not crash the run.

Operator mandate honoured: this uses the vendored ``TradingClient``, never a
hand-rolled WS client.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pynecore.lib import log

from ._vendor.dnse.websocket.client import TradingClient


def _mask(value: Any) -> str:
    """Mask an id for logs — keep the last 4 chars only."""
    s = str(value or "")
    return ("*" * max(len(s) - 4, 0)) + s[-4:] if s else ""


def _order_model_to_raw_row(order: Any) -> dict:
    """Normalise a vendored WS ``Order`` model into the SAME raw-row shape the
    REST poll produces, so it can go straight through ``broker._scan_row``
    (which reads ``id`` / ``fillQuantity`` / ``orderStatus`` / ``averagePrice``
    and hands the rest to ``_to_exchange_order``). A NORMAL-book fill carries no
    ``stopPrice`` (``None`` -> ``_to_exchange_order`` treats it as a LIMIT)."""
    return {
        "id": str(getattr(order, "id", "") or ""),
        "symbol": getattr(order, "symbol", None),
        "side": getattr(order, "side", None),
        "quantity": getattr(order, "quantity", 0),
        "fillQuantity": getattr(order, "fillQuantity", 0),
        "leaveQuantity": getattr(order, "leaveQuantity", 0),
        "canceledQuantity": getattr(order, "canceledQuantity", 0),
        "price": getattr(order, "price", 0),
        "averagePrice": getattr(order, "averagePrice", 0),
        "orderStatus": getattr(order, "orderStatus", None),
        "marketType": getattr(order, "marketType", None),
    }


class WSOrderSource:
    """PROD broker order/position frames for one investor, feeding
    ``broker.watch_orders`` alongside the REST poll (#121)."""

    def __init__(self, api_key: str, api_secret: str, investor_id: str,
                 market_type: str, queue_max: int = 10_000,
                 ws_url: "str | None" = None) -> None:
        # #135/2 — HONOUR THE CONFIGURED ENDPOINT. This used to call
        # ``TradingClient(api_key, api_secret, auto_reconnect=True)``, letting the
        # vendored default ``base_url="wss://ws-openapi.dnse.com.vn"`` win, so the
        # WS ORDER feed always dialled PROD even when the plugin was pointed at the
        # sandbox. With sandbox keys against the prod host it fails auth ("invalid
        # API key") and degrades to poll-only — which is why the sandbox had NEVER
        # exercised the WS order path (measured 2026-09-16, during the watermark
        # double-count investigation). ``base_url`` is passed ONLY when configured,
        # so the vendored constant stays the single definition of the prod host.
        client_kwargs = {"auto_reconnect": True}
        if ws_url:
            client_kwargs["base_url"] = ws_url
        self._client = TradingClient(api_key, api_secret, **client_kwargs)
        self._investor_id = investor_id
        self._market_type = market_type
        #: normalised raw-row dicts awaiting _scan_row (order frames only).
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self._started = False
        #: #134 — channels we ASKED for. A venue ACK is not confirmation (#131),
        #: so this is a record of intent, never of a live transport.
        self._requested_channels: list[str] = []
        #: #134 — the first delivered frame is the ONLY positive evidence that
        #: the transport works; the milestone fires once, on that frame.
        self._first_frame_logged = False

    def _on_order(self, order: Any) -> None:
        """Vendored-client callback (its dispatch task). Log the frame (masked)
        for observability — this is the evidence a prod order frame arrived —
        then enqueue the normalised raw row for the watch task to _scan_row."""
        raw = _order_model_to_raw_row(order)
        if not self._first_frame_logged:
            self._first_frame_logged = True
            log.broker_info(
                "[BROKER] WS ORDER SOURCE FIRST LIVE FRAME — the transport is "
                "DELIVERING (channels requested: %s). This, not the subscribe "
                "acknowledgement, is proof the WS order path works (#134).",
                " + ".join(self._requested_channels) or "<none recorded>")
        log.broker_info(
            "[BROKER] order frame via WS: id=%s status=%s fillQty=%s "
            "avgPx=%s acct=%s",
            _mask(raw.get("id")), raw.get("orderStatus"),
            raw.get("fillQuantity"), raw.get("averagePrice"),
            _mask(getattr(order, "accountNo", "")),
        )
        try:
            self._queue.put_nowait(raw)
        except asyncio.QueueFull:                             # pragma: no cover
            # Never crash the SDK dispatch task: the poll floor still detects
            # the fill. Drop the WS copy loudly.
            log.broker_warning(
                "WS order queue full — dropping a WS frame (poll still detects "
                "the fill); consumer stalled")

    def _on_server_error(self, error: Any) -> None:
        """A server ERROR control frame (#134).

        ATTRIBUTION IS NOT POSSIBLE HERE, and the log says so rather than
        guessing: the vendored client drops both the error CODE and the channel
        (``client.py`` emits ``Exception(data.get("message"))`` only), so a
        ``SUBSCRIBE_FAILED`` for the broker channel arrives indistinguishable
        from any other server error. We therefore report it and let the
        first-frame milestone decide what is actually live — never mark a
        specific channel dead from this."""
        log.broker_warning(
            "[BROKER] WS server error frame: %s — a subscription may have been "
            "REFUSED (the venue sends this AFTER acking the subscribe, #131). "
            "The channel cannot be identified from this frame (the client "
            "passes no code and no channel), so treat the absence of a FIRST "
            "LIVE FRAME as the real signal. Poll remains the floor.", error)

    def _on_position(self, position: Any) -> None:
        """Position frames are OBSERVABILITY ONLY — the engine's position is
        fill-derived, so a position frame never drives protection. Logged
        (masked) so today's live run captures the first prod position frame."""
        log.broker_info(
            "[BROKER] position frame via WS: id=%s symbol=%s status=%s "
            "openQty=%s side=%s",
            _mask(getattr(position, "id", "")),
            getattr(position, "symbol", None),
            getattr(position, "status", None),
            getattr(position, "openQuantity", None),
            getattr(position, "side", None),
        )

    async def start(self) -> None:
        """Connect, then subscribe BOTH order channels (short + broker) plus the
        broker position channel. One channel failing must NOT take the others
        down — each subscribe is guarded and merely logged. Raises only when the
        connect fails or NO order channel came up, so the caller
        (``broker._ensure_ws_order_source``) degrades to poll-only."""
        if self._started:
            return
        await self._client.connect()
        # UPPERCASE: a lowercase channel name is accepted and streams nothing.
        market_type = str(self._market_type or "").upper()
        # Register the dispatch callbacks ONCE, before any subscribe: the
        # vendored client dispatches by EVENT name ("order_event"), not by
        # channel, so passing the callback to each subscribe would append a
        # second handler and invoke us twice per frame. Registering here also
        # means a failed FIRST subscribe cannot leave the surviving channel
        # without a handler.
        self._client.on("order_event", self._on_order)
        self._client.on("position_event", self._on_position)
        # #134: the venue refuses a subscription ASYNCHRONOUSLY, with an error
        # CONTROL frame delivered after the subscribe coroutine has already
        # returned cleanly — so `_try` below can never see it.
        self._client.on("error", self._on_server_error)

        subscribed: list[str] = []

        async def _try(channel: str, coroutine_factory) -> None:
            try:
                await coroutine_factory()
            except Exception as exc:                              # noqa: BLE001
                log.broker_warning(
                    "WS subscribe failed for %s (%s: %s) — continuing with the "
                    "remaining channels (poll remains the floor)",
                    channel, type(exc).__name__, exc)
            else:
                subscribed.append(channel)

        short_channel = f"order.{market_type}.json"
        broker_channel = f"order.broker.{market_type}.{_mask(self._investor_id)}.json"
        position_channel = (
            f"position.broker.{market_type}.{_mask(self._investor_id)}.json")

        # PROD-measured deliverer (2026-09-15) + the sandbox channel.
        await _try(short_channel, lambda: self._client.subscribe_order_event(
            market_type, on_order_event=None))
        # Kept as the failsafe second transport (never yet captured on prod).
        await _try(broker_channel, lambda: self._client.subscribe_broker_order_event(
            self._investor_id, market_type, on_order_event=None))
        order_channels = list(subscribed)
        await _try(position_channel,
                   lambda: self._client.subscribe_broker_position_event(
                       self._investor_id, market_type, on_position_event=None))

        if not order_channels:
            raise RuntimeError(
                "WS order feed: BOTH order channels failed to subscribe "
                f"({short_channel} and {broker_channel}) — poll-only")
        self._started = True
        self._requested_channels = list(subscribed)
        log.broker_info(
            "[BROKER] WS order feed subscribe REQUESTED for: %s — the venue "
            "ACKs a subscription it may never honour (#131: a broker channel "
            "answered SUBSCRIBE_FAILED asynchronously, AFTER a clean "
            "subscribe), so this line is NOT evidence a channel is live. The "
            "FIRST DELIVERED FRAME is (#134; poll remains the floor).",
            " + ".join(subscribed))

    async def collect(self, timeout: float) -> list[dict]:
        """Wait up to ``timeout`` s for the next WS order frame; return it plus
        any already queued (>=1 as soon as one arrives, else ``[]`` on timeout).
        This is what lets a WS fill be yielded within ms while the caller keeps
        polling on its own cadence. WS-quiet behaves like ``sleep(timeout)``."""
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return []
        rows = [first]
        while True:
            try:
                rows.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return rows

    async def stop(self) -> None:
        if self._started:
            try:
                await self._client.disconnect()
            except Exception:                                 # noqa: BLE001
                pass
            self._started = False
