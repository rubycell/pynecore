"""#121 — the PROD WS order/position feed, on the vendored TradingClient.

A FAILSAFE, ADDITIVE speed path that runs ALONGSIDE the REST order-book poll in
``broker.watch_orders``. It subscribes to the DNSE **broker** channels the
official example uses:

* ``order.broker.{market_type}.{investorId}.{encoding}`` (fills/cancels)
* ``position.broker.{market_type}.{investorId}.{encoding}`` (position updates)

via ``TradingClient.subscribe_broker_order_event`` /
``subscribe_broker_position_event`` (``_vendor/dnse/websocket/client.py`` — the
PROD channels; ``subscribe_order_event`` WITHOUT ``broker.`` is the SANDBOX
channel every prior prod probe wrongly used, which is why no prod frame has ever
been captured).

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
                 market_type: str, queue_max: int = 10_000) -> None:
        self._client = TradingClient(api_key, api_secret, auto_reconnect=True)
        self._investor_id = investor_id
        self._market_type = market_type
        #: normalised raw-row dicts awaiting _scan_row (order frames only).
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self._started = False

    def _on_order(self, order: Any) -> None:
        """Vendored-client callback (its dispatch task). Log the frame (masked)
        for observability — this is the evidence a prod order frame arrived —
        then enqueue the normalised raw row for the watch task to _scan_row."""
        raw = _order_model_to_raw_row(order)
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
        """Connect + subscribe the PROD broker channels. Raises on failure —
        the caller (``broker._ensure_ws_order_source``) swallows it to
        poll-only."""
        if self._started:
            return
        await self._client.connect()
        await self._client.subscribe_broker_order_event(
            self._investor_id, self._market_type, on_order_event=self._on_order)
        await self._client.subscribe_broker_position_event(
            self._investor_id, self._market_type,
            on_position_event=self._on_position)
        self._started = True
        log.broker_info(
            "[BROKER] WS order feed subscribed: order.broker.%s.%s / "
            "position.broker.%s.%s (#121 failsafe; poll remains the floor)",
            self._market_type, _mask(self._investor_id),
            self._market_type, _mask(self._investor_id))

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
