"""TESTING ONLY (#157) stage C2: the socket adapter — one venue state machine, served over HTTP.

This is the SECOND adapter over :mod:`venue_core`, not a second venue. The in-process adapter is
the state machine used directly; this one puts it behind a real socket so the REAL plugin client,
the REAL vendored SDK and REAL urllib3 run unmodified against it. That is what an in-process stub
cannot prove: path building, query encoding, headers, status codes and response parsing all
execute here.

**REST only, and that is measured rather than chosen.** The vendored WS connection passes an SSL
context unconditionally (``_vendor/dnse/websocket/connection.py:69-74``) and websockets 17.1
raises ``ssl argument is incompatible with a ws:// URI``, so no local ``ws://`` server can ever
be reached by the real client — which is also why ``fake_dnse_ws.py`` stopped working the day the
SDK was vendored. The WS side goes through #160's injected client factory instead. REST has no
equivalent problem: the PoolManager's ``cert_reqs`` applies only to ``https``.

**Two safety rules, both enforced rather than documented.** The server binds loopback only, so it
is never an unauthenticated order endpoint on a routable interface. And
:meth:`VenueHTTP.assert_not_production` refuses a production-looking base url, so a runner cannot
believe it is driving the fake while addressing DNSE.
"""
from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from venue_core import VenueReject

#: Hosts that must never be addressed by anything calling itself a fake.
_PRODUCTION_HOSTS = ("dnse.com.vn", "entrade.com.vn")

#: Wire side codes (DNSE) to the state machine's own vocabulary.
_SIDE = {"NB": "buy", "NS": "sell", "BUY": "buy", "SELL": "sell"}

_ORDER_PATH = re.compile(r"^/accounts/(?P<account>[^/]+)/orders/(?P<order_id>[^/]+)$")
_ORDERS_PATH = re.compile(r"^/accounts/(?P<account>[^/]+)/orders$")
_EXEC_PATH = re.compile(r"^/accounts/(?P<account>[^/]+)/executions/(?P<order_id>[^/]+)$")


class ProductionRefused(RuntimeError):
    """Raised rather than risk the fake and production being confused for one another."""


def _default_final_trade_date() -> str:
    """A final trade date comfortably in the REAL future.

    Deliberately not derived from the replayed day. The plugin computes GTD from
    ``datetime.now()`` and clamps it into ``[next open day, final trade date]``, so a date taken
    from a historical session would be in the past and every conditional would be refused with a
    venue error that has nothing to do with the behaviour under test.
    """
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(days=90)).strftime("%Y-%m-%d")


class _Handler(BaseHTTPRequestHandler):
    """Translates HTTP into state-machine calls. Holds no venue state of its own."""

    venue = None  # injected per server instance

    # -------------------------------------------------------------- plumbing

    def log_message(self, *args):          # noqa: D401 - silence the default stderr spam
        """Quiet: a test suite's output is signal, and this server would drown it."""

    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Served because the engine's 429 path can only ever be exercised here: no 429 has been
        # recorded live, and the SDK drops headers, so this is where that gap gets closed.
        self.send_header("X-RateLimit-Remaining", "9999")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except json.JSONDecodeError:
            return {}

    def _reject(self, exc: VenueReject) -> None:
        """Structured reject codes are the contract the engine branches on, so the CODE travels
        rather than collapsing into a bare 400."""
        self._send(400, {"code": exc.code, "message": exc.message or exc.code})

    # -------------------------------------------------------------- routes

    def do_POST(self):                                          # noqa: N802
        parsed = urlparse(self.path)
        match = _ORDERS_PATH.match(parsed.path)
        if not match:
            return self._not_found(parsed.path)

        query = parse_qs(parsed.query)
        category = (query.get("orderCategory") or ["NORMAL"])[0]
        payload = self._read_body()
        try:
            order = self.venue.place(
                category=category,
                side=str(payload.get("side", "")).upper() or "NB",
                qty=float(payload.get("quantity") or 0),
                price=_as_float(payload.get("price")),
                stop_price=_as_float(payload.get("stopPrice")),
                # R2: the plugin computes this THROUGH the trigger and sends it
                # (broker.py:1496). Discarding it makes the gap-through-unfilled case
                # unreachable, which is the case _stop_fill_price exists for.
                stop_order_price=_as_float(payload.get("stopOrderPrice")),
            )
        except VenueReject as exc:
            return self._reject(exc)
        return self._send(200, order)

    def do_GET(self):                                           # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/accounts":
            return self._send(200, {"accounts": [{"accountNo": self.catalogue["account_no"],
                                                  "custodyCode": "FAKE"}]})

        if parsed.path == "/market/instruments":
            # The instrument catalogue drives resolve_contract, the monthly roll (#113) and the
            # GTD clamp (#118). finalTradeDate is served from the catalogue rather than computed,
            # and the value served is RECORDED (see VenueHTTP.served_final_trade_date) because
            # the plugin computes GTD from the REAL clock (broker.py:1244) while the day being
            # replayed may be historical. A mismatch there refuses every conditional before the
            # fake is ever reached, which looks like a fake bug and is not one.
            # The plugin reads rows from "data" and matches on symbolType (provider.py:285-288),
            # so that is the shape served. A first attempt used an "instruments" key: the run
            # still worked, but the plugin found no finalTradeDate and fell back to a COMPUTED
            # third-Thursday date that was PAST — the exact silent degradation the GTD acceptance
            # line exists to prevent, and it was visible only in a log line, not in a failure.
            row = {"symbolType": "VN30F1M", "symbol": self.catalogue["contract"],
                   "marketType": "DERIVATIVE",
                   "finalTradeDate": self.catalogue["final_trade_date"]}
            return self._send(200, {"data": [row], "total": 1})

        if parsed.path.endswith("/loan-packages"):
            return self._send(200, {"loanPackages": [{"id": 1, "name": "FAKE-MARGIN"}]})

        if parsed.path.endswith("/positions"):
            # Positions are VENUE-DERIVED from fills (CLAUDE.md), so they are computed from the
            # state machine's filled orders rather than stored separately. `total` is served
            # because the plugin PROVES completeness against it and refuses to conclude from a
            # possibly truncated page (#57/#62) — omitting it would make every read inconclusive.
            rows = self._positions()
            return self._send(200, {"positions": rows, "total": len(rows)})

        if parsed.path == "/price/ohlc":
            # CLAMPED to the replay position. A history endpoint that serves bars the replay has
            # not reached yet is serving the FUTURE as history: warmup would then consume the
            # whole day and the live stream would have nothing left to deliver. The cursor is
            # advanced by the broker as it hands each bar to the engine; before the first bar it
            # is None and the full day is served, which is what a cold history read expects.
            cursor = self.catalogue.get("replay_cursor")
            bars = self.catalogue["bars"]
            if cursor is not None:
                bars = [b for b in bars if b["timestamp"] <= cursor]
            return self._send(200, {"data": bars})

        if parsed.path.endswith("/secdef"):
            band = self.catalogue["band"]
            # finalTradeDate belongs HERE, on the secdef. broker.py:1344 reads it from
            # self._secdef(...), not from the instruments row, and without it the plugin falls
            # back to a COMPUTED third-Thursday date that can be in the past. securityStatus is
            # "UNSPECIFIED" because that is what the venue actually returns and it is absent
            # from the documented enum — the fake reproduces the venue, not the documentation.
            return self._send(200, {"symbol": self.catalogue["contract"],
                                    "ceilingPrice": band[0], "floorPrice": band[1],
                                    "securityStatus": "UNSPECIFIED",
                                    "finalTradeDate": self.catalogue["final_trade_date"]})

        if _EXEC_PATH.match(parsed.path):
            # Production answers 404 for executions on this account (CLAUDE.md), and the plugin
            # books at cumulative VWAP because of it. Inventing a payload here would send the
            # plugin down a path production never gives it.
            return self._send(404, {"code": "RESOURCE_NOT_FOUND", "message": "no executions"})

        match = _ORDER_PATH.match(parsed.path)
        if match:
            category = (parse_qs(parsed.query).get("orderCategory") or [None])[0]
            found = self.venue.order(match.group("order_id"), category=category)
            if found is None:
                return self._send(404, {"code": "RESOURCE_NOT_FOUND",
                                        "message": match.group("order_id")})
            return self._send(200, found)

        if _ORDERS_PATH.match(parsed.path):
            query = parse_qs(parsed.query)
            category = (query.get("orderCategory") or ["NORMAL"])[0]
            book = "STOP" if category in ("STOP", "OCO") else "NORMAL"
            orders = self.venue.orders(book=book)
            # totalPages, not total. The poll proves the page is COMPLETE from totalPages
            # (broker.py:2598, book_page_count) and discards a read it cannot prove — measured
            # 2026-09-18: serving only "total" made every book read unprovable, so the engine
            # never saw a fill the venue had already booked, and an order sat filled-but-unseen
            # for the whole run. The completeness discipline is the plugin being careful; the
            # fake has to answer it in the field it actually reads.
            if os.environ.get("FAKE_VENUE_DEBUG"):
                # What the adapter SERVED to the poll, so it can be compared with what the
                # engine did next. Adapter-side only: it proves or clears the fake's half.
                with open("/tmp/fake_venue_served.log", "a") as handle:
                    for row in orders:
                        handle.write(f"{book} id={row['id']} status={row['orderStatus']} "
                                     f"filled={row['fillQuantity']} avg={row['averagePrice']} px={row['price']} "
                                     f"side={row['side']} sym={row['symbol']}\n")
            return self._send(200, {"orders": orders, "total": len(orders), "totalPages": 1})

        return self._not_found(parsed.path)

    def do_DELETE(self):                                        # noqa: N802
        parsed = urlparse(self.path)
        match = _ORDER_PATH.match(parsed.path)
        if not match:
            return self._not_found(parsed.path)
        category = (parse_qs(parsed.query).get("orderCategory") or [None])[0]
        try:
            cancelled = self.venue.cancel(match.group("order_id"), category=category)
        except VenueReject as exc:
            return self._reject(exc)
        return self._send(200, cancelled)

    def _positions(self) -> list[dict]:
        """Net position per symbol, derived from filled NORMAL orders.

        The venue creates positions from fills; we create orders. So this reads the fills rather
        than tracking a position object, and a flat account legitimately returns an EMPTY list —
        which the engine treats as proof of absence, not as a failed read.
        """
        net = 0.0
        for order in self.venue.orders(book="NORMAL"):
            filled = float(order.get("fillQuantity") or 0)
            if filled:
                net += filled if order["side"] == "NB" else -filled
        if not net:
            return []
        return [{
            "id": "900000000000001",
            "symbol": self.catalogue["contract"],
            "side": "NB" if net > 0 else "NS",
            "status": "OPEN",
            "accumulateQuantity": abs(net), "closedQuantity": 0.0,
            "openQuantity": abs(net), "tradeQuantity": abs(net), "overNightQuantity": 0.0,
        }]

    def _not_found(self, path: str):
        """Anything the plugin does not call answers 404 rather than a plausible fiction.

        The stage A inventory found 31 SDK paths of which the plugin calls 13. Serving a made-up
        response for the other 18 would pin shapes nothing produces.
        """
        return self._send(404, {"code": "RESOURCE_NOT_FOUND", "message": path})


def _as_float(value):
    return None if value is None else float(value)


class VenueHTTP:
    """Serves one :class:`~venue_core.FakeVenue` over loopback HTTP."""

    def __init__(self, venue, *, host: str = "127.0.0.1", port: int = 0,
                 contract: str = "41I1G9000", account_no: str = "0001000000",
                 bars: list | None = None, band: tuple[float, float] | None = None,
                 final_trade_date: str | None = None):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ProductionRefused(
                f"refusing to bind {host!r}: the fake venue serves loopback only. A fake bound "
                f"to a routable interface is an order endpoint with no authentication.")
        self.venue = venue
        # The GTD clamp reads the REAL clock (broker.py:1244) and compares it against the
        # catalogue's finalTradeDate, so a historical replay must still serve a date in the
        # real future or every conditional is refused before the fake is reached. The served
        # value is recorded on the instance so a run can state which date it used rather than
        # leaving a reader to infer it.
        self.served_final_trade_date = final_trade_date or _default_final_trade_date()
        self.catalogue = {
            "contract": contract,
            "account_no": account_no,
            "bars": bars or [],
            "band": band or (2200.0, 1800.0),
            "final_trade_date": self.served_final_trade_date,
            # Advanced by the broker as each bar is handed to the engine; see the
            # /price/ohlc clamp above.
            "replay_cursor": None,
        }
        handler = type("_BoundHandler", (_Handler,),
                       {"venue": venue, "catalogue": self.catalogue})
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None
        self.host, self.port = self._server.server_address[:2]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def assert_not_production(base_url: str) -> None:
        """Refuse a production-looking endpoint.

        Crude by design: a hostname check is what would actually have caught a runner that
        believed it was driving the fake while addressing DNSE.
        """
        host = (urlparse(base_url).hostname or "").lower()
        if any(host == bad or host.endswith("." + bad) for bad in _PRODUCTION_HOSTS):
            raise ProductionRefused(
                f"{base_url!r} addresses production ({host}). The fake venue must never be "
                f"pointed at DNSE, and a runner must never think DNSE is the fake.")

    def start(self) -> "VenueHTTP":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "VenueHTTP":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
