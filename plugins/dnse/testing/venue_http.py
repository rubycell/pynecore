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
                side=_SIDE.get(str(payload.get("side", "")).upper(), "buy"),
                qty=float(payload.get("quantity") or 0),
                price=_as_float(payload.get("price")),
                stop_price=_as_float(payload.get("stopPrice")),
            )
        except VenueReject as exc:
            return self._reject(exc)
        return self._send(200, order)

    def do_GET(self):                                           # noqa: N802
        parsed = urlparse(self.path)

        if _EXEC_PATH.match(parsed.path):
            # Production answers 404 for executions on this account (CLAUDE.md), and the plugin
            # books at cumulative VWAP because of it. Inventing a payload here would send the
            # plugin down a path production never gives it.
            return self._send(404, {"code": "RESOURCE_NOT_FOUND", "message": "no executions"})

        match = _ORDER_PATH.match(parsed.path)
        if match:
            found = self.venue.order(match.group("order_id"))
            if found is None:
                return self._send(404, {"code": "RESOURCE_NOT_FOUND",
                                        "message": match.group("order_id")})
            return self._send(200, found)

        if _ORDERS_PATH.match(parsed.path):
            query = parse_qs(parsed.query)
            category = (query.get("orderCategory") or ["NORMAL"])[0]
            book = "STOP" if category in ("STOP", "OCO") else "NORMAL"
            orders = self.venue.orders(book=book)
            return self._send(200, {"orders": orders, "total": len(orders)})

        return self._not_found(parsed.path)

    def do_DELETE(self):                                        # noqa: N802
        parsed = urlparse(self.path)
        match = _ORDER_PATH.match(parsed.path)
        if not match:
            return self._not_found(parsed.path)
        try:
            cancelled = self.venue.cancel(match.group("order_id"))
        except VenueReject as exc:
            return self._reject(exc)
        return self._send(200, cancelled)

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

    def __init__(self, venue, *, host: str = "127.0.0.1", port: int = 0):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ProductionRefused(
                f"refusing to bind {host!r}: the fake venue serves loopback only. A fake bound "
                f"to a routable interface is an order endpoint with no authentication.")
        self.venue = venue
        handler = type("_BoundHandler", (_Handler,), {"venue": venue})
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
