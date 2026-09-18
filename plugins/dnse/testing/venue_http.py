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
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from venue_core import BUY, VenueReject, VenueServerError

#: Hosts that must never be addressed by anything calling itself a fake.
_PRODUCTION_HOSTS = ("dnse.com.vn", "entrade.com.vn")

#: The sandbox publishes 666666 as its OTP constant; this fake accepts the same, so an example
#: that drives the token mint can run without any real code existing anywhere.
_FAKE_OTP = "666666"

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

        if parsed.path == "/registration/send-email-otp":
            # The venue mails a six-digit code. There is no mailbox here, and there must not be
            # one: entering a real OTP is prohibited for every agent, and a fake that invented a
            # credential flow would be teaching exactly the habit that rule exists to prevent.
            # The sandbox's published constant 666666 is what this venue accepts, so an example
            # driving the mint flow can be answered without any secret existing anywhere.
            return self._send(200, {"message": "OTP sent"})

        if parsed.path == "/registration/trading-token":
            payload = self._read_body()
            passcode = str(payload.get("passcode") or "")
            if passcode != _FAKE_OTP:
                # Refused the way the venue refuses, so an example's error path is reachable.
                return self._send(400, {"code": "INVALID_OTP",
                                        "message": f"passcode must be {_FAKE_OTP} on this fake"})
            return self._send(200, {"tradingToken": "fake-venue-trading-token",
                                    "expiredAt": 28_800})

        match = _ORDERS_PATH.match(parsed.path)
        if not match:
            return self._not_found(parsed.path)

        query = parse_qs(parsed.query)
        category = (query.get("orderCategory") or ["NORMAL"])[0]
        payload = self._read_body()
        try:
            order = self.venue.place(
                category=category,
                side=str(payload.get("side", "")).upper() or BUY,
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
            # The venue's documented shape (dnse-get-accounts.md): an investorId plus accounts
            # keyed by "id". This served "accountNo" and no investorId until the SDK examples
            # were read — and broker.py:405 resolves an unconfigured account with
            # accounts[0]["id"], so the plugin would have broken on it too. It never did only
            # because a fake run always pins account_no in config and never resolves. Same
            # latent-shape family as the /price/ohlc defect: built to satisfy the one reader
            # that happened to be exercised.
            return self._send(200, {
                "investorId": "1000000000",
                "accounts": [{
                    "id": self.catalogue["account_no"],
                    "custodyCode": "FAKE",
                    "dealAccount": False,
                    "derivativeAccount": True,
                    "accountName": "FAKE VENUE",
                }],
            })

        if parsed.path.endswith("/balances"):
            # Nested per asset class, as the venue serves it. NOT a flat set of cash fields —
            # that was the shape assumed before the documented sample was read.
            return self._send(200, {
                "stock": {"totalCash": 1_000_000_000, "availableCash": 1_000_000_000,
                          "depositInterest": 0, "totalDebt": 0, "depositFeeAmount": 0,
                          "secureAmount": 0, "orderSecured": 0,
                          "withdrawableCash": 1_000_000_000, "cashDividendReceiving": 0},
                "derivative": {"pendingDepositWithdraw": 0, "remainSecure": 1_000_000_000,
                               "usedSecure": 0, "pendingSecure": 0, "holdTaxAndFee": 0,
                               "totalLoanDebt": 0},
                "bond": {"totalValue": 0},
                "egg": {"totalValue": 0},
            })

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
            # The caller passes marketType and symbol and then uses the package id to place an
            # order for that symbol. A package that does not say what it is FOR cannot be checked
            # against, so the answer names both rather than ignoring them.
            query = parse_qs(parsed.query)
            return self._send(200, {"loanPackages": [{
                "id": 1, "name": "FAKE-MARGIN",
                "symbol": (query.get("symbol") or [self.catalogue["contract"]])[0],
                "marketType": (query.get("marketType") or ["DERIVATIVE"])[0],
                "initialRate": 1.0, "interestRate": 0.0,
            }]})

        if parsed.path.endswith("/positions"):
            # Positions are VENUE-DERIVED from fills (CLAUDE.md), so they are computed from the
            # state machine's filled orders rather than stored separately. `total` is served
            # because the plugin PROVES completeness against it and refuses to conclude from a
            # possibly truncated page (#57/#62) — omitting it would make every read inconclusive.
            wanted = (parse_qs(parsed.query).get("marketType") or ["DERIVATIVE"])[0].upper()
            # This venue replays ONE instrument. Asked about a different market type the honest
            # answer is none, not these rows relabelled: a route that ignores the parameter makes
            # every test that varies it prove the same thing twice.
            rows = self._positions() if wanted == self.venue.market_type.upper() else []
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
            # TradingView-UDF parallel arrays, with `t` in SECONDS — the venue's own shape,
            # measured against production 2026-09-18: the keys are exactly c, h, l, nextTime,
            # o, t, v, and there is no `s` status field at all.
            #
            # This served `{"data": [ {timestamp, open, ...} ]}` until the second pass, and
            # nothing noticed for weeks: FakeVenueBroker overrides download_ohlcv and
            # watch_ohlcv, so no reader of this payload was reached through the normal fake
            # door. Through the other door — the direct-client scripts — it broke two readers,
            # and the quiet one was the dangerous one. `_stop_already_crossed` reads
            # `body.get("c")`, found nothing, and FAILED OPEN with False, so a stop the market
            # had already passed read as not crossed: no exception, no log line, and the branch
            # under test never ran. A fake that makes a safety check answer "no problem" is
            # worse than a fake that crashes.
            return self._send(200, {
                "t": [int(b["timestamp"]) // 1000 for b in bars],
                "o": [float(b["open"]) for b in bars],
                "h": [float(b["high"]) for b in bars],
                "l": [float(b["low"]) for b in bars],
                "c": [float(b["close"]) for b in bars],
                "v": [float(b.get("volume", 0.0)) for b in bars],
                "nextTime": 0,
            })

        if parsed.path.endswith("/orders/history"):
            # Rows under "data", inside an envelope carrying accountNo/total/marketType, and ids
            # DATE-PREFIXED as 20260312_241 (dnse-get-orders-history.md). The date prefix is the
            # reason a cross-day numeric id does not resolve on the cancel endpoint and why the
            # venue tool falls back here for previous-day ids — so the fake reproduces it rather
            # than serving today's bare numeric ids, which would make that whole behaviour
            # untestable offline.
            stamp = self.catalogue.get("history_date", "20260918")
            rows = []
            for order in self.venue.all_orders():
                rows.append({
                    "id": f"{stamp}_{order['id']}",
                    "symbol": order.get("symbol", self.catalogue["contract"]),
                    "side": order.get("side"),
                    "orderType": order.get("orderType", "LO"),
                    "orderStatus": order.get("orderStatus"),
                    "price": order.get("price"),
                    "quantity": order.get("quantity"),
                    "fillQuantity": order.get("fillQuantity", 0.0),
                    "leaveQuantity": max(0.0, float(order.get("quantity") or 0)
                                         - float(order.get("fillQuantity") or 0)),
                    "canceledQuantity": order.get("canceledQuantity", 0.0),
                    "averagePrice": order.get("averagePrice", 0.0),
                    "loanPackageId": 1,
                    "transDate": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}",
                })
            query = parse_qs(parsed.query)
            # from/to are inclusive trading dates. Ignoring them would make every window return
            # the same rows, so a caller narrowing its window would learn nothing from doing so.
            since = (query.get("from") or [""])[0]
            until = (query.get("to") or [""])[0]
            rows = [row for row in rows
                    if (not since or row["transDate"] >= since)
                    and (not until or row["transDate"] <= until)]
            return self._send(200, {
                "accountNo": self.catalogue["account_no"],
                "fillQuantity": 0,
                "total": len(rows), "start": 0, "end": len(rows),
                "marketType": (query.get("marketType") or ["DERIVATIVE"])[0],
                "data": rows,
            })

        if parsed.path.endswith("/secdef"):
            band = self.catalogue["band"]
            # finalTradeDate belongs HERE, on the secdef. broker.py:1344 reads it from
            # self._secdef(...), not from the instruments row, and without it the plugin falls
            # back to a COMPUTED third-Thursday date that can be in the past. securityStatus is
            # "UNSPECIFIED" because that is what the venue actually returns and it is absent
            # from the documented enum — the fake reproduces the venue, not the documentation.
            # A LIST, one row per board — the venue's shape, the plugin's own parser
            # (provider.py:361) and the SDK examples all expect that. It served a bare dict
            # until the examples were read; the plugin tolerated it, so nothing said so.
            #
            # securityGroupId is load-bearing and was missing: classify_market_type answers
            # AUTHORITATIVELY from it and otherwise falls through to the symbol-prefix GUESS,
            # which calls every dated derivative code a STOCK. #119/G1 exists because a guess
            # must never scale a price, so a fake that forces the guess teaches exactly the
            # wrong thing. basicPrice is the reference the examples order at.
            derivative = str(self.catalogue["contract"]).upper().startswith(("VN30F", "41I"))
            basic = round((band[0] + band[1]) / 2, 1)
            return self._send(200, [{
                "marketId": "STO",
                "boardId": (parse_qs(parsed.query).get("boardId") or ["G1"])[0],
                "symbol": self.catalogue["contract"],
                "productGrpId": "STO",
                "securityGroupId": "FU" if derivative else "ST",
                "basicPrice": basic,
                "ceilingPrice": band[0],
                "floorPrice": band[1],
                # "UNSPECIFIED" is what the venue actually returns here and it is absent from
                # the documented enum. The fake reproduces the venue, not the documentation.
                "securityStatus": "UNSPECIFIED",
                "symbolAdminStatusCode": "NRM",
                "symbolTradingMethodStatusCode": "NRM",
                "symbolTradingSanctionStatusCode": "NRM",
                "finalTradeDate": self.catalogue["final_trade_date"],
                "time": self.catalogue.get("secdef_time", "2026-09-18 08:00:00.000"),
            }])

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
                with open(os.environ.get("FAKE_VENUE_DEBUG_LOG",
                         str(Path(__file__).resolve().parents[3]
                             / "workdir" / "output" / "fake_venue_served.log")),
                          "a") as handle:
                    for row in orders:
                        handle.write(f"{book} id={row['id']} status={row['orderStatus']} "
                                     f"filled={row['fillQuantity']} avg={row['averagePrice']} px={row['price']} "
                                     f"side={row['side']} sym={row['symbol']}\n")
            return self._send(200, {"orders": orders, "total": len(orders), "totalPages": 1})

        return self._not_found(parsed.path)

    def do_PUT(self):                                           # noqa: N802
        """Amend. Without this the server returned a default HTML 501 and the staged probe
        crashed at its first amend stage — found by the round-2 checker, not by me."""
        parsed = urlparse(self.path)
        match = _ORDER_PATH.match(parsed.path)
        if not match:
            return self._not_found(parsed.path)
        category = (parse_qs(parsed.query).get("orderCategory") or [None])[0]
        payload = self._read_body()
        try:
            amended = self.venue.amend(match.group("order_id"),
                                       price=_as_float(payload.get("price")),
                                       qty=_as_float(payload.get("quantity")),
                                       category=category)
        except VenueServerError as exc:
            # A 5xx, not a coded rejection: the measured derivative amend-500. The engine must
            # see a server error here, because that is what makes it fall back to its own
            # cancel+replace instead of believing the amend landed.
            return self._send(exc.status, {"message": exc.message})
        except VenueReject as exc:
            return self._reject(exc)
        return self._send(200, amended)

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
