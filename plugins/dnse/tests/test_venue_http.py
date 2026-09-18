"""#157 stage C2: the socket adapter's REST half — tests written FIRST.

The adapter serves ONE state machine (``venue_core.FakeVenue``) over HTTP so the REAL plugin
client, the REAL vendored SDK and REAL urllib3 all run unmodified against it. That is the whole
value: a fake reached through the actual wire exercises path building, query encoding, headers
and response parsing, none of which an in-process stub touches.

Only the REST half is reachable this way, and that is a measured constraint rather than a choice.
The vendored WS connection passes an SSL context unconditionally
(``_vendor/dnse/websocket/connection.py:69-74``) and websockets 17.1 raises ``ssl argument is
incompatible with a ws:// URI``, so no local ``ws://`` server can ever be reached by the real
client; the WS side goes through #160's injected client factory instead. REST has no such
problem: the PoolManager's ``cert_reqs`` applies only to ``https``.

The first test is the one that matters most. A fake venue that could be pointed at production,
or a production client that could be pointed at the fake, is worse than no fake at all.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue                                      # noqa: E402
from venue_http import VenueHTTP, ProductionRefused                   # noqa: E402


@pytest.fixture
def served():
    """A venue served over loopback HTTP, torn down after the test."""
    venue = FakeVenue(symbol="41I1G9000", market_type="DERIVATIVE",
                      last_price=1980.0, seed=99)
    with VenueHTTP(venue) as server:
        yield venue, server


# --------------------------------------------------------------------------- safety first

def __test_the_adapter_refuses_to_bind_a_non_loopback_address__():
    """The fake must be unreachable from anywhere but this machine. A fake venue listening on a
    routable interface is an order endpoint with no authentication."""
    venue = FakeVenue(symbol="41I1G9000")
    with pytest.raises(ProductionRefused):
        VenueHTTP(venue, host="0.0.0.0")


def __test_the_adapter_refuses_a_production_looking_base_url__():
    """The other direction of the same hazard: a runner that thinks it is talking to the fake
    while actually addressing DNSE. Refusing by hostname is crude and it is the check that would
    have caught the confusion."""
    venue = FakeVenue(symbol="41I1G9000")
    for host in ("api.dnse.com.vn", "services.entrade.com.vn", "sb-openapi.dnse.com.vn"):
        with pytest.raises(ProductionRefused):
            VenueHTTP.assert_not_production(f"https://{host}")


def __test_the_adapter_accepts_its_own_loopback_url__():
    """The discriminating half: a checker that refused EVERY url would pass the test above and
    make the fake unusable, so the loopback address it actually serves must be accepted."""
    VenueHTTP.assert_not_production("http://127.0.0.1:8899")


# --------------------------------------------------------------------------- the wire

def __test_a_place_through_the_real_client_reaches_the_state_machine__(served):
    """End-to-end over the socket, with the plugin's own client wrapper and the vendored SDK
    doing the signing, path building and parsing. An in-process stub cannot prove any of that."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    status, body = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1975.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")

    assert status in (200, 201), f"place failed: {status} {body}"
    assert str(body["id"]).isdigit(), "a NORMAL order must come back with an integer-shaped id"
    assert venue.order(body["id"])["orderStatus"] == "New", (
        "the order must exist in the SAME state machine the adapter serves")


def __test_a_fill_driven_by_a_print_is_visible_over_the_wire__(served):
    """The venue is driven by prints, and the adapter must expose the result, not a snapshot
    taken at placement."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1980.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")

    venue.feed_print(price=1979.0, volume=5)

    status, detail = client.get_order_detail("0001000000", placed["id"], "DERIVATIVE")
    assert status == 200
    assert detail["orderStatus"] == "Filled"


def __test_an_uncalled_endpoint_answers_404_as_production_does__(served):
    """The executions endpoint answers 404 on this account (CLAUDE.md), and the plugin books at
    cumulative VWAP because of it. A fake that invented an executions payload would make the
    plugin take a path production never gives it."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    status, _ = client.get_execution_detail("0001000000", "12345", "DERIVATIVE")

    assert status == 404


def __test_the_fake_broker_refuses_a_production_endpoint_from_the_config__(tmp_path, monkeypatch):
    """The CONFIG side of the production guard, not only the server side.

    The broker overwrites base_url with its own loopback port moments after start, so a
    production host in the toml would usually be harmless by accident. Harmless by accident is
    not a safety property: a mis-edited config must stop the run.
    """
    from pynecore_dnse.fake_broker import FakeVenueBroker
    from pynecore_dnse.fake_broker import FakeVenueConfig

    broker = FakeVenueBroker.__new__(FakeVenueBroker)
    broker.config = FakeVenueConfig(api_key="k", api_secret="s",
                                     base_url="https://openapi.dnse.com.vn",
                                     ws_url="ws://127.0.0.1:1")
    broker._server = None
    monkeypatch.setenv("FAKE_VENUE_DAY", str(tmp_path / "unused.json"))

    with pytest.raises(ProductionRefused):
        broker._ensure_venue()


def __test_the_fake_broker_accepts_a_loopback_config__(tmp_path, monkeypatch):
    """The discriminating half: a guard that refused every config would pass the test above and
    make the fake unrunnable. A loopback config must get PAST the endpoint check — it then fails
    on the missing day file, which proves the endpoint guard was not what stopped it."""
    from pynecore_dnse.fake_broker import FakeVenueBroker
    from pynecore_dnse.fake_broker import FakeVenueConfig
    from venue_day import MalformedDay

    broker = FakeVenueBroker.__new__(FakeVenueBroker)
    broker.config = FakeVenueConfig(api_key="k", api_secret="s",
                                     base_url="http://127.0.0.1:0", ws_url="ws://127.0.0.1:0")
    broker._server = None
    monkeypatch.setenv("FAKE_VENUE_DAY", str(tmp_path / "missing.json.gz"))

    with pytest.raises((FileNotFoundError, MalformedDay, OSError)):
        broker._ensure_venue()


def __test_a_cancel_refusal_carries_the_venue_code_over_the_wire__(served):
    """The structured reject codes are the contract the engine branches on, so they must survive
    the HTTP round trip rather than collapsing into a bare 400."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1980.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")
    venue.feed_print(price=1979.0, volume=5)          # fills it, so a cancel must be refused

    status, body = client.cancel_order("0001000000", placed["id"], "DERIVATIVE",
                                       "trading-token")

    assert status >= 400
    assert "ORDER_CANCEL_STATUS_REJECTED" in str(body)

def __test_a_wire_round_trip_reads_back_a_sell_as_a_sell__(served):
    """R1. The pin that would have caught the worst defect in this fake.

    The plugin's read funnel is a lookup with a "buy" DEFAULT (broker.py:1134), so a row whose
    side it cannot parse is booked as a BUY and every position sign derived from it is wrong
    while nothing raises. Reading back only id and status — as this file first did — cannot see
    that. So the assertion goes all the way through the plugin's own parser."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NS", "orderType": "LO",
         "price": 2100.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")
    _, detail = client.get_order_detail("0001000000", placed["id"], "DERIVATIVE")

    assert detail["side"] == "NS", "the venue speaks NB/NS on the wire, never buy/sell"
    from pynecore_dnse.broker import _DNSE_TO_SIDE
    assert _DNSE_TO_SIDE.get(detail["side"], "buy") == "sell", (
        "and the plugin's own map must resolve it to sell rather than falling to its buy default")


def __test_the_listing_hides_external_order_id_and_the_detail_shows_it__(served):
    """R3. Measured: externalOrderId is a DETAIL field. A listing that volunteers it lets an
    engine path find the child without the detail read production forces, so a #39-class
    regression could pass offline and fail live."""
    venue, server = served
    oco = venue.place(category="OCO", side="NS", qty=1, price=2100.0, stop_price=1900.0,
                      stop_order_price=1899.8)

    listing = venue.orders(book="STOP")
    detail = venue.order(oco["id"])

    assert all("externalOrderId" not in row for row in listing), "listing must not carry it"
    assert detail["externalOrderId"], "detail must"
    for internal in ("parent_id", "book", "_category"):
        assert all(internal not in row for row in listing), f"{internal} is bookkeeping, not wire"


def __test_an_id_looked_up_on_the_wrong_book_is_not_found__(served):
    """R4. The books are separate, and the venue says so by answering RESOURCE_NOT_FOUND."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1975.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")

    ok, _ = client.get_order_detail("0001000000", placed["id"], "DERIVATIVE",
                                    order_category="NORMAL")
    wrong, _ = client.get_order_detail("0001000000", placed["id"], "DERIVATIVE",
                                       order_category="STOP")

    assert ok == 200, "the right book resolves"
    assert wrong == 404, "the wrong book does not"


def __test_a_filled_row_carries_an_average_price__(served):
    """R5. Without averagePrice the plugin books fill_price None on every fill
    (broker.py:3086), because the executions endpoint 404s on this account."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1980.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")
    venue.feed_print(price=1979.0, volume=5)

    _, detail = client.get_order_detail("0001000000", placed["id"], "DERIVATIVE")

    assert detail["orderStatus"] == "Filled"
    assert detail["averagePrice"] == 1979.0, "filled rows must carry the price they filled at"
    for field in ("symbol", "orderCategory", "orderType", "canceledQuantity"):
        assert field in detail, f"{field} is part of the venue row"


def __test_a_normal_book_derivative_amend_succeeds_IN_PLACE__(served):
    """Live-L1-T06-AmendNormal, PASS 2026-08-14 re-verified 08-17: a NORMAL-book amend on the
    derivative SUCCEEDS at the venue, same id, new price.

    This pin exists because the fake got it WRONG in the other direction first: it answered 500
    for every derivative amend, which made the staged probe park on a refusal the real venue
    would never have sent. A fake that refuses where the venue accepts teaches the engine to
    take a recovery path it does not need."""
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1975.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")

    status, amended = client.put_order("0001000000", placed["id"], "DERIVATIVE",
                                       {"price": 1976.0, "quantity": 1}, "trading-token")

    assert status == 200, "a NORMAL-book derivative amend succeeds"
    assert amended["id"] == placed["id"], "and it is amended IN PLACE, keeping its id"
    assert amended["price"] == 1976.0
    assert amended["orderStatus"] == "New"


def __test_a_conditional_amend_answers_500_whatever_the_asset__(served):
    """Live-L1-T07-AmendConditional500 (#18): the 500 belongs to the CONDITIONAL book, and it is
    why broker.py:2278 routes a conditional modify away from a PUT entirely — a conditional entry
    becomes the plugin's own cancel+replace, a conditional exit becomes a park."""
    venue, server = served
    stop = venue.place(category="STOP", side="NB", qty=1, stop_price=1990.0,
                       price=1990.2, stop_order_price=1990.4)

    from venue_core import VenueServerError
    with pytest.raises(VenueServerError) as excinfo:
        venue.amend(stop["id"], price=1991.0)
    assert excinfo.value.status == 500


def __test_a_stock_amend_returns_a_NEW_id_and_cancels_the_old_one__():
    """#117, measured on prod 2026-09-15: a stock PUT answers 200 with a NEW order id, the venue
    having cancelled the old one itself, which reads Canceled untouched. Anything that keeps
    tracking the OLD id after a stock amend goes blind — the #39 family."""
    venue = FakeVenue(symbol="HPG", market_type="STOCK", last_price=26.5, seed=117)
    original = venue.place(category="NORMAL", side="NB", qty=100, price=26.5)

    replacement = venue.amend(original["id"], price=26.6, qty=200)

    assert replacement["id"] != original["id"], "a stock amend mints a NEW id"
    assert venue.order(original["id"])["orderStatus"] == "Canceled", "and cancels the old one"
    assert replacement["price"] == 26.6 and replacement["quantity"] == 200, (
        "both price and quantity land in one PUT")


def __test_a_delete_on_the_wrong_book_is_refused__(served):
    """R4's cancel half. The books are separate, so a cancel addressed to the wrong one does not
    find the order.

    Note the asymmetry, which is the fake's own and is NOT measured against production: a wrong-
    book DELETE surfaces as 400 (the venue's RESOURCE_NOT_FOUND rejection) while a wrong-book GET
    surfaces as 404. Production's status codes for these two are unmeasured; the CODE is what the
    engine branches on and it is the same in both.
    """
    venue, server = served
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient("key", "secret", base_url=server.base_url)
    _, placed = client.post_order(
        "0001000000", "DERIVATIVE",
        {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
         "price": 1975.0, "quantity": 1, "loanPackageId": 1},
        "trading-token", order_category="NORMAL")

    status, body = client.cancel_order("0001000000", placed["id"], "DERIVATIVE",
                                       "trading-token", order_category="STOP")

    assert status >= 400
    assert "RESOURCE_NOT_FOUND" in str(body)


def __test_the_fake_config_class_is_its_own_so_the_cache_cannot_poison_production__():
    """R6. ensure_config caches on config_cls._ensured and tests it with hasattr, which follows
    inheritance (#165), so whichever class is ensured FIRST answers for its whole hierarchy. The
    fake therefore uses a distinct leaf class: a live dnse_broker run can never be handed the
    fake's loopback endpoints and nonsense token."""
    from pynecore_dnse.fake_broker import FakeVenueConfig, FakeVenueBroker
    from pynecore_dnse.config import DNSEBrokerConfig

    assert FakeVenueBroker.Config is FakeVenueConfig
    assert issubclass(FakeVenueConfig, DNSEBrokerConfig)
    assert "_ensured" not in DNSEBrokerConfig.__dict__, (
        "the fake's config must never leave an _ensured cache on the SHARED broker config class")


def __test_a_missing_tracked_example_raises_rather_than_using_production_defaults__(tmp_path,
                                                                                    monkeypatch):
    """R7. Proceeding with an empty dict would leave the PRODUCTION token path and endpoints in
    place on a run that believes it is driving a fake. Silence here is the one outcome that could
    route a test at the live venue."""
    from pynecore_dnse import fake_broker as fb
    from pynecore_dnse.fake_broker import FakeVenueBroker, FakeVenueConfig

    broker = FakeVenueBroker.__new__(FakeVenueBroker)
    broker.config = FakeVenueConfig(api_key="", api_secret="")
    monkeypatch.setattr(fb, "__file__", str(tmp_path / "nowhere" / "fake_broker.py"))

    with pytest.raises(RuntimeError, match="refusing to build a fake-venue config"):
        broker._repair_config_if_degraded()
