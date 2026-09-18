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
