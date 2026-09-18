"""#157 — the endpoints DNSE's Trading SDKs guide demonstrates. Tests FIRST.

The guide (developers.dnse.com.vn, "Trading SDKs") walks a user through fifteen calls. Twelve
were already served. Three were not, and they are not obscure corners: buying power is the check
a user makes BEFORE placing, and closing a position is how a derivative user gets flat.

Shapes come from DNSE's published samples, not from what a caller here happens to want:
``dnse-get-ppse.md``, ``dnse-get-positions-position-id.md``,
``dnse-post-positions-position-id-close.md``.

Also pinned here: the order-category matrix from the 2026-08-06 changelog. A one-cancels-other
order is supported on DERIVATIVE and refused on STOCK. The fake accepted it on both, which is the
same error as the amend model corrected in round 1 — a fake more permissive than the venue
teaches the engine a capability that will fail live, and no pin of ours can catch it because the
fake and the pin agree with each other.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue, VenueReject                          # noqa: E402
from venue_http import VenueHTTP                                       # noqa: E402

ACCOUNT = "0001000000"


def _serve(market_type="DERIVATIVE", contract="41I1GA000", last=1980.0):
    bars = [{"timestamp": 1_789_524_000_000, "open": last, "high": last + 5, "low": last - 5,
             "close": last, "volume": 100.0}]
    venue = FakeVenue(symbol=contract, market_type=market_type, last_price=last, seed=7)
    server = VenueHTTP(venue, contract=contract, bars=bars, account_no=ACCOUNT,
                       final_trade_date="2026-12-17",
                       band=(round(last * 1.07, 1), round(last * 0.93, 1))).start()
    return venue, server


@pytest.fixture
def served():
    venue, server = _serve()
    try:
        yield server
    finally:
        server.stop()


def _client(server):
    from pynecore_dnse.client import DNSEClient
    return DNSEClient("fake-venue-key", "fake-venue-secret", base_url=server.base_url)


# --------------------------------------------------------------------------- buying power

def __test_ppse_answers_the_documented_fields__(served):
    """The check a user makes before placing. Fields per DNSE's own sample, plus the two the
    2026-05-28 changelog added."""
    status, body = _client(served).get_ppse(
        ACCOUNT, "DERIVATIVE", "41I1GA000", price=1980.0, loan_package_id=1)

    assert status == 200
    for field in ("qmaxBuy", "qmaxSell", "price", "pp0Buy"):
        assert field in body, f"{field} is documented"
    assert body["price"] == 1980.0, "the answer must be for the price that was ASKED about"


def __test_ppse_buying_power_falls_as_the_price_rises__(served):
    """The discriminating half, and the reason a constant would be useless. Buying power is a
    quantity at a price: the same cash buys fewer contracts as the price goes up. A route
    answering a fixed number regardless of its inputs is the parameter-blind defect again."""
    client = _client(served)

    _, cheap = client.get_ppse(ACCOUNT, "DERIVATIVE", "41I1GA000", price=1000.0, loan_package_id=1)
    _, dear = client.get_ppse(ACCOUNT, "DERIVATIVE", "41I1GA000", price=2000.0, loan_package_id=1)

    assert cheap["qmaxBuy"] > dear["qmaxBuy"], "the same cash must buy fewer at a higher price"


# --------------------------------------------------------------------------- one position

def __test_a_position_can_be_read_by_its_own_id__(served):
    """``get_position_by_id`` is in the guide and had no route. The id must be the one the
    listing gave out, or a caller cannot follow the guide's own two-step."""
    venue = served.venue
    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    venue.feed_print(price=1980.0, volume=10.0)

    client = _client(served)
    _, listing = client.get_positions(ACCOUNT, "DERIVATIVE")
    position_id = listing["positions"][0]["id"]

    status, body = client.get_position_by_id("DERIVATIVE", position_id)

    assert status == 200
    assert str(body["id"]) == str(position_id)
    assert body["openQuantity"] == 1.0
    assert body["symbol"] == "41I1GA000"


def __test_an_unknown_position_id_is_not_found__(served):
    """The discriminating half: a route that invents a position for any id would let a caller
    believe in exposure that does not exist."""
    status, _ = _client(served).get_position_by_id("DERIVATIVE", "no-such-position")

    assert status == 404


def __test_the_position_listing_carries_the_average_close_price__(served):
    """Added by the 2026-05-12 changelog and present in the published sample; the fake omitted
    it."""
    venue = served.venue
    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    venue.feed_print(price=1980.0, volume=10.0)

    _, body = _client(served).get_positions(ACCOUNT, "DERIVATIVE")

    assert "averageClosePrice" in body["positions"][0]


# --------------------------------------------------------------------------- closing out

def __test_closing_a_position_places_the_opposing_order_and_flattens_it__(served):
    """The guide states the mechanic exactly: a close is an order in the OPPOSITE direction, type
    LO, priced at the instrument's ceiling or floor, for the position's open quantity. Serving a
    bare acknowledgement would hide all of that — and the resulting order is the thing a caller
    then has to track."""
    venue = served.venue
    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    venue.feed_print(price=1980.0, volume=10.0)

    client = _client(served)
    _, listing = client.get_positions(ACCOUNT, "DERIVATIVE")
    position_id = listing["positions"][0]["id"]

    status, body = client.close_position(position_id, "DERIVATIVE", "fake-venue-trading-token")

    assert status == 200
    assert body.get("side") == "NS", "closing a long is a SELL"
    assert body.get("orderType") == "LO"
    assert float(body["quantity"]) == 1.0
    assert float(body["price"]) == pytest.approx(round(1980.0 * 0.93, 1)), (
        "a sell to close is priced at the FLOOR so it is marketable")

    venue.feed_print(price=1980.0, volume=10.0)
    _, after = client.get_positions(ACCOUNT, "DERIVATIVE")
    assert after["positions"] == [], "the position must actually be gone once that order fills"


def __test_closing_an_unknown_position_is_not_found__(served):
    status, _ = _client(served).close_position("no-such-position", "DERIVATIVE", "tok")

    assert status == 404


# --------------------------------------------------------------------------- the category matrix

def __test_an_oco_order_is_refused_on_a_stock__():
    """MEASURED against the 2026-08-06 changelog matrix: OCO is supported on DERIVATIVE and NOT
    on STOCK. The fake accepted it on both, so the engine could have been taught a bracket the
    venue would refuse — and every one of our own pins would have agreed with it."""
    venue, server = _serve(market_type="STOCK", contract="HPG", last=21.5)
    try:
        with pytest.raises(VenueReject) as excinfo:
            venue.place(category="OCO", side="sell", qty=100, price=22.0, stop_price=21.0)
        assert "OCO" in str(excinfo.value) or "UNSUPPORTED" in str(excinfo.value)
    finally:
        server.stop()


def __test_a_stop_order_is_still_accepted_on_a_stock__():
    """The discriminating half, straight from the same matrix row: STOCK supports NORMAL and
    STOP. A blanket refusal of conditionals on stocks would satisfy the test above and break the
    stock stop path, which the venue does support."""
    venue, server = _serve(market_type="STOCK", contract="HPG", last=21.5)
    try:
        order = venue.place(category="STOP", side="sell", qty=100, price=21.0, stop_price=21.2)
        assert order["orderStatus"] == "New"
    finally:
        server.stop()


def __test_an_oco_order_is_still_accepted_on_a_derivative__():
    """The other discriminating half: the matrix allows it here, and this is the path the whole
    OCO bracket model depends on."""
    venue, server = _serve()
    try:
        order = venue.place(category="OCO", side="sell", qty=1, price=2000.0, stop_price=1960.0)
        assert order["orderStatus"] == "Activated", "an OCO umbrella is Activated from birth"
    finally:
        server.stop()
