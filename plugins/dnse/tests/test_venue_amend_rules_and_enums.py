"""#157 — the amend rules and market enums DNSE publishes. Tests FIRST.

Two sources, both from the venue's own documentation rather than from our reading of it:

* **FAQ #33**, "Sửa lệnh cơ sở và phái sinh khác nhau như thế nào" — the amend rules differ by
  market, and the difference is not one the fake modelled at all;
* **Enums Dữ liệu thị trường** — the published value sets for ``marketId``, ``productGrpId`` and
  ``securityGroupId``.

PROVENANCE MATTERS HERE, and it is stated on each pin. The amend restrictions below are
DOCUMENTED, not measured. The project rule is that a measurement beats documentation, and there
is a measured fact next door: #117 measured on production that a STOCK amend takes price AND
quantity in one request, which is exactly what the FAQ says for stocks. Nothing has been measured
for the derivative restriction, so it is implemented from the document and labelled so a live
measurement can overturn it without anyone having to guess where the rule came from.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue, VenueReject                          # noqa: E402
from venue_http import VenueHTTP                                       # noqa: E402


def _derivative():
    return FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)


def _stock():
    return FakeVenue(symbol="HPG", market_type="STOCK", last_price=21.5, seed=7)


# --------------------------------------------------------------------------- amend, by market

def __test_a_derivative_amend_cannot_change_price_and_quantity_at_once__():
    """DOCUMENTED (FAQ #33), not measured. A derivative amend changes price OR quantity, never
    both; a stock amend may change both. The fake accepted both on either market, which is the
    permissive direction — the engine would learn a single-request amend the venue refuses, and
    the refusal would arrive live."""
    venue = _derivative()
    order = venue.place(category="NORMAL", side="buy", qty=2, price=1980.0)

    with pytest.raises(VenueReject) as excinfo:
        venue.amend(order["id"], price=1985.0, qty=3)

    assert "price" in str(excinfo.value).lower() or "quantity" in str(excinfo.value).lower()


def __test_a_derivative_amend_of_price_alone_is_accepted__():
    """The discriminating half. A guard that refused every derivative amend would satisfy the
    test above and re-create the exact bug corrected in round 1, where the fake answered 500 for
    every derivative amend and the staged probe parked on a refusal the venue never sends."""
    venue = _derivative()
    order = venue.place(category="NORMAL", side="buy", qty=2, price=1980.0)

    amended = venue.amend(order["id"], price=1985.0)

    assert amended["price"] == 1985.0
    assert amended["id"] == order["id"], "a normal-book derivative amends IN PLACE"


def __test_a_derivative_amend_of_quantity_alone_is_accepted__():
    venue = _derivative()
    order = venue.place(category="NORMAL", side="buy", qty=2, price=1980.0)

    amended = venue.amend(order["id"], qty=3)

    assert amended["quantity"] == 3


def __test_a_stock_amend_may_change_both_at_once__():
    """MEASURED, #117 on production 2026-09-15, and the FAQ agrees: a stock amend is a
    cancel-and-replace, so both fields travel in one request and a NEW id comes back."""
    venue = _stock()
    order = venue.place(category="NORMAL", side="buy", qty=100, price=21.5)

    amended = venue.amend(order["id"], price=21.6, qty=200)

    assert amended["price"] == 21.6
    assert amended["quantity"] == 200
    assert amended["id"] != order["id"], "a stock amend mints a new id (#117)"


def __test_a_derivative_amend_below_the_filled_quantity_is_refused__():
    """DOCUMENTED (FAQ #33): the amended quantity must be GREATER than what has already filled.
    Shrinking an order below its own fills is not a thing the venue permits, and a fake that
    allowed it would let the engine believe in a position size the venue would never produce."""
    venue = _derivative()
    order = venue.place(category="NORMAL", side="buy", qty=3, price=1980.0)
    venue.feed_print(price=1980.0, volume=2.0)
    assert venue.order(order["id"])["fillQuantity"] == 2.0, "fixture must have a partial fill"

    with pytest.raises(VenueReject):
        venue.amend(order["id"], qty=1)


def __test_a_derivative_amend_above_the_filled_quantity_is_accepted__():
    """The discriminating half of the rule above."""
    venue = _derivative()
    order = venue.place(category="NORMAL", side="buy", qty=3, price=1980.0)
    venue.feed_print(price=1980.0, volume=2.0)

    amended = venue.amend(order["id"], qty=5)

    assert amended["quantity"] == 5


# --------------------------------------------------------------------------- published enums

@pytest.fixture
def secdef_of():
    """Read a secdef row from a running venue of the given market type."""
    servers = []

    def _read(market_type, contract):
        bars = [{"timestamp": 1_789_524_000_000, "open": 100.0, "high": 101.0, "low": 99.0,
                 "close": 100.0, "volume": 1.0}]
        venue = FakeVenue(symbol=contract, market_type=market_type, last_price=100.0, seed=7)
        server = VenueHTTP(venue, contract=contract, bars=bars, final_trade_date="2026-12-17",
                           band=(107.0, 93.0)).start()
        servers.append(server)
        from pynecore_dnse.client import DNSEClient
        _, body = DNSEClient("k", "s", base_url=server.base_url).get_security_definition(contract)
        return (body if isinstance(body, list) else [body])[0]

    try:
        yield _read
    finally:
        for server in servers:
            server.stop()


def __test_a_derivative_carries_the_derivative_market_and_product_group__(secdef_of):
    """From the published enum tables: ``marketId`` DVX is "Phái sinh sàn HNX" and
    ``productGrpId`` FIO is "Hợp đồng tương lai Chỉ số" — an index future. The fake stamped STO
    on everything, which is the HOSE *stock* market, so a derivative described itself as a
    HOSE-listed share."""
    row = secdef_of("DERIVATIVE", "41I1GA000")

    assert row["marketId"] == "DVX"
    assert row["productGrpId"] == "FIO"
    assert row["securityGroupId"] == "FU", "FU is the futures group"


def __test_a_stock_carries_the_hose_market_and_product_group__(secdef_of):
    """The discriminating half: STO really is right here, so the fix must not simply swap one
    constant for another."""
    row = secdef_of("STOCK", "HPG")

    assert row["marketId"] == "STO"
    assert row["productGrpId"] == "STO"
    assert row["securityGroupId"] == "ST"


# --------------------------------------------------------------------------- loan packages

def __test_a_stock_account_is_offered_a_cash_and_a_margin_package__(secdef_of):
    """FAQ #31: for a stock order the venue returns at most two packages, a cash one (``type: N``)
    and a margin one (``type: M``). The fake offered exactly one, unlabelled, so a caller could
    not tell which it had been given — and choosing between them is the documented purpose of
    the call."""
    bars = [{"timestamp": 1_789_524_000_000, "open": 21.5, "high": 22.0, "low": 21.0,
             "close": 21.5, "volume": 1.0}]
    venue = FakeVenue(symbol="HPG", market_type="STOCK", last_price=21.5, seed=7)
    server = VenueHTTP(venue, contract="HPG", bars=bars, final_trade_date="2026-12-17",
                       band=(23.0, 20.0)).start()
    try:
        from pynecore_dnse.client import DNSEClient
        _, body = DNSEClient("k", "s", base_url=server.base_url).get_loan_packages(
            "0001000000", "STOCK", symbol="HPG")
    finally:
        server.stop()

    packages = body["loanPackages"]
    types = {package.get("type") for package in packages}
    assert types == {"N", "M"}, "a cash package and a margin package"
    assert packages[0]["type"] == "N", (
        "the cash package comes first: a caller taking packages[0] without reading further must "
        "get the one that borrows nothing")


# --------------------------------------------------------------------------- rate-limit headers

def __test_every_response_carries_the_three_documented_rate_limit_headers__():
    """The error guide tells a client to read ``X-RateLimit-Remaining`` and ``X-RateLimit-Reset``
    on a 429 and wait until the reset. The fake served only Remaining, so the documented wait
    could not be exercised offline at all."""
    bars = [{"timestamp": 1_789_524_000_000, "open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.0, "volume": 1.0}]
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=100.0, seed=7)
    server = VenueHTTP(venue, contract="41I1GA000", bars=bars, final_trade_date="2026-12-17",
                       band=(107.0, 93.0)).start()
    try:
        import urllib.request
        with urllib.request.urlopen(f"{server.base_url}/accounts") as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
    finally:
        server.stop()

    for header in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"):
        assert header in headers, f"{header} is documented in the error guide"
    assert int(headers["x-ratelimit-reset"]) > 0
