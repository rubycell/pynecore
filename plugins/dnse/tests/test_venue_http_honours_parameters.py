"""#157 — the fake must CONDITION its answers on what was asked. Tests FIRST.

A route that ignores a parameter the caller sent is the same class of defect as a route that
serves the wrong shape: it answers, the answer looks well-formed, and it is simply not an answer
to the question. The shape defects found this week were caught because a reader choked on them.
An ignored parameter chokes nobody — it just quietly makes every test that varies that parameter
prove the same thing twice.

Four parameters the SDK examples actually send, each pinned by asking two DIFFERENT questions and
requiring two different answers. One question can never show that a parameter was read.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue                                       # noqa: E402
from venue_http import VenueHTTP                                       # noqa: E402

ACCOUNT = "0001000000"


@pytest.fixture
def served():
    bars = [{"timestamp": 1_789_524_000_000 + index * 60_000, "open": 1980.0, "high": 1985.0,
             "low": 1979.0, "close": 1984.0, "volume": 100.0} for index in range(3)]
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)
    server = VenueHTTP(venue, contract="41I1GA000", bars=bars, account_no=ACCOUNT,
                       final_trade_date="2026-12-17", band=(2118.6, 1841.4)).start()
    try:
        yield server
    finally:
        server.stop()


def _client(server):
    from pynecore_dnse.client import DNSEClient
    return DNSEClient("fake-venue-key", "fake-venue-secret", base_url=server.base_url)


# --------------------------------------------------------------------------- board_id on secdef

def __test_the_secdef_answers_for_the_board_that_was_asked_for__(served):
    """The examples pass ``board_id="G1"`` (the round-lot board). A venue serves one row per
    board and they differ — an odd-lot board has its own rules — so a fake that returns the same
    row whatever is asked makes board selection untestable."""
    _, g1 = _client(served).get_security_definition("41I1GA000", board_id="G1")
    _, g4 = _client(served).get_security_definition("41I1GA000", board_id="G4")

    assert g1[0]["boardId"] == "G1"
    assert g4[0]["boardId"] == "G4", "the answer must name the board that was requested"


def __test_the_secdef_without_a_board_still_answers_the_default_board__(served):
    """The discriminating half: the plugin does NOT pass a board, and it must keep working."""
    _, body = _client(served).get_security_definition("41I1GA000")

    assert body[0]["boardId"] == "G1", "no board asked for means the round-lot board"


# --------------------------------------------------------------------------- market_type

def __test_positions_answer_for_the_market_type_that_was_asked_for__(served):
    """``order-history.py`` asks for STOCK positions. This venue replays a derivative contract,
    so the honest STOCK answer is none — not the derivative rows relabelled."""
    venue = served.venue
    order = venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    venue.feed_print(price=1980.0, volume=10.0)
    assert venue.order(order["id"])["orderStatus"] == "Filled", "fixture must produce a position"

    _, derivative = _client(served).get_positions(ACCOUNT, "DERIVATIVE")
    _, stock = _client(served).get_positions(ACCOUNT, "STOCK")

    assert derivative["positions"], "the contract replayed here IS a derivative"
    assert stock["positions"] == [], (
        "a STOCK query must not be answered with derivative positions")


def __test_loan_packages_answer_for_the_symbol_that_was_asked_for__(served):
    """The examples pass both ``market_type`` and ``symbol``, then use the package id to place an
    order. A package that does not name the symbol it is for cannot be checked against."""
    _, body = _client(served).get_loan_packages(ACCOUNT, "DERIVATIVE", symbol="41I1GA000")

    package = body["loanPackages"][0]
    assert package["symbol"] == "41I1GA000"
    assert package["marketType"] == "DERIVATIVE"


# --------------------------------------------------------------------------- history dates

def __test_order_history_honours_the_requested_date_window__(served):
    """``order-history.py`` asks for the last seven days. A route that ignores ``from``/``to``
    returns the same rows for every window, so no test that varies the window proves anything."""
    venue = served.venue
    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    client = _client(served)

    _, inside = client.get_order_history(ACCOUNT, market_type="DERIVATIVE",
                                         from_date="2026-09-01", to_date="2026-09-30")
    _, outside = client.get_order_history(ACCOUNT, market_type="DERIVATIVE",
                                          from_date="2026-01-01", to_date="2026-01-31")

    assert inside["data"], "the order was placed inside this window"
    assert outside["data"] == [], "a window the order falls outside must not return it"
    assert outside["total"] == 0, "the envelope's total must agree with its own rows"


def __test_order_history_honours_page_size_and_page_index__(served):
    """The plugin walks this endpoint with ``page_size=200`` and a rising ``page_index``, and
    proves completeness by comparing the rows it has accumulated against ``total``
    (broker.py:1946-1965).

    Serving every row on every page did not break that loop — it terminated on the first call,
    because ``total`` equalled what was served. The cost was narrower and worse: the MULTI-PAGE
    branch of a completeness-critical loop could not be exercised offline at all. So this pins
    pages that differ, and a ``total`` that describes the whole result rather than the page.
    """
    venue = served.venue
    for _ in range(5):
        venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    client = _client(served)

    _, first = client.get_order_history(ACCOUNT, market_type="DERIVATIVE",
                                        page_size=2, page_index=0)
    _, second = client.get_order_history(ACCOUNT, market_type="DERIVATIVE",
                                         page_size=2, page_index=1)

    assert len(first["data"]) == 2, "a page must hold at most pageSize rows"
    assert first["total"] == 5, "total describes the whole result, not the page"
    assert [row["id"] for row in second["data"]] != [row["id"] for row in first["data"]], (
        "page 1 must not be page 0 again — that is what made the paging loop untestable")


def __test_a_page_past_the_end_is_empty_rather_than_wrapping__(served):
    """The discriminating half. A route that clamped an out-of-range page back to the last one
    would satisfy the test above and make the loop's termination condition unreachable."""
    venue = served.venue
    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)

    _, body = _client(served).get_order_history(ACCOUNT, market_type="DERIVATIVE",
                                                page_size=10, page_index=9)

    assert body["data"] == []
    assert body["total"] == 1, "the result is still one row; this page just holds none of it"
