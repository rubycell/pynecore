"""#157 — the endpoints the official SDK examples use must answer in the VENUE's shapes.

The DNSE SDK ships a set of example scripts (``python/examples`` in dnse-tech/openapi-sdk). They
drive the same vendored client this plugin uses, but they read the payloads DIRECTLY rather than
through the plugin, so they are a second door onto the fake — and the first pass of the suite run
already showed what a second door finds.

Two shape defects were found the moment the examples were read, both latent for the same reason
the ``/price/ohlc`` one was: the plugin never exercises them.

* **``/accounts`` served ``accountNo`` where the venue serves ``id``**, and omitted
  ``investorId`` entirely. ``broker.py:405`` reads ``accounts[0]["id"]`` — so the plugin would
  have broken too, except that a fake run always pins ``account_no`` in config and therefore
  never resolves an account. Documented shape: ``docs/dnse-openapi-documentation/dnse-get-accounts.md``
  lines 139-144.
* **``/secdef`` omitted ``basicPrice``**, which is the reference price the examples place their
  order at.

As with the OHLC shape, these tests drive the READERS. The account test goes through the plugin's
own resolution path rather than asserting on a key, because asserting on keys is what let the
previous shape stand.
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
    bars = [{"timestamp": 1_789_524_000_000, "open": 1980.0, "high": 1985.0, "low": 1979.0,
             "close": 1984.0, "volume": 100.0}]
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)
    server = VenueHTTP(venue, contract="41I1GA000", bars=bars,
                       final_trade_date="2026-12-17", band=(2118.6, 1841.4),
                       account_no=ACCOUNT).start()
    try:
        yield server
    finally:
        server.stop()


def _client(server):
    from pynecore_dnse.client import DNSEClient
    return DNSEClient("fake-venue-key", "fake-venue-secret", base_url=server.base_url)


# --------------------------------------------------------------------------- /accounts

def __test_the_plugin_can_resolve_its_account_from_the_fake__(served):
    """The reader that would have broken. ``broker.py`` resolves an unconfigured account with
    ``accounts[0]["id"]``; the fake served ``accountNo``, so this path could never have worked.
    It never failed in practice only because a fake run always pins ``account_no`` in config —
    the same kind of luck that hid the OHLC shape."""
    from pynecore_dnse.broker import DNSEBroker

    broker = DNSEBroker.__new__(DNSEBroker)
    broker._client = _client(served)
    broker._account_no = ""

    status, body = broker.client.get_accounts()

    assert status == 200
    accounts = body.get("accounts") or []
    assert accounts, "the venue always answers at least one account"
    assert accounts[0]["id"] == ACCOUNT, "broker.py reads ['id']; the venue serves 'id'"


def __test_the_accounts_payload_carries_the_fields_the_examples_print__(served):
    """``portfolio-check.py`` prints the investor id and each account's deal/derivative flags.
    They are documented fields, so a fake that omits them makes the example print ``None`` and
    look broken."""
    _, body = _client(served).get_accounts()

    assert body.get("investorId"), "documented at dnse-get-accounts.md:139"
    account = body["accounts"][0]
    assert isinstance(account.get("dealAccount"), bool)
    assert isinstance(account.get("derivativeAccount"), bool)


# --------------------------------------------------------------------------- /secdef

def __test_the_secdef_carries_a_basic_price_to_order_at__(served):
    """``place-a-trade.py`` reads ``basicPrice`` and orders at it. Without the field the example
    dies on a KeyError before it places anything."""
    _, body = _client(served).get_security_definition("41I1GA000")

    assert isinstance(body, list), (
        "the venue answers a LIST, one row per board — dnse-get-symbol-secdef.md, and the "
        "plugin's own parser says so at provider.py:361")
    row = body[0]
    assert row["basicPrice"] > 0
    assert row["floorPrice"] <= row["basicPrice"] <= row["ceilingPrice"], (
        "a reference price outside its own band is not a price any order can use")
    assert row.get("boardId"), "each row IS a board; it must say which"


def __test_the_secdef_still_carries_the_final_trade_date__(served):
    """The regression guard on the field the PLUGIN needs. #118's GTD clamp reads it here, and a
    reshape that served the examples while dropping this would refuse every conditional."""
    _, body = _client(served).get_security_definition("41I1GA000")

    row = (body if isinstance(body, list) else [body])[0]
    assert row["finalTradeDate"] == "2026-12-17"


def __test_the_secdef_classifies_the_instrument_authoritatively__(served):
    """``securityGroupId`` is how ``classify_market_type`` answers from a SOURCE instead of from
    the symbol-prefix GUESS (provider.py:449-453). The fake omitted it, so every classification
    against the fake fell through to that guess — which answers STOCK for any dated derivative
    contract code. #119/G1 exists precisely because a guess must never be allowed to scale a
    price, so a fake that forces the guess is teaching the wrong lesson at the wrong place.
    """
    from pynecore_dnse.provider import DNSEConfig, DNSEProvider

    provider = DNSEProvider(symbol="41I1GA000", timeframe="1",
                            config=DNSEConfig(api_key="k", api_secret="s",
                                              base_url=served.base_url))
    market_type, authoritative = provider.classify_market_type("41I1GA000")

    assert market_type == "DERIVATIVE"
    assert authoritative is True, "the venue said so; this must not be the prefix guess"


# --------------------------------------------------------------------------- balances

def __test_balances_answers_the_documented_per_asset_class_shape__(served):
    """The portfolio example calls ``get_balances`` per account; the fake had no such route, so
    it printed a 404 body for every account.

    The shape is NOT a flat set of cash fields. That was this test's first draft, and it was my
    assumption rather than DNSE's spec. The venue nests by asset class —
    ``{"stock": {...}, "derivative": {...}, "bond": {...}, "egg": {...}}``, documented in
    ``dnse-get-account-balances.md``. A fake built to a guessed shape teaches every reader the
    wrong one, which is the failure this suite keeps finding.
    """
    status, body = _client(served).get_balances(ACCOUNT)

    assert status == 200
    assert set(body) >= {"stock", "derivative"}, "the venue nests balances by asset class"
    for field in ("totalCash", "availableCash", "withdrawableCash"):
        assert field in body["stock"], f"stock.{field} is documented"
    for field in ("remainSecure", "usedSecure"):
        assert field in body["derivative"], f"derivative.{field} is documented"


# --------------------------------------------------------------------------- order history

def __test_order_history_answers_rows_under_data__(served):
    """``order-history.py`` reads ``json.loads(body)["data"]``. Production serves history rows
    under ``data``, date-prefixed (CLAUDE.md), which is also why the venue tool falls back to it
    for previous-day ids."""
    venue = served.venue
    placed = venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)

    status, body = _client(served).get_order_history(
        ACCOUNT, market_type="DERIVATIVE", from_date="2026-09-01", to_date="2026-09-30")

    assert status == 200
    rows = body.get("data")
    assert isinstance(rows, list), "the examples index ['data']"
    for field in ("accountNo", "total", "marketType"):
        assert field in body, f"the documented envelope carries {field}"
    assert any(str(placed["id"]) in str(row.get("id")) for row in rows), (
        "an order placed on this venue must appear in its own history")
    assert all("_" in str(row["id"]) for row in rows), (
        "history ids are DATE-PREFIXED, as in 20260312_241. That is why a cross-day numeric id "
        "does not resolve on the cancel endpoint, and why the venue tool falls back to history "
        "for previous-day ids.")


def __test_order_history_is_not_a_second_copy_of_the_live_book__(served):
    """The discriminating half. History that simply mirrors the open-order book would satisfy the
    test above and tell nobody anything: the point of history is that it still holds an order
    after that order is gone from the book."""
    venue = served.venue
    placed = venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    venue.cancel(placed["id"])

    _, body = _client(served).get_order_history(
        ACCOUNT, market_type="DERIVATIVE", from_date="2026-09-01", to_date="2026-09-30")

    rows = body.get("data") or []
    assert any(str(placed["id"]) in str(row.get("id")) for row in rows), (
        "a cancelled order must still appear in history")
