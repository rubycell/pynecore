"""#119 repro — a STOCK order's WIRE price must be in ĐỒNG, not thousands.

Measured live on the prod order book 2026-09-14 (HPG, continuous session,
book bid 21.3 / offer 21.35, secdef floor 19.85 / ceiling 22.75):

    price = 21.0    -> HTTP 400 PRICE_MUST_GREATER_THAN_OR_EQUAL_TO_FLOOR_PRICE
    price = 21000   -> HTTP 200, accepted

So DNSE's stock ORDER api takes đồng, while every DNSE stock PRICE FEED
(``/price/ohlc``, ``/price/{symbol}/quotes``, ``/price/{symbol}/secdef``)
speaks thousands — documented (``dnse-get-ohlc-history.md`` ACB o=23.8,
``dnse-get-quotes.md`` ACB bid 101, ``dnse-get-symbol-secdef.md`` HPG
floor 87.6) and visible in the tracked dataset (``dnse_HPG_1D.ohlcv``
close=21.7). The engine therefore hands ``broker._place`` a price in
thousands and ``broker.py:983`` puts it on the wire unchanged
(``"price": round(float(price), 1)`` — no ``market_type`` branch, no unit
conversion), so the venue floor-checks ~21.3 against 19,850 đồng and
refuses. DERIVATIVES are unaffected: index points are the venue's own unit
there (docs: ``price: 1990`` for 41I1G9000; L0 gate green).

A/B are RED on the unmodified tree (the plugin sends 21.3 / 21.3+stopPrice).
C is the GREEN control proving the derivative path must NOT be rescaled.

Same fake-client seam as ``test_journal_wiring.py`` / ``test_cancel_disposition.py``
— no live venue, no test hook in plugin code.
"""
import asyncio

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import DispatchEnvelope, EntryIntent, OrderType

#: HPG secdef as the venue serves it — THOUSANDS (measured 2026-09-14:
#: floor 19.85 / ceiling 22.75); ``securityGroupId="ST"`` makes
#: ``provider.market_type`` classify STOCK.
_STOCK_SECDEF = [{"symbol": "HPG", "securityGroupId": "ST", "basicPrice": "21.3",
                  "ceilingPrice": "22.75", "floorPrice": "19.85"}]
#: VN30F1M's dated contract, in index points.
_DERIV_SECDEF = [{"symbol": "41I1G9000", "securityGroupId": "FU",
                  "ceilingPrice": "2200", "floorPrice": "1900"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})
_PLACED_STOCK = (201, {"id": "437346", "symbol": "HPG", "side": "NB",
                       "quantity": 100, "orderStatus": "New", "price": 21300})
_PLACED_DERIV = (201, {"id": "437347", "symbol": "41I1G9000", "side": "NB",
                       "quantity": 1, "orderStatus": "New", "price": 2058.6})

#: intent price as the ENGINE sees it (bars/quotes are thousands) and the
#: đồng the venue needs. 21.3 -> 21,300 đ, a valid HOSE 50 đ tick in the
#: 10,000-49,950 band.
_INTENT_THOUSANDS = 21.3
_WIRE_DONG = 21300.0


def _broker(fake_client, tmp_path, symbol, secdef, **client_responses):
    responses = {"get_security_definition": (200, secdef),
                 "get_loan_packages": _LOAN_OK,
                 "get_instruments": (200, {"data": [
                     {"symbolType": "VN30F1M", "symbol": "41I1G9000"}]})}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"),
        # #119/G2: live STOCK placement is opt-in until a hand-verified real
        # stock fill confirms the readback units. These tests are about the
        # WIRE UNIT, not the enablement policy, so they opt in explicitly —
        # the gate itself is pinned by ``test_stock_price_guards.py``.
        enable_stock_orders=True)
    instance = broker.DNSEBroker(symbol=symbol, timeframe="15", config=config)
    instance._client = fake_client(**responses)
    return instance


def _entry_envelope(symbol, *, limit=None, stop=None, qty=100, pine_id="L"):
    return DispatchEnvelope(
        intent=EntryIntent(pine_id=pine_id, symbol=symbol, side="buy", qty=qty,
                           order_type=OrderType.STOP if stop else OrderType.LIMIT,
                           limit=limit, stop=stop),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)


def _payload(instance):
    """The body of the single ``post_order`` the place made."""
    posts = [c for c in instance._client.calls if c[0] == "post_order"]
    assert len(posts) == 1, f"expected exactly one post_order, got {len(posts)}"
    return posts[0][1][2]          # post_order(account, market_type, payload, tok, ...)


# --- A (RED): a STOCK limit entry must go on the wire in đồng ---------------

def __test_stock_limit_entry_wire_price_is_dong__(fake_client, tmp_path):
    """Engine intent 21.3 (thousands, the unit of every DNSE stock feed) must
    reach the venue as 21300 đồng. Today ``_place`` sends 21.3 and prod answers
    400 PRICE_MUST_GREATER_THAN_OR_EQUAL_TO_FLOOR_PRICE (#119)."""
    b = _broker(fake_client, tmp_path, "HPG", _STOCK_SECDEF,
                post_order=_PLACED_STOCK)
    assert b.market_type == "STOCK", "fixture must classify HPG as STOCK"

    asyncio.run(b.execute_entry(_entry_envelope("HPG", limit=_INTENT_THOUSANDS)))

    payload = _payload(b)
    assert float(payload["price"]) == _WIRE_DONG, (
        f"STOCK order price must be ĐỒNG on the wire: intent "
        f"{_INTENT_THOUSANDS} thousands -> {_WIRE_DONG:.0f} đ, got "
        f"{payload['price']!r} — the venue floor-checks this against "
        f"19,850 đ and rejects (#119)")


# --- B (RED): the STOP legs of a STOCK conditional carry the same unit ------

def __test_stock_stop_entry_wire_prices_are_dong__(fake_client, tmp_path):
    """``stopPrice`` (and the LO price the trigger emits) cross the same
    boundary — ``_place`` rounds both with no market_type branch
    (broker.py:989/996-997)."""
    b = _broker(fake_client, tmp_path, "HPG", _STOCK_SECDEF,
                post_order=_PLACED_STOCK)

    asyncio.run(b.execute_entry(_entry_envelope("HPG", stop=_INTENT_THOUSANDS)))

    payload = _payload(b)
    assert float(payload["stopPrice"]) == _WIRE_DONG, (
        f"STOCK stopPrice must be ĐỒNG on the wire: {_INTENT_THOUSANDS} "
        f"thousands -> {_WIRE_DONG:.0f} đ, got {payload['stopPrice']!r} (#119)")
    assert float(payload["price"]) >= _WIRE_DONG, (
        f"the LO a triggered stop emits must be ĐỒNG too (priced through the "
        f"trigger); got {payload['price']!r} against a "
        f"{_WIRE_DONG:.0f} đ trigger (#119)")


# --- C (GREEN control): DERIVATIVES must NOT be rescaled --------------------

def __test_derivative_limit_entry_wire_price_is_unchanged__(fake_client, tmp_path):
    """Index points ARE the venue's unit for derivatives (docs: price 1990 on
    41I1G9000; 2058.6 accepted live 2026-09-14). A fix that multiplies every
    price by 1000 breaks this control."""
    b = _broker(fake_client, tmp_path, "VN30F1M", _DERIV_SECDEF,
                post_order=_PLACED_DERIV)
    assert b.market_type == "DERIVATIVE", "fixture must classify VN30F1M as DERIVATIVE"

    asyncio.run(b.execute_entry(_entry_envelope("VN30F1M", limit=2058.6, qty=1)))

    payload = _payload(b)
    assert float(payload["price"]) == 2058.6, (
        f"DERIVATIVE price must reach the wire unchanged (index points); got "
        f"{payload['price']!r}")
