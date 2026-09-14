"""#119 — the STOCK price-unit codec: ladder, readbacks, amend diff, guards.

Companion to ``test_stock_price_unit.py`` (which pins the WRITE unit itself).
Everything here is about the parts a 400 would never have caught:

* the HOSE tick ladder, which only exists in ĐỒNG (10 / 50 / 100 by band) and
  which the old ``round(price, 1)`` flattened to a 100 đ grid;
* the READBACKS (fill price, slice price) — silent at 1000x;
* the AMEND diff, which compared a đồng detail against a thousands intent, so
  every stock modify looked like a change;
* **G1**, the load-bearing guard: the stock x1000 must NEVER ride on a guessed
  classification (``provider.classify_market_type`` answers STOCK for any
  symbol whose secdef read came back empty — including a dated derivative
  contract code, which would then be 1000x'd into an unprotected position);
* **G2**, enablement: live stock placement stays opt-in until a real stock
  fill confirms the readback units.

Same fake-client seam as the rest of the suite — no live venue, no test hook
in plugin code.
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker, provider as dnse_provider
from pynecore_dnse.price_units import (
    StockOrdersDisabledError, UnverifiedClassificationError,
)
from pynecore.core.broker.models import (
    DispatchEnvelope, EntryIntent, LegType, OrderType,
)

#: HPG as the venue serves it — secdef is THOUSANDS, ``ST`` = stock.
_STOCK_SECDEF = [{"symbol": "HPG", "securityGroupId": "ST", "basicPrice": "21.3",
                  "ceilingPrice": "22.75", "floorPrice": "19.85"}]
#: The VN30 front month's DATED contract code — note it does NOT start with
#: ``VN30F``, so the prefix fallback guesses STOCK for it.
_DERIV_SECDEF = [{"symbol": "41I1G9000", "securityGroupId": "FU",
                  "ceilingPrice": "2200", "floorPrice": "1900"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})
_PLACED = (201, {"id": "437346", "symbol": "HPG", "side": "NB", "quantity": 100,
                 "orderStatus": "New", "price": 21300})


def _broker(fake_client, tmp_path, symbol="HPG", secdef=_STOCK_SECDEF, *,
            enable_stock_orders=True, **client_responses):
    responses = {"get_security_definition": (200, secdef),
                 "get_loan_packages": _LOAN_OK,
                 "get_instruments": (200, {"data": []})}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"),
        enable_stock_orders=enable_stock_orders)
    instance = broker.DNSEBroker(symbol=symbol, timeframe="15", config=config)
    instance._client = fake_client(**responses)
    return instance


def _envelope(intent):
    return DispatchEnvelope(intent=intent, run_tag="abcd",
                            bar_ts_ms=1_700_000_000_000, retry_seq=0,
                            coid_max_len=30)


def _entry(symbol="HPG", *, limit=21.3, qty=100, pine_id="L"):
    return _envelope(EntryIntent(pine_id=pine_id, symbol=symbol, side="buy",
                                 qty=qty, order_type=OrderType.LIMIT,
                                 limit=limit, stop=None))


def _posts(instance):
    return [c for c in instance._client.calls if c[0] == "post_order"]


# --- the HOSE ladder, in đồng ----------------------------------------------

@pytest.mark.parametrize("feed_thousands, wire_dong", [
    pytest.param(21.31, 21300.0, id="50d-band-21.31k->21300"),
    pytest.param(9.994, 9990.0, id="10d-band-9.994k->9990"),
    pytest.param(51.03, 51000.0, id="100d-band-51.03k->51000"),
])
def __test_stock_prices_snap_to_the_hose_ladder_in_dong__(
        fake_client, tmp_path, feed_thousands, wire_dong):
    """HOSE ticks are 10 đ below 10,000, 50 đ to 49,950, 100 đ from 50,000 —
    i.e. 0.01 / 0.05 / 0.10 in feed units. ``round(price, 1)`` could only ever
    express the 100 đ step, so every sub-50k stock price was coarsened."""
    b = _broker(fake_client, tmp_path)

    wire = b._wire_price(feed_thousands, b._wire_scale(writing=True))

    assert wire == wire_dong, (
        f"{feed_thousands} thousands must snap to {wire_dong:.0f} đ on the "
        f"venue's ladder, got {wire!r} (#119)")


def __test_derivative_quantization_is_unchanged__(fake_client, tmp_path):
    """CONTROL: derivatives keep the pre-#119 quantizer verbatim — scale 1,
    ``round(price, 1)``, the 0.1-point tick."""
    b = _broker(fake_client, tmp_path, "VN30F1M", _DERIV_SECDEF)
    scale = b._wire_scale(writing=True)

    assert scale == 1.0, "a derivative must never be rescaled"
    for price in (2058.6, 2058.65, 1900.0, 1989.94):
        assert b._wire_price(price, scale) == round(price, 1), (
            f"derivative quantization must stay round(price, 1): {price}")


# --- amend: one unit on both sides of the diff ------------------------------

def __test_stock_no_change_amend_emits_zero_puts__(fake_client, tmp_path):
    """The venue detail is ĐỒNG (21300) and the intent is THOUSANDS (21.3).
    Comparing them raw made every stock modify look like a price change, so a
    NO-CHANGE modify emitted a wrong-unit PUT (and the #86 half-applied check
    could never confirm). The diff must run in ONE unit."""
    order_id = "437346"
    b = _broker(fake_client, tmp_path,
                put_order=(200, {"id": order_id, "orderStatus": "New",
                                 "quantity": 100}),
                get_order_detail=(200, {"id": order_id, "orderStatus": "New",
                                        "price": 21300, "quantity": 100}))
    envelope = _entry()
    b._order_ids[envelope.intent.intent_key] = [order_id]
    b._order_category[order_id] = "NORMAL"

    asyncio.run(b._amend(envelope, _entry(), is_exit=False))

    assert b._client.count("put_order") == 0, (
        f"a stock modify that changes nothing must write NOTHING; the venue "
        f"rests at 21300 đ and the intent is 21.3 thousands — the same price "
        f"(#119)")


def __test_stock_real_amend_still_puts_the_dong_price__(fake_client, tmp_path):
    """DISCRIMINATING control for the test above: a modify that DOES move the
    price must still emit exactly one PUT, carrying đồng."""
    order_id = "437346"
    b = _broker(fake_client, tmp_path,
                put_order=(200, {"id": order_id, "orderStatus": "New",
                                 "quantity": 100}),
                get_order_detail=(200, {"id": order_id, "orderStatus": "New",
                                        "price": 21300, "quantity": 100}))
    envelope = _entry()
    b._order_ids[envelope.intent.intent_key] = [order_id]
    b._order_category[order_id] = "NORMAL"

    asyncio.run(b._amend(envelope, _entry(limit=21.35), is_exit=False))

    puts = [c for c in b._client.calls if c[0] == "put_order"]
    assert len(puts) == 1, f"one changed field -> one PUT, got {len(puts)}"
    assert float(puts[0][1][3]["price"]) == 21350.0, (
        f"the amended price must be ĐỒNG: 21.35 thousands -> 21350 đ, got "
        f"{puts[0][1][3]['price']!r} (#119)")


# --- readbacks: wire -> feed ------------------------------------------------

def _own(instance, order_id="437346"):
    instance._identity[order_id] = ("L", None, LegType.ENTRY)
    instance._order_category[order_id] = "NORMAL"


def _row(order_id="437346", *, status="Filled", cumulative=100.0, avg=24250.0):
    return {"id": order_id, "symbol": "HPG", "side": "NB", "quantity": 100,
            "orderStatus": status, "fillQuantity": cumulative,
            "averagePrice": avg, "price": avg}


def __test_stock_vwap_fill_price_reads_back_in_thousands__(fake_client, tmp_path):
    """The documented stock execution price is 24250 đ (dnse-get-executions.md)
    — the engine must see 24.25, the unit its bars and its Pine levels use.
    A 1000x fill price never 400s; it just books a 1000x P&L."""
    b = _broker(fake_client, tmp_path)
    _own(b)

    events = asyncio.run(b._scan_row(_row()))

    assert events and events[0].fill_price == pytest.approx(24.25), (
        f"a 24250 đ fill must reach the engine as 24.25 thousands, got "
        f"{events[0].fill_price if events else None!r} (#119)")
    assert events[0].order.price == pytest.approx(24.25), (
        "the order row's own price crosses the same boundary")


def __test_stock_slice_fill_price_reads_back_in_thousands__(fake_client, tmp_path):
    """Same for the per-slice path (#56): ``lastPrice`` is ĐỒNG too."""
    b = _broker(fake_client, tmp_path, get_execution_detail=(200, {"reports": [
        {"orderStatus": "Filled", "fillQuantity": 100, "lastQuantity": 100,
         "lastPrice": 24250}]}))
    _own(b)

    events = asyncio.run(b._scan_row(_row()))

    assert events and events[0].fill_price == pytest.approx(24.25), (
        f"a 24250 đ slice must book at 24.25 thousands, got "
        f"{events[0].fill_price if events else None!r} (#119)")


def __test_stock_position_entry_price_reads_back_in_thousands__(
        fake_client, tmp_path):
    """The positions book prices in đồng like the order book; a 1000x
    entry_price silently poisons every P&L and adoption decision."""
    b = _broker(fake_client, tmp_path, get_positions=(200, {"positions": [
        {"symbol": "HPG", "side": "NB", "openQuantity": 100, "quantity": 100,
         "costPrice": 21300, "status": "OPEN"}]}))

    position = asyncio.run(b.get_position("HPG"))

    assert position is not None and position.entry_price == pytest.approx(21.3), (
        f"entry_price must be thousands, got "
        f"{position.entry_price if position else None!r} (#119)")


def __test_derivative_fill_price_readback_is_unchanged__(fake_client, tmp_path):
    """CONTROL: an index-point fill must reach the engine verbatim."""
    b = _broker(fake_client, tmp_path, "VN30F1M", _DERIV_SECDEF)
    _own(b)
    row = _row(avg=2058.6)
    row["symbol"] = "VN30F1M"

    events = asyncio.run(b._scan_row(row))

    assert events and events[0].fill_price == pytest.approx(2058.6)


# --- G1: never scale on a GUESSED classification ----------------------------

def __test_guessed_stock_write_raises_instead_of_scaling__(fake_client, tmp_path):
    """``41I1G9000`` is a DERIVATIVE, but it does not start with ``VN30F`` and
    its secdef came back empty, so the fallback guesses STOCK. Scaling on that
    guess would put 2,058,600 on the wire — or, on the fill side, book a 1000x
    position and mint SL/TP levels 1000x away (unprotected). Fail LOUD."""
    b = _broker(fake_client, tmp_path, "41I1G9000", [], post_order=_PLACED)
    assert b.classify_market_type() == ("STOCK", False), (
        "fixture must reproduce the guessed-STOCK classification")

    with pytest.raises(UnverifiedClassificationError):
        asyncio.run(b.execute_entry(_entry("41I1G9000", limit=2058.6, qty=1)))

    assert _posts(b) == [], (
        "a refused write must never reach the venue — and must never be the "
        "1000x'd payload (#119/G1)")


def __test_authoritative_derivative_is_never_blocked__(fake_client, tmp_path):
    """DISCRIMINATING control for G1: the guard must reject GUESSES, not
    derivatives. The same dated code WITH its FU secdef places unscaled."""
    b = _broker(fake_client, tmp_path, "41I1G9000", _DERIV_SECDEF,
                post_order=(201, {"id": "437347", "symbol": "41I1G9000",
                                  "side": "NB", "quantity": 1,
                                  "orderStatus": "New", "price": 2058.6}))
    assert b.classify_market_type() == ("DERIVATIVE", True)

    asyncio.run(b.execute_entry(_entry("41I1G9000", limit=2058.6, qty=1)))

    payload = _posts(b)[0][1][2]
    assert float(payload["price"]) == 2058.6, (
        f"an authoritative derivative must reach the wire in index points, "
        f"got {payload['price']!r}")


def __test_guessed_stock_read_keeps_the_identity_scale__(fake_client, tmp_path):
    """A READ cannot refuse (one bad poll would kill reconcile), so it keeps
    the pre-#119 behaviour — venue prices verbatim — and warns once."""
    b = _broker(fake_client, tmp_path, "41I1G9000", [])

    assert b._wire_scale(writing=False) == 1.0
    assert b._unverified_scale_warned is True, "the unprovable read must warn"


def __test_empty_secdef_is_not_cached_permanently__(fake_client, tmp_path,
                                                    monkeypatch):
    """G1's other half: the classification cache must not make a TRANSIENT
    read failure sticky for the whole run. With the old permanent ``{}``,
    one failed secdef read pinned ``41I1G9000`` to the STOCK guess forever."""
    monkeypatch.setattr(dnse_provider, "_SECDEF_RETRY_S", 0.0)
    answers = [(500, {}), (200, _DERIV_SECDEF)]
    b = _broker(fake_client, tmp_path, "41I1G9000", [],
                get_security_definition=lambda *a, **k: answers.pop(0))

    assert b.classify_market_type() == ("STOCK", False), "the failed read guesses"
    assert b.classify_market_type() == ("DERIVATIVE", True), (
        "the NEXT read must heal the classification — a transient failure "
        "must not be sticky (#119/G1)")


# --- G2: live stock placement is opt-in -------------------------------------

def __test_stock_placement_requires_the_opt_in_flag__(fake_client, tmp_path):
    """The write unit is measured; the READBACK units are not. Until one
    hand-verified real stock fill confirms them, a stock order must not go
    out by default."""
    b = _broker(fake_client, tmp_path, enable_stock_orders=False,
                post_order=_PLACED)

    with pytest.raises(StockOrdersDisabledError):
        asyncio.run(b.execute_entry(_entry()))

    assert _posts(b) == [], "the gate must block BEFORE the POST"


def __test_g2_flag_does_not_gate_derivatives__(fake_client, tmp_path):
    """DISCRIMINATING control: the flag is about stocks only — the derivative
    path must place with it off."""
    b = _broker(fake_client, tmp_path, "VN30F1M", _DERIV_SECDEF,
                enable_stock_orders=False,
                post_order=(201, {"id": "437347", "symbol": "VN30F1M",
                                  "side": "NB", "quantity": 1,
                                  "orderStatus": "New", "price": 2058.6}))

    asyncio.run(b.execute_entry(_entry("VN30F1M", limit=2058.6, qty=1)))

    assert len(_posts(b)) == 1, "derivatives must be unaffected by the G2 flag"


def __test_g2_reads_are_not_gated__(fake_client, tmp_path):
    """A disabled stock must still READ correctly — an existing position or a
    resting order placed elsewhere is reported in feed units regardless."""
    b = _broker(fake_client, tmp_path, enable_stock_orders=False)

    assert b._wire_scale(writing=False) == 1000.0
