"""#117 — a STOCK amend returns a NEW order id; every tracked ref must move.

Measured live 2026-09-15: ``PUT /orders/{id}`` on a stock answers 200 with a
DIFFERENT id — the venue REPLACED the order and auto-cancels the predecessor
(whose ``Canceled`` push arrives later). Before the fix the plugin kept
tracking the OLD id: the new order was unknown (its fill could not be routed),
a later cancel/amend addressed a dead id, and the predecessor's cancel push
arrived unowned. Derivatives answer with the SAME id and must be unaffected.

Same fake-client seam as ``test_broker_orders.py``: no network, no real files.
"""
import asyncio
import logging

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    DispatchEnvelope, EntryIntent, LegType, OrderType,
)

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})

#: The venue's resting values the amend diffs against (``_order_detail_dict``).
_RESTING = (200, {"id": "OLD1", "symbol": "VN30F1M", "side": "NB",
                  "price": 1500.0, "quantity": 1, "orderStatus": "New"})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK, "get_order_detail": _RESTING}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    return instance


def _entry(price, qty=1):
    return EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=qty,
                       order_type=OrderType.LIMIT, limit=price)


def _envelope(intent):
    return DispatchEnvelope(intent=intent, run_tag="abcd",
                            bar_ts_ms=1_700_000_000_000)


def _track(instance, intent, order_id="OLD1"):
    instance._order_ids[intent.intent_key] = [order_id]
    instance._identity[order_id] = ("L", None, LegType.ENTRY)
    instance._order_category[order_id] = "NORMAL"


def _amend(instance, old_price, new_price, *, qty=1, new_qty=None):
    old, new = _entry(old_price, qty), _entry(new_price, new_qty or qty)
    _track(instance, old)
    return asyncio.run(instance.modify_entry(_envelope(old), _envelope(new))), old


def __test_amend_new_id_remaps_every_tracked_reference__(fake_client, tmp_path):
    """The PUT hands back ``NEW2``: the intent's id list, the identity map and
    the book record must all name it — otherwise the live order is untracked."""
    b = _broker(fake_client, tmp_path, put_order=(
        200, {"id": "NEW2", "symbol": "VN30F1M", "side": "NB", "price": 1510.0,
              "quantity": 1, "orderStatus": "New"}))

    orders, old = _amend(b, 1500.0, 1510.0)

    assert b._order_ids[old.intent_key] == ["NEW2"], (
        f"the intent must map to the live id, got "
        f"{b._order_ids[old.intent_key]!r} (#117)")
    assert b._identity.get("NEW2") == ("L", None, LegType.ENTRY), \
        "the new id must carry the old id's identity or its fill is unroutable"
    assert "OLD1" not in b._identity, "the superseded id must not stay tracked"
    assert b._order_category.get("NEW2") == "NORMAL", \
        "the book record must follow the id a cancel would target"
    assert orders[0].id == "NEW2"


def __test_new_id_receives_the_intents_events__(fake_client, tmp_path):
    """The point of the re-map: a fill on the NEW id resolves to the intent."""
    b = _broker(fake_client, tmp_path, put_order=(
        200, {"id": "NEW2", "symbol": "VN30F1M", "side": "NB", "price": 1510.0,
              "quantity": 1, "orderStatus": "New"}))
    _amend(b, 1500.0, 1510.0)

    events = asyncio.run(b._scan_row(
        {"id": "NEW2", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
         "orderStatus": "Filled", "fillQuantity": 1, "averagePrice": 1510.0}))

    assert len(events) == 1 and events[0].pine_id == "L", (
        f"the new id's fill must resolve to the amended intent, got {events!r}")


def __test_old_ids_cancel_push_is_expected_not_unowned__(
        fake_client, tmp_path, caplog):
    """The venue auto-cancels the predecessor. That push must be consumed as
    the engine's own amend landing — never leak out as a cancel nobody asked
    for (which is what fires the unexpected-cancel policy downstream)."""
    b = _broker(fake_client, tmp_path, put_order=(
        200, {"id": "NEW2", "symbol": "VN30F1M", "side": "NB", "price": 1510.0,
              "quantity": 1, "orderStatus": "New"}))
    _amend(b, 1500.0, 1510.0)
    assert "OLD1" in b._superseded_amend_order_ids

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(
            {"id": "OLD1", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
             "orderStatus": "Canceled", "fillQuantity": 0}))

    assert events == [], \
        "the predecessor's cancel must not reach the engine as an event"
    assert any("#117 expected cancel of amend predecessor OLD1" in r.message
               for r in caplog.records), "and it must say so in the log"
    assert "OLD1" not in b._superseded_amend_order_ids, "one-shot consumption"


def __test_second_amend_leg_targets_the_new_id__(fake_client, tmp_path):
    """#86 sends one PUT per changed field. Once the first PUT replaced the
    order, the SECOND must address the replacement — a PUT on the superseded
    id addresses an order the venue already cancelled."""
    ids_handed_back = iter(["NEW2", "NEW3"])

    def _put(*_args, **_kwargs):
        return (200, {"id": next(ids_handed_back), "symbol": "VN30F1M",
                      "side": "NB", "price": 1510.0, "quantity": 2,
                      "orderStatus": "New"})

    b = _broker(fake_client, tmp_path, put_order=_put)

    orders, old = _amend(b, 1500.0, 1510.0, qty=1, new_qty=2)

    put_calls = [c for c in b._client.calls if c[0] == "put_order"]
    assert len(put_calls) == 2, "price leg then quantity leg (#86)"
    assert put_calls[0][1][1] == "OLD1"
    assert put_calls[1][1][1] == "NEW2", (
        f"the quantity leg must target the replacement, got "
        f"{put_calls[1][1][1]!r} (#117)")
    assert b._order_ids[old.intent_key] == ["NEW3"]
    assert orders[0].id == "NEW3"


def __test_same_id_amend_changes_nothing__(fake_client, tmp_path, caplog):
    """The derivatives PUT returns the SAME id — the guard must make that path
    byte-identical: no re-map, no expected-cancel registration, no log line."""
    b = _broker(fake_client, tmp_path, put_order=(
        200, {"id": "OLD1", "symbol": "VN30F1M", "side": "NB", "price": 1510.0,
              "quantity": 1, "orderStatus": "New"}))

    with caplog.at_level(logging.DEBUG):
        orders, old = _amend(b, 1500.0, 1510.0)

    assert b._order_ids[old.intent_key] == ["OLD1"]
    assert b._identity.get("OLD1") == ("L", None, LegType.ENTRY)
    assert b._superseded_amend_order_ids == set(), \
        "nothing was superseded, so nothing may be registered as expected"
    assert not [r for r in caplog.records if "#117 amend re-mapped" in r.message]
    assert orders[0].id == "OLD1"
