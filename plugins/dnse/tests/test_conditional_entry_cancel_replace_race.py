"""#85 chasing-entry cancel-vs-fill race — the DOUBLE-OPEN guard, end to end.

``l2b_entry_update.pine`` (padPct=0) re-issues ``strategy.entry("E", stop=…)``
every 1m candle so a resting conditional STOP tracks a moving breakout level.
DNSE cannot amend a conditional (#18/#85), so each re-issue is a CANCEL+REPLACE:
``modify_entry`` -> :meth:`DNSEBroker._amend` -> :meth:`_cancel_replace_entry`.

The hazard is the classic race: on the candle price crosses the resting stop,
the venue FILLS the entry at the same moment the bar-close sync tries to cancel
it to re-place higher. The cancel then comes back "order is done" — and if the
plugin read that as a clean cancel and went on to place the replacement, the
account would hold TWO longs (the filled entry + the fresh replacement) = the
#55/#85 DOUBLE-OPEN on a netting account.

``test_cancel_disposition.py`` already pins the DISPOSITION layer (CO-ORD-013 +
a filled child -> ``ALREADY_FILLED``, tests B and G2). This file pins the level
ABOVE it: that :meth:`_cancel_replace_entry` HONOURS a non-confirmed cancel by
PARKING (``OrderDispositionUnknownError``) and NEVER posts the replacement — so
the double-open cannot happen — while a genuinely confirmed cancel (the benign
chase-update / missed-fill runaway, no position) DOES place the replacement.

The discriminating control is load-bearing (red-first): a broken plugin that
always placed the replacement fails the race test; one that never placed it
fails the control. The two together fence the exact behaviour.

Engine-side follow-through (the parked predecessor cancel is consumed as
engine-initiated, NOT an unexpected external cancel -> no quarantine; the fill
restores legs and the bracket arms) is pinned separately by the sync-engine
suite — ``tests/t00_pynecore/core/test_025_order_sync_engine.py``
``__test_parked_modify_predecessor_cancel_is_not_unexpected__`` (asserts
``engine.quarantined is False``). This file is the DNSE-plugin half.

Same fake-client seam as ``test_cancel_disposition.py``.
"""
import asyncio

import pytest

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.exceptions import OrderDispositionUnknownError
from pynecore.core.broker.models import (
    EntryIntent, DispatchEnvelope, OrderType,
)

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    instance._cancel_verify_attempts = 2
    instance._cancel_verify_delay = 0.0
    return instance


def _entry_env(stop_level):
    """A chasing conditional STOP entry, pine_id 'E' (intent_key == 'E')."""
    return DispatchEnvelope(
        intent=EntryIntent(pine_id="E", symbol="VN30F1M", side="buy", qty=1,
                           order_type=OrderType.STOP, stop=stop_level),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)


def _track_stop_entry(b, order_id="SHELL"):
    """Register a resting conditional STOP entry on the STOP book under key 'E'."""
    b._order_ids["E"] = [order_id]
    b._order_category[order_id] = "STOP"


# --- THE HAZARD: cancel raced by the fill must PARK, never place the replace --

def __test_chasing_entry_cancel_raced_by_fill_does_not_double_open__(fake_client, tmp_path):
    """The re-issue's cancel loses the race: the venue answers CO-ORD-013
    ("order is done") because the conditional Activated and its NORMAL-book
    child already FILLED. ``_cancel_replace_entry`` must classify that as
    ``ALREADY_FILLED`` and PARK (``OrderDispositionUnknownError``) WITHOUT
    posting the replacement — otherwise the filled entry + the replacement =
    a second open lot (the #55/#85 double-open on a netting account)."""
    def _detail_by_id(_account, order_id, _market, order_category=None):
        # The Activated conditional shell names its NORMAL-book child…
        if order_id == "SHELL":
            return (200, {"id": "SHELL", "orderStatus": "Activated",
                          "externalOrderId": "CH1"})
        # …and the child FILLED — the entry executed mid-race.
        return (200, {"id": "CH1", "symbol": "VN30F1M", "side": "NB",
                      "quantity": 1, "orderStatus": "Filled", "fillQuantity": 1.0})

    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "CO-ORD-013"}),
                get_order_detail=_detail_by_id,
                # If the guard were broken and the replace went out, it would
                # succeed here — making the double-open observable as a post.
                post_order=(201, {"id": "REPLACEMENT", "symbol": "VN30F1M",
                                  "side": "NB", "quantity": 1, "orderStatus": "New"}))
    _track_stop_entry(b)

    old = _entry_env(1500.0)
    new = _entry_env(1505.0)  # the chased-higher level

    with pytest.raises(OrderDispositionUnknownError) as excinfo:
        asyncio.run(b.modify_entry(old, new))

    # The park explicitly names the filled outcome — proof the raise is the
    # fill-race branch, not some unrelated ambiguity.
    assert "already_filled" in str(excinfo.value), (
        f"the park must surface the ALREADY_FILLED disposition; "
        f"got: {excinfo.value}")
    # THE INVARIANT: the replacement was NEVER placed -> no second open lot.
    assert b._client.count("post_order") == 0, (
        "the replacement entry was placed after a cancel that really meant "
        "'it filled' — DOUBLE-OPEN (#55/#85). A non-CANCEL_CONFIRMED "
        "disposition must PARK, never place the replace half.")


# --- DISCRIMINATING CONTROL: a genuinely confirmed cancel DOES re-place ------

def __test_chasing_entry_confirmed_cancel_places_the_replacement__(fake_client, tmp_path):
    """Red-first control (the benign chase-update / missed-fill runaway): the
    cancel WINS — the resting stop is cancelled with zero fill before price
    crossed — so ``_cancel_replace_entry`` gets ``CANCEL_CONFIRMED`` and MUST
    post the replacement at the new level (no position, just a non-fill that
    re-arms higher). Without this, a plugin that 'never places' would pass the
    hazard test above while silently breaking the whole chasing mechanism."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(200, {"orderStatus": "New"}),
                get_order_detail=(200, {"id": "SHELL", "symbol": "VN30F1M",
                                        "side": "NB", "quantity": 1,
                                        "orderStatus": "Canceled", "fillQuantity": 0.0}),
                post_order=(201, {"id": "REPLACEMENT", "symbol": "VN30F1M",
                                  "side": "NB", "quantity": 1, "orderStatus": "New"}))
    _track_stop_entry(b)

    old = _entry_env(1500.0)
    new = _entry_env(1505.0)

    orders = asyncio.run(b.modify_entry(old, new))

    assert b._client.count("post_order") == 1, (
        "a positively confirmed cancel must place the replacement — the chase "
        "cannot track the breakout otherwise")
    assert orders and orders[0].id == "REPLACEMENT"
    # The new working order replaced the cancelled predecessor under key 'E'.
    assert b._order_ids["E"] == ["REPLACEMENT"]


# --- UNKNOWN disposition also parks (cannot prove the old order is gone) ------

def __test_chasing_entry_unknown_cancel_does_not_double_open__(fake_client, tmp_path):
    """Not only the fill-race: a cancel whose disposition cannot be positively
    read (venue serves a cancelled STOP as ``New`` for >12 s, or the detail is
    unreachable) is UNKNOWN — replacing could rest TWO live stops. It too must
    PARK and post nothing."""
    b = _broker(fake_client, tmp_path,
                # 200-ACK but the read-back keeps saying 'New' -> UNKNOWN (G5).
                cancel_order=(200, {"orderStatus": "New"}),
                get_order_detail=(200, {"id": "SHELL", "symbol": "VN30F1M",
                                        "side": "NB", "quantity": 1,
                                        "orderStatus": "New", "fillQuantity": 0.0}),
                post_order=(201, {"id": "REPLACEMENT", "orderStatus": "New"}))
    _track_stop_entry(b)

    with pytest.raises(OrderDispositionUnknownError) as excinfo:
        asyncio.run(b.modify_entry(_entry_env(1500.0), _entry_env(1505.0)))

    assert "unknown" in str(excinfo.value)
    assert b._client.count("post_order") == 0, (
        "an UNKNOWN cancel disposition cannot prove the old stop is gone — "
        "placing the replacement risks two live stops on the netting account")
