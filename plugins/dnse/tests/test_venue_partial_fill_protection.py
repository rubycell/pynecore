"""#157 — protecting a PARTIALLY filled position: the coverage invariant, at real sizes.

The operator's case: an entry for **80** contracts fills **50** first, so protection must be sized
to 50 and not to 80. A second later another **10** fills, so protection must become **60**.

The mechanism is already pinned next door in ``test_broker_lifecycle.py`` (the #123 add-a-leg
path, at quantities 1 -> 2). What is NOT pinned anywhere is the INVARIANT those tests exist to
deliver, at sizes where getting it wrong costs something:

    total protective coverage == the entry's cumulative FILLED quantity

Both directions are expensive on a netting account, and they are expensive in different ways:

* **Over-protection** (a stop for 80 against 50 filled) does not merely close the position — the
  extra 30 REVERSES it through flat and opens a short the strategy never asked for.
* **Under-protection** (protection still at 50 after 60 has filled) leaves the newly filled lot
  naked, which is the exposure the protective exit exists to remove.

So each step is pinned from both sides. The helpers are imported from the lifecycle module rather
than copied: one definition of a fixture, not two that drift.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_broker_lifecycle import _broker, _envelope                   # noqa: E402

from pynecore.core.broker.models import ExitIntent                     # noqa: E402

ENTRY_QTY = 80
FIRST_FILL = 50
SECOND_FILL = 60          # 50 already filled, plus another 10


def _exit_at(qty):
    return ExitIntent(pine_id="X", from_entry="E", symbol="VN30F1M",
                      side="sell", qty=qty, sl_price=1960.0)


def _placed_quantities(broker):
    """Every quantity this broker actually sent to the venue, in order.

    Read from the WIRE rather than from any broker-side map, for two reasons. The broker keeps no
    per-order quantity (checked: it tracks ids and categories only), and the wire is the better
    evidence anyway — it is what the venue would have received.
    """
    out = []
    for name, args, kwargs in broker._client.calls:
        if name != "post_order":
            continue
        payload = kwargs.get("payload") if "payload" in kwargs else args[2]
        out.append(int(payload["quantity"]))
    return out


def _coverage(broker, *, already_armed=0):
    """Total protective quantity covering the position: legs already armed plus legs placed."""
    return already_armed + sum(_placed_quantities(broker))


# --------------------------------------------------------------------------- the first partial

def __test_protection_is_sized_to_the_filled_50_not_the_ordered_80__(fake_client, tmp_path):
    """The entry asked for 80 and 50 has filled. A stop for 80 would, on triggering, sell 80
    against a 50-lot long: the position closes and REVERSES into a 30-lot short the strategy
    never asked for. Protection follows the fill, not the order."""
    broker = _broker(fake_client, tmp_path,
                     post_order=(201, {"id": "STOP-50", "symbol": "VN30F1M", "side": "NS",
                                       "quantity": FIRST_FILL, "orderStatus": "New"}))
    envelope = _envelope(_exit_at(FIRST_FILL))

    asyncio.run(broker.execute_exit(envelope))

    assert _coverage(broker) == FIRST_FILL
    assert _coverage(broker) != ENTRY_QTY, (
        "a stop sized to the ORDER rather than the FILL reverses the position through flat")


# --------------------------------------------------------------------------- the second partial

def __test_protection_grows_to_60_when_another_10_fills__(fake_client, tmp_path):
    """Ten more fill a second later. Total coverage must become 60."""
    broker = _broker(fake_client, tmp_path,
                     post_order=(201, {"id": "STOP-10", "symbol": "VN30F1M", "side": "NS",
                                       "quantity": 10, "orderStatus": "New"}))
    old, new = _envelope(_exit_at(FIRST_FILL)), _envelope(_exit_at(SECOND_FILL))
    key = old.intent.intent_key
    broker._order_ids[key] = ["STOP-50"]
    broker._order_category["STOP-50"] = "STOP"

    asyncio.run(broker.modify_exit(old, new))

    assert _coverage(broker, already_armed=FIRST_FILL) == SECOND_FILL, (
        "after a second partial fill the position is 60 and so is its protection")


def __test_protection_does_not_stay_at_50_after_the_second_fill__(fake_client, tmp_path):
    """The discriminating half of the step above, stated as the hazard rather than the number.

    An implementation that simply PARKED the resize — which is what a conditional-book qty amend
    does on this venue (#18/#85/#93) — would leave the armed 50 in place and read as 'nothing
    broke'. The extra 10 lots would be naked, silently, which is the whole failure the protective
    exit exists to prevent."""
    broker = _broker(fake_client, tmp_path,
                     post_order=(201, {"id": "STOP-10", "symbol": "VN30F1M", "side": "NS",
                                       "quantity": 10, "orderStatus": "New"}))
    old, new = _envelope(_exit_at(FIRST_FILL)), _envelope(_exit_at(SECOND_FILL))
    key = old.intent.intent_key
    broker._order_ids[key] = ["STOP-50"]
    broker._order_category["STOP-50"] = "STOP"

    asyncio.run(broker.modify_exit(old, new))

    assert _coverage(broker, already_armed=FIRST_FILL) > FIRST_FILL, (
        "the newly filled lots must not be left naked")


def __test_the_grown_protection_never_exceeds_the_filled_quantity__(fake_client, tmp_path):
    """The other side of the same invariant, and the reason a delta leg is sized to the DELTA.

    A resize that armed a second FULL-SIZE leg (50 + 60) rather than the delta (50 + 10) would
    also satisfy 'coverage grew'. On a netting account it would then sell 110 against a 60-lot
    long."""
    broker = _broker(fake_client, tmp_path,
                     post_order=(201, {"id": "STOP-10", "symbol": "VN30F1M", "side": "NS",
                                       "quantity": 10, "orderStatus": "New"}))
    old, new = _envelope(_exit_at(FIRST_FILL)), _envelope(_exit_at(SECOND_FILL))
    key = old.intent.intent_key
    broker._order_ids[key] = ["STOP-50"]
    broker._order_category["STOP-50"] = "STOP"

    asyncio.run(broker.modify_exit(old, new))

    assert _coverage(broker, already_armed=FIRST_FILL) <= SECOND_FILL, (
        "over-protection on a netting account reverses the position through flat")


# --------------------------------------------------------------------------- how it resizes

def __test_the_resize_never_bares_the_armed_protection__(fake_client, tmp_path):
    """RECORDED BECAUSE IT CONTRADICTS THE REQUEST, and the contradiction is deliberate.

    The case was specified as 'delete and re-place the stop order' for the resize. The plugin does
    NOT do that, by design (#123): a cancel followed by a re-place bares the WHOLE position for
    the gap between the two, and on the conditional book a qty amend is refused outright
    (#18/#85/#93), so neither route is safe. Instead the armed leg is left live and an ADDITIONAL
    leg is placed for the delta, which reaches the same coverage with no naked window at any
    instant.

    The end state is identical to the one asked for — protection totalling 60 — so the invariant
    above holds either way. Only the path differs, and this pins the property that makes the
    chosen path the safer one.
    """
    broker = _broker(fake_client, tmp_path,
                     post_order=(201, {"id": "STOP-10", "symbol": "VN30F1M", "side": "NS",
                                       "quantity": 10, "orderStatus": "New"}))
    old, new = _envelope(_exit_at(FIRST_FILL)), _envelope(_exit_at(SECOND_FILL))
    key = old.intent.intent_key
    broker._order_ids[key] = ["STOP-50"]
    broker._order_category["STOP-50"] = "STOP"

    asyncio.run(broker.modify_exit(old, new))

    assert broker._client.count("cancel_order") == 0, (
        "the armed 50-lot protection is never cancelled — a cancel-then-replace would leave the "
        "whole position unprotected for the gap between them")
    assert broker._client.count("put_order") == 0, (
        "and never a conditional-book qty amend, which this venue refuses (#18)")
    assert broker._client.count("post_order") == 1, "exactly one additional leg, for the delta"
