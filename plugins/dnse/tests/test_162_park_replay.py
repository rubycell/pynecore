"""#162 offline park replay — the venue half, pinned against the fake.

On 2026-09-18 a live run held a position with no protective order for 66 seconds. The mechanism,
from that run's own sequence: an OCO's stop leg AMENDED its child in place and the child filled,
so the child became terminal BY VENUE AMENDMENT. A forced cancel was then issued against that
id, and the venue refused it — permanently. The engine's deferral guard assumed a forced cancel
either lands or fails transiently, so the park never cleared and every later exit dispatch for
that intent key queued behind it.

This replays the venue side of that sequence offline and pins the property the engine's
assumption depended on: the refusal is PERMANENT, not transient. Until now that could only be
observed with real money during market hours.

What this file does NOT claim: it does not exercise the engine's park or its release. That needs
the engine at both code versions and is the next piece of work (#164's pin bed). Pinning the
venue behaviour first is what makes the engine-level pin meaningful, because an engine test
written against an invented refusal would prove nothing about the real venue.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue, VenueReject                          # noqa: E402


def _armed_bracket():
    """A long position protected by an OCO: TP above, stop below — the l2b shape."""
    venue = FakeVenue(symbol="41I1G9000", market_type="DERIVATIVE",
                      last_price=1995.0, seed=162)
    oco = venue.place(category="OCO", side="sell", qty=1, price=2005.3, stop_price=1994.3)
    child_id = venue.order(oco["id"])["externalOrderId"]
    return venue, oco["id"], child_id


def __test_the_run_a_sequence_leaves_the_child_terminal_by_amendment__():
    """Replays the measured order of events: the stop leg rewrites the child, then it fills.

    The child never becomes terminal by our action — no cancel, no fill request from us. It is
    the VENUE that rewrites and fills it, which is why the engine held a stale belief about it.
    """
    venue, umbrella_id, child_id = _armed_bracket()

    venue.feed_print(price=1994.0, volume=5)        # through the stop leg

    child = venue.order(child_id)
    assert "PendingReplace" in [r["orderStatus"] for r in venue.records()
                                if r["id"] == child_id], "the amend must be observable"
    assert child["orderStatus"] == "Filled", "the rewritten child fills"
    assert venue.order(umbrella_id)["orderStatus"] == "Activated", (
        "and the umbrella still reads Activated, indistinguishable from an armed one")


def __test_the_forced_cancel_is_refused_and_stays_refused__():
    """The property the engine's deferral guard assumed away.

    Three attempts, because the live engine retried three times and got the same answer each
    time. A transient refusal would have cleared the park; a permanent one cannot, and that is
    what left the position unprotected.
    """
    venue, _, child_id = _armed_bracket()
    venue.feed_print(price=1994.0, volume=5)

    codes = []
    for _ in range(3):
        with pytest.raises(VenueReject) as excinfo:
            venue.cancel(child_id)
        codes.append(excinfo.value.code)

    assert codes == ["ORDER_CANCEL_STATUS_REJECTED"] * 3, (
        f"the refusal must be identical and permanent across retries, got {codes}")


def __test_a_resting_child_can_still_be_cancelled__():
    """The discriminating half. Without it, a venue that refused EVERY cancel would satisfy the
    test above while telling us nothing about WHY the park could not clear. The refusal must be
    caused by the child having become terminal, not by cancels being broken."""
    venue, _, child_id = _armed_bracket()

    cancelled = venue.cancel(child_id)              # no print yet: the child is still resting

    assert cancelled["orderStatus"] == "Canceled"


def __test_the_umbrella_cannot_be_cancelled_either_so_there_is_no_second_route__():
    """Why the engine had no fallback. The umbrella is Activated from birth, and an Activated
    conditional is done, so cancelling it answers CO-ORD-013. There is no far leg to retire and
    no second id to act on: the child is the only handle, and it is terminal."""
    venue, umbrella_id, child_id = _armed_bracket()
    venue.feed_print(price=1994.0, volume=5)

    with pytest.raises(VenueReject) as excinfo:
        venue.cancel(umbrella_id)
    assert excinfo.value.code == "CO-ORD-013"
