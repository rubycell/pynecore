"""#157 second pass — the fake's trading phase follows the REPLAYED DAY, not a clock. Tests FIRST.

The fake's phase was a string fixed when the venue object was constructed, and ``fake_broker``
never passed one, so it was permanently ``continuous``. Two things followed, both measured in the
first suite pass:

* a closed-hours question could not be posed at all — every "this must be refused because the
  venue is shut" probe was accepted, so T33 could only ever fail;
* the direct probes refused to start, because they compute the phase from ``datetime.now(ICT)``
  and the suite was run at 17:11 ICT, hours after the session they were replaying.

The fix here is the half that belongs to the fake: its phase follows the ORIGINAL timestamp of
the bar being replayed. That is deliberately not the shifted timestamp and not the wall clock.
The day is re-stamped onto the current minute so the engine's live path stays anchored, but a
replay of the 09-18 session is still a replay of 09:00 to 14:45 — so the venue should walk
continuous, lunch, continuous, ATC and closed exactly as that session did, whatever hour someone
runs it at. Deriving the phase from the shifted stamp would just reproduce the wall clock and
change nothing.

The other half — the SCRIPTS computing their own phase from ``datetime.now(ICT)`` — is not fixed
here. Those files belong to another session, and the change they need is an injected clock.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue, phase_at                             # noqa: E402

ICT_OFFSET_MS = 7 * 3600 * 1000


def _at(hour, minute, day_epoch_ms=1_789_488_000_000):
    """Epoch ms for a given ICT wall time on the replayed session's date (2026-09-18)."""
    midnight_utc = day_epoch_ms - (day_epoch_ms % 86_400_000)
    return midnight_utc + (hour * 3600 + minute * 60) * 1000 - ICT_OFFSET_MS


# --------------------------------------------------------------------------- the phase map

def __test_the_session_walks_its_real_phases__():
    """One assertion per phase the VN derivatives session actually has."""
    assert phase_at(_at(8, 30)) == "closed", "before the ATO window"
    assert phase_at(_at(10, 0)) == "continuous"
    assert phase_at(_at(12, 0)) == "lunch"
    assert phase_at(_at(13, 30)) == "continuous", "the afternoon leg"
    assert phase_at(_at(14, 35)) == "atc"
    assert phase_at(_at(15, 30)) == "closed", "after the close"


def __test_the_boundaries_land_on_the_documented_minute__():
    """The discriminating half. A phase map that returned `continuous` for the whole day would
    satisfy half the test above and silently restore the behaviour this replaces."""
    assert phase_at(_at(11, 29)) == "continuous"
    assert phase_at(_at(11, 30)) == "lunch", "the lunch break starts AT 11:30"
    assert phase_at(_at(12, 59)) == "lunch"
    assert phase_at(_at(13, 0)) == "continuous"
    assert phase_at(_at(14, 29)) == "continuous"
    assert phase_at(_at(14, 30)) == "atc", "the continuous session ends AT 14:30"
    assert phase_at(_at(14, 45)) == "closed"


# --------------------------------------------------------------------------- the venue follows it

def __test_the_venue_takes_its_phase_from_the_bar_being_replayed__():
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)

    venue.advance_to(_at(12, 0))

    assert venue.phase == "lunch"


def __test_a_placement_during_the_replayed_lunch_still_works_and_one_after_the_close_does_not__():
    """The point of the whole change: a phase the venue can actually ACT on.

    Lunch is the one phase where an order queues harmlessly, and after the close nothing can be
    placed at all. Until now neither could be reproduced, because the venue was always open.
    """
    from venue_core import VenueReject

    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)

    venue.advance_to(_at(12, 0))
    resting = venue.place(category="NORMAL", side="buy", qty=1, price=1900.0)
    assert resting["orderStatus"] == "New"

    venue.advance_to(_at(16, 0))
    try:
        venue.place(category="NORMAL", side="buy", qty=1, price=1900.0)
    except VenueReject as reject:
        assert reject.code, "a closed-session refusal must carry the venue's code"
    else:
        raise AssertionError("placing after the close must be refused, not accepted")


def __test_an_explicit_phase_still_wins_so_a_unit_test_can_pin_one__():
    """Every existing pin constructs the venue with a phase and drives it directly, without ever
    advancing a clock. Those must keep working, so a venue that is never advanced keeps the phase
    it was given."""
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0,
                      seed=7, phase="atc")

    assert venue.phase == "atc"
