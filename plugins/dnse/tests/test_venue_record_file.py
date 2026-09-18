"""#157 — the venue's own record, persisted so a run can be GRADED from it. Tests FIRST.

A live DNSE result is graded from the venue's order record, never from the run log. A fake-venue
run should be graded the same way, and until now it could not be: the record lives in the venue
object, the venue lives inside the ``pyne run`` process, and when that process ends the record
goes with it. Grading then falls back to the run log, which is the thing the rule exists to stop
anyone trusting.

So the venue can be given a file to keep its record in. Two properties matter, and both are
pinned here:

1. **It is written EAGERLY, after every state transition.** Not at exit. A supervised run is
   ended by the operator, and a run stopped with a signal flushes nothing — the parity harness
   already cost a day to exactly that, comparing a backtest against itself because the fake arm
   had been killed and had written no trades.
2. **It is OFF unless asked for.** A fake run that silently wrote files somewhere would be a
   surprise in a component whose whole point is that it touches nothing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue                                      # noqa: E402


def __test_the_record_is_on_disk_before_the_run_ends__(tmp_path):
    """The load-bearing property. Nothing is closed, finished or flushed here on purpose: a
    supervised run is stopped by the operator, and a run killed by a signal never reaches an
    exit hook."""
    target = tmp_path / "record.json"
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7,
                      record_file=target)

    order = venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)

    written = json.loads(target.read_text())
    assert [row["id"] for row in written] == [order["id"]]
    assert written[-1]["orderStatus"] == "New"


def __test_every_later_transition_reaches_the_file_too__(tmp_path):
    """The discriminating half of the test above: an implementation that wrote the record ONCE,
    at the first transition, would pass it and still lose the fill that a grade depends on."""
    target = tmp_path / "record.json"
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7,
                      record_file=target)
    order = venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)

    venue.feed_print(price=1980.0, volume=10.0)

    written = json.loads(target.read_text())
    assert len(written) >= 2, "the fill transition never reached the file"
    assert written[-1]["id"] == order["id"]
    assert written[-1]["orderStatus"] == "Filled"


def __test_the_file_matches_the_record_the_venue_holds_in_memory__(tmp_path):
    """The file is the SAME record, not a second summary that could drift from it."""
    target = tmp_path / "record.json"
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7,
                      record_file=target)
    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    venue.feed_print(price=1980.0, volume=10.0)

    assert json.loads(target.read_text()) == venue.records()


def __test_no_file_is_written_when_none_is_asked_for__(tmp_path):
    """Off by default. A component whose point is that it touches nothing must not start
    writing files because someone ran it."""
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)

    venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)

    assert list(tmp_path.iterdir()) == []
    assert venue.records(), "the in-memory record must still be kept"
