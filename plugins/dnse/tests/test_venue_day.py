"""#157 stage B: the recorded/synthetic venue day — tests written FIRST.

A "venue day" is the replayable input to the fake venue: the per-print tick stream plus the 1m
bars for one session. Two ways to get one:

* **RECORDED** — captured from the real venue during a live session (scheduled with the leader
  for 2026-09-21; this session never contacts the venue).
* **SYNTHETIC** — derived from tracked ``.ohlcv`` bars, so stage C can proceed before any live
  recording exists.

The single most important property pinned here is that the two are **never confusable**. A
synthetic day that could be read as a recorded one would let a measurement be claimed that never
happened, which is the most expensive kind of wrong this project produces. So the loader refuses
a day that does not say which it is, rather than defaulting.

A recorder must also survive being started mid-session and say so (PARTIAL), because the live
recording window may be joined late and a partial day silently presented as a whole one would
misrepresent the session's open.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue                                      # noqa: E402
from venue_day import (                                              # noqa: E402
    VenueDay, DayLabel, MalformedDay, synthesise_day, load_day, save_day,
)


def _bars():
    """Three 1m bars, ascending, with a clean high/low structure to reconstruct."""
    return [
        {"timestamp": 1_800_000_000, "open": 1980.0, "high": 1985.0, "low": 1979.0,
         "close": 1984.0, "volume": 120.0},
        {"timestamp": 1_800_000_060, "open": 1984.0, "high": 1991.0, "low": 1983.5,
         "close": 1990.0, "volume": 200.0},
        {"timestamp": 1_800_000_120, "open": 1990.0, "high": 1990.5, "low": 1969.0,
         "close": 1970.0, "volume": 300.0},
    ]


# --------------------------------------------------------------------------- synthesis

def __test_a_synthetic_day_reconstructs_each_bar_from_its_prints__():
    """The prints must be consistent with the bar they came from: the first print is the open,
    the last is the close, and the high and low are both touched. A synthesiser that emitted
    only closes would drive no stop trigger that the real day would have driven."""
    day = synthesise_day(_bars(), symbol="41I1G9000")

    for bar in _bars():
        prints = [p for p in day.prints if p["bar_ts"] == bar["timestamp"]]
        assert prints, f"no prints for bar {bar['timestamp']}"
        assert prints[0]["price"] == bar["open"]
        assert prints[-1]["price"] == bar["close"]
        assert max(p["price"] for p in prints) == bar["high"]
        assert min(p["price"] for p in prints) == bar["low"]


def __test_a_synthetic_day_conserves_each_bar_volume__():
    """Partial fills are matched against print VOLUME, so a synthesiser that invented volume
    would change how orders fill. The per-bar total must equal the bar's own volume."""
    day = synthesise_day(_bars(), symbol="41I1G9000")

    for bar in _bars():
        total = sum(p["volume"] for p in day.prints if p["bar_ts"] == bar["timestamp"])
        assert total == pytest.approx(bar["volume"])


def __test_a_synthetic_day_is_labelled_synthetic__():
    day = synthesise_day(_bars(), symbol="41I1G9000")
    assert day.label is DayLabel.SYNTHETIC


# --------------------------------------------------------------------------- the label is load-bearing

def __test_a_day_without_a_label_is_REFUSED_rather_than_assumed__():
    """The discriminating safety pin. A day whose provenance is missing must fail loudly; it
    must never default to RECORDED (a fabricated measurement) or to SYNTHETIC (a real capture
    quietly discounted). Refusing is the only answer that cannot mislead."""
    day = synthesise_day(_bars(), symbol="41I1G9000")
    raw = day.to_dict()
    del raw["label"]

    with pytest.raises(MalformedDay) as excinfo:
        VenueDay.from_dict(raw)
    assert "label" in str(excinfo.value).lower()


def __test_an_unknown_label_is_REFUSED_rather_than_coerced__():
    day = synthesise_day(_bars(), symbol="41I1G9000")
    raw = day.to_dict()
    raw["label"] = "PROBABLY_REAL"

    with pytest.raises(MalformedDay):
        VenueDay.from_dict(raw)


def __test_a_day_round_trips_through_disk_keeping_its_label__(tmp_path):
    day = synthesise_day(_bars(), symbol="41I1G9000")
    path = tmp_path / "day.json.gz"

    save_day(day, path)
    loaded = load_day(path)

    assert loaded.label is DayLabel.SYNTHETIC
    assert loaded.symbol == day.symbol
    assert len(loaded.prints) == len(day.prints)
    assert loaded.prints[0]["price"] == day.prints[0]["price"]


# --------------------------------------------------------------------------- partial days

def __test_a_day_started_mid_session_is_labelled_partial_and_says_where_it_joined__():
    """The live recording may be joined late. A partial day presented as whole would
    misrepresent the session's open, so it carries the flag and its first print's timestamp."""
    all_bars = _bars()
    day = synthesise_day(all_bars[1:], symbol="41I1G9000",
                         session_open_ts=all_bars[0]["timestamp"])

    assert day.partial is True
    assert day.first_print_ts == all_bars[1]["timestamp"]


def __test_a_day_covering_the_session_open_is_not_partial__():
    """The discriminating half: without it, a recorder that flagged EVERY day partial would
    pass the test above and tell us nothing."""
    all_bars = _bars()
    day = synthesise_day(all_bars, symbol="41I1G9000",
                         session_open_ts=all_bars[0]["timestamp"])

    assert day.partial is False


# --------------------------------------------------------------------------- replay into the venue

def __test_replaying_a_day_triggers_a_resting_stop_at_the_right_bar__():
    """The integration that matters: a day replayed into the venue must drive the venue exactly
    as the market did. The third bar trades down to 1969.0, so a sell stop at 1970.0 triggers
    there and not before."""
    day = synthesise_day(_bars(), symbol="41I1G9000")
    venue = FakeVenue(symbol="41I1G9000", market_type="DERIVATIVE",
                      last_price=1980.0, seed=7)
    oco = venue.place(category="OCO", side="sell", qty=1, price=2100.0, stop_price=1970.0)
    child_id = venue.order(oco["id"])["externalOrderId"]

    for bar_ts in (_bars()[0]["timestamp"], _bars()[1]["timestamp"]):
        day.replay_bar(bar_ts, into=venue)
        assert venue.order(child_id)["orderStatus"] == "New", (
            "the stop must not trigger on a bar whose low stays above it")

    day.replay_bar(_bars()[2]["timestamp"], into=venue)
    assert venue.order(child_id)["orderStatus"] == "Filled"


def __test_two_replays_of_one_day_produce_identical_venue_records__():
    """Replay determinism at the day level, not just the venue level."""
    day = synthesise_day(_bars(), symbol="41I1G9000")

    def _run():
        venue = FakeVenue(symbol="41I1G9000", market_type="DERIVATIVE",
                          last_price=1980.0, seed=7)
        venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)
        day.replay_into(venue)
        return venue.records()

    assert _run() == _run()
