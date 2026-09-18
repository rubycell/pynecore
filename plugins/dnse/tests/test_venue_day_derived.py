"""#157 — the THIRD venue-day provenance: a day derived from downloaded 1m history.

Tests written FIRST, before the label exists.

A day built from 1m OHLCV downloaded from the venue's history endpoint is genuinely a third
thing. It is not RECORDED — nobody captured a tick stream, and its intrabar sequence is
reconstructed, not observed. It is not SYNTHETIC either — SYNTHETIC means "derived from bars
already tracked in this repo", and a day pulled fresh from the venue for a specific session is
a real, dated measurement of that session's bars. Collapsing it into either existing label
would either overstate what was observed or discard where the numbers came from.

So the label is a new value, and this file pins the two properties that make it worth having:

1. It is DISTINCT from the other two, in the enum and after a disk round trip.
2. It carries WHERE it came from — symbol, session date, download time and the endpoint — and a
   day claiming this label without that provenance is REFUSED rather than loaded. A provenance
   label with no provenance attached is just a longer string.

The SYNTHETIC limitation applies to it unchanged and is pinned here too: intrabar sequence is
unknowable from bar data, so no day of this kind can answer questions about the order in which
trades arrived. What changes is only where the bars came from, never what they can prove.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_day import (                                               # noqa: E402
    DayLabel, MalformedDay, VenueDay, derive_day_from_1m_bars, group_bars_by_session,
    load_day, save_day, synthesise_day,
)

MINUTE_MS = 60_000
#: 2026-09-16 09:00:00 +07:00 as epoch milliseconds — a real VN session open.
SESSION_OPEN_MS = 1_789_524_000_000


def _minute_bars(count=3, start=SESSION_OPEN_MS):
    """``count`` consecutive 1m bars in MILLISECONDS, the unit the fake broker replays."""
    out = []
    price = 1980.0
    for index in range(count):
        out.append({"timestamp": start + index * MINUTE_MS,
                    "open": price, "high": price + 5.0, "low": price - 1.0,
                    "close": price + 4.0, "volume": 100.0 + index})
        price += 4.0
    return out


def _derived(bars=None, **overrides):
    kwargs = {"symbol": "41I1GA000", "session_date": "2026-09-16",
              "downloaded_at": "2026-09-18T15:04:05+07:00",
              "source": "GET /price/ohlc resolution=1"}
    kwargs.update(overrides)
    return derive_day_from_1m_bars(bars if bars is not None else _minute_bars(), **kwargs)


# --------------------------------------------------------------------------- the label itself

def __test_a_day_built_from_downloaded_1m_history_is_labelled_derived_from_1m__():
    assert _derived().label is DayLabel.DERIVED_FROM_1M
    assert DayLabel.DERIVED_FROM_1M.value == "DERIVED-FROM-1M"


def __test_the_derived_label_is_its_own_value_not_an_alias_of_the_other_two__():
    """The discriminating half. An implementation that made the new name a second spelling of
    SYNTHETIC would satisfy every other test in this file and lose the whole point: a reader
    could no longer tell a repo-tracked derivation from a dated download of a real session."""
    assert DayLabel.DERIVED_FROM_1M is not DayLabel.SYNTHETIC
    assert DayLabel.DERIVED_FROM_1M is not DayLabel.RECORDED
    assert len({d.value for d in DayLabel}) == 3


# --------------------------------------------------------------------------- provenance

def __test_a_derived_day_says_which_symbol_session_and_download_it_came_from__():
    day = _derived()

    assert day.provenance["symbol"] == "41I1GA000"
    assert day.provenance["session_date"] == "2026-09-16"
    assert day.provenance["downloaded_at"] == "2026-09-18T15:04:05+07:00"
    assert "price/ohlc" in day.provenance["source"]
    assert day.provenance["bar_count"] == 3


def __test_provenance_survives_the_disk_round_trip__(tmp_path):
    """Provenance that lives only in memory is provenance nobody will ever read: these days are
    written once and replayed for months."""
    path = save_day(_derived(), tmp_path / "derived.json.gz")

    loaded = load_day(path)

    assert loaded.label is DayLabel.DERIVED_FROM_1M
    assert loaded.provenance["session_date"] == "2026-09-16"
    assert loaded.provenance["symbol"] == "41I1GA000"


def __test_a_derived_day_with_no_provenance_is_REFUSED_on_load__(tmp_path):
    """The label claims a specific download of a specific session. A file carrying the label
    with nothing behind it makes that claim unverifiable, so it is refused rather than loaded —
    the same reasoning that refuses a day with no label at all."""
    raw = _derived().to_dict()
    raw["provenance"] = {}
    path = tmp_path / "bare.json.gz"
    import gzip
    import json
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle)

    with pytest.raises(MalformedDay) as excinfo:
        load_day(path)

    assert "provenance" in str(excinfo.value).lower()


def __test_a_derived_day_missing_ONE_provenance_field_is_REFUSED__(tmp_path):
    """Present-but-incomplete is the realistic failure, not empty. A day that names its symbol
    but not its session date cannot be matched back to the day it claims to reproduce."""
    raw = _derived().to_dict()
    del raw["provenance"]["session_date"]
    path = tmp_path / "partial_prov.json.gz"
    import gzip
    import json
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(raw, handle)

    with pytest.raises(MalformedDay) as excinfo:
        load_day(path)

    assert "session_date" in str(excinfo.value)


def __test_the_older_two_labels_still_load_without_provenance__(tmp_path):
    """The refusal above must be specific to the new label, not a blanket new requirement that
    invalidates every day already on disk."""
    path = save_day(synthesise_day(_minute_bars(), symbol="41I1G9000"), tmp_path / "s.json.gz")

    loaded = load_day(path)

    assert loaded.label is DayLabel.SYNTHETIC
    assert loaded.provenance == {}


# --------------------------------------------------------------------------- the bars themselves

def __test_a_derived_day_reconstructs_each_bar_from_its_prints__():
    """Same invariant the synthetic path carries: the prints must be consistent with the bar
    they came from, or a replay drives no trigger the real session drove."""
    bars = _minute_bars()
    day = _derived(bars)

    for bar in bars:
        prints = [p for p in day.prints if p["bar_ts"] == bar["timestamp"]]
        assert prints, f"no prints for bar {bar['timestamp']}"
        assert prints[0]["price"] == bar["open"]
        assert prints[-1]["price"] == bar["close"]
        assert max(p["price"] for p in prints) == bar["high"]
        assert min(p["price"] for p in prints) == bar["low"]


def __test_a_derived_day_conserves_each_bar_volume__():
    bars = _minute_bars()
    day = _derived(bars)

    for bar in bars:
        total = sum(p["volume"] for p in day.prints if p["bar_ts"] == bar["timestamp"])
        assert total == pytest.approx(bar["volume"])


def __test_a_file_of_five_minute_bars_is_REFUSED_rather_than_stamped_1m__():
    """The label names its source resolution, so the builder must not accept a 5m or 15m
    download and stamp it DERIVED-FROM-1M. Without this the label would be a claim nobody
    checks, and the derived day would replay a fifth of the price path it claims to carry."""
    five_minute = [dict(bar, timestamp=SESSION_OPEN_MS + index * 5 * MINUTE_MS)
                   for index, bar in enumerate(_minute_bars(6))]

    with pytest.raises(MalformedDay) as excinfo:
        _derived(five_minute)

    assert "1m" in str(excinfo.value) or "minute" in str(excinfo.value).lower()


def __test_the_lunch_break_and_the_odd_missing_minute_are_gaps_not_a_wrong_resolution__():
    """The discriminating half of the check above, and the reason it cannot demand a uniform 60s
    step. A real VN session breaks for lunch and again before the closing auction, and a quiet
    minute with no trades is simply absent from the venue's answer. A check that refused every
    irregular step would refuse every real session it exists to accept — so the rule is that the
    DOMINANT step must be one minute, and the rest are counted and reported as gaps."""
    morning = _minute_bars(4, start=SESSION_OPEN_MS)
    afternoon = _minute_bars(4, start=SESSION_OPEN_MS + 4 * 3600 * 1000)   # after lunch
    holed = morning + afternoon
    del holed[2]                                                           # one absent minute

    day = _derived(holed)

    assert day.provenance["bar_count"] == 7
    assert day.provenance["gaps"] == 2, "the lunch break and the missing minute, both counted"


def __test_timestamps_must_be_milliseconds_not_seconds__():
    """The venue's history endpoint answers in SECONDS and every consumer here replays
    MILLISECONDS. A day built from unconverted seconds replays one bar per minute-thousandth and
    the engine's wall-clock anchoring silently substitutes flat bars for the whole session — the
    exact failure the re-stamping work already cost a day to. Refuse it at the boundary."""
    seconds = [dict(b, timestamp=b["timestamp"] // 1000) for b in _minute_bars()]

    with pytest.raises(MalformedDay) as excinfo:
        _derived(seconds)

    assert "millisecond" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- grouping sessions

def __test_bars_are_grouped_into_sessions_by_their_local_trading_date__():
    """A download of N sessions arrives as one flat array; each venue day is ONE session, so the
    split has to happen on the Vietnamese trading date, not on UTC midnight — a UTC split would
    cut a VN morning away from its own afternoon."""
    day_one = _minute_bars(3, start=SESSION_OPEN_MS)
    day_two = _minute_bars(3, start=SESSION_OPEN_MS + 24 * 3600 * 1000)

    grouped = group_bars_by_session(day_one + day_two)

    assert list(grouped) == ["2026-09-16", "2026-09-17"]
    assert len(grouped["2026-09-16"]) == 3
    assert len(grouped["2026-09-17"]) == 3


def __test_a_late_evening_utc_bar_still_belongs_to_the_vietnamese_session_that_ran_it__():
    """The discriminating half: 02:00 UTC is 09:00 ICT the SAME day, but 23:00 UTC on the 15th is
    06:00 ICT on the 16th. A grouper keyed on UTC dates puts the two halves of one session into
    two different files, and every day built from it is silently truncated."""
    utc_previous_evening = SESSION_OPEN_MS - 3 * 3600 * 1000   # 06:00 ICT on the 16th
    grouped = group_bars_by_session(_minute_bars(1, start=utc_previous_evening))

    assert list(grouped) == ["2026-09-16"]


def __test_a_derived_day_can_be_replayed_like_any_other__():
    """Provenance is metadata; it must not change the replay contract the fake venue depends on."""
    from venue_core import FakeVenue

    bars = _minute_bars(3)
    day = _derived(bars)
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)

    fed = day.replay_into(venue)

    assert fed == len(day.prints)
    assert isinstance(day, VenueDay)
