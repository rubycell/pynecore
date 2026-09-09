"""#80 phase 1 — tick→bar synthesis, pinned against the LIVE measurement.

The corpus fixture is a raw 2026-09-09 capture of the DNSE tick channel
(public market data only — key union verified; the scrubber's WARN hits
are epoch timestamps). The venue's OWN REST 1m bar for the corpus's one
COMPLETE minute (09:43 ICT, bar time 1788921780) is embedded below as the
exact-parity expectation — the live dv=0 measurement, replayable offline
forever. A wrong boundary grid, a dropped print, a mis-parsed time format
or a float drift all break the equality.
"""
import json
import pathlib

from pynecore.core.tick_aggregator import TickAggregator
from pynecore_dnse.tick_frames import parse_tick_frame, parse_tick_time

_CORPUS = (pathlib.Path(__file__).parent / "fixtures"
           / "tick_corpus_vn30f1m.jsonl")

#: The venue's own REST 1m bar for the corpus's complete minute
#: (fetched 2026-09-09 from /price/ohlc, resolution=1).
_REST_0943 = {"time": 1788921780, "o": 1973.9, "h": 1975.8, "l": 1973.7,
              "c": 1973.8, "v": 1994.0}


def _ticks():
    for line in _CORPUS.read_text().splitlines():
        parsed = parse_tick_frame(json.loads(line))
        if parsed is not None:
            yield parsed


def _aggregate(tf_seconds):
    agg = TickAggregator(tf_seconds)
    bars = []
    for ts, price, qty, cum in _ticks():
        bars.extend(agg.add(ts, price, qty, cumulative=cum))
    final = agg.flush()
    return bars, final


def __test_corpus_replicates_the_venue_1m_bar_exactly__():
    """THE parity pin (dv=0, measured live, now offline): the synthesized
    09:43 bar must equal the venue's own REST bar field-for-field. Catches
    wrong grids, dropped/duplicated prints, and parse drift."""
    bars, _final = _aggregate(60)
    by_time = {b.time: b for b in bars}
    bar = by_time.get(_REST_0943["time"])
    assert bar is not None, f"no closed bar at {_REST_0943['time']}; got {sorted(by_time)}"
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (
        _REST_0943["o"], _REST_0943["h"], _REST_0943["l"],
        _REST_0943["c"], _REST_0943["v"]), (
        f"synthesized {bar} != venue REST {_REST_0943} — the live dv=0 "
        f"parity is broken")
    assert not bar.suspect, "a complete capture must not flag suspect"


def __test_15s_sub_bars_recompose_to_the_1m_bar__():
    """Grid + conservation: the 09:43 minute's 15S sub-bars are epoch-
    aligned and recompose exactly (first open, last close, max high, min
    low, summed volume). Catches boundary off-by-one and volume leaks."""
    bars, _final = _aggregate(15)
    minute = [b for b in bars if _REST_0943["time"] <= b.time
              < _REST_0943["time"] + 60]
    assert minute, "no 15S bars inside the complete minute"
    assert all(b.time % 15 == 0 for b in minute), "grid must be epoch-aligned"
    assert minute[0].open == _REST_0943["o"]
    assert minute[-1].close == _REST_0943["c"]
    assert max(b.high for b in minute) == _REST_0943["h"]
    assert min(b.low for b in minute) == _REST_0943["l"]
    assert sum(b.volume for b in minute) == _REST_0943["v"]


def __test_time_parsing_covers_all_three_measured_wire_formats__():
    """The probe hit three formats across captures — protobuf dict, ISO
    string, plain epoch (docs). Catches a parser pinned to one format
    (the first two probe iterations each missed one)."""
    assert parse_tick_time({"Seconds": 1788921780, "Nanos": 500000000}) \
        == 1788921780.5
    iso = parse_tick_time("2026-09-09T09:43:00+07:00")
    assert iso == 1788921780.0
    assert parse_tick_time(1788921780) == 1788921780.0
    assert parse_tick_time(1788921780000) == 1788921780.0     # ms epoch
    assert parse_tick_time("not-a-time") is None
    assert parse_tick_time(None) is None


def __test_missed_print_flags_the_bar_suspect__():
    """Gap detection via cumulative volume: drop one print from a window
    and the cumulative delta disagrees with the summed volume — the bar
    must come back suspect=True. Catches an aggregator that trusts an
    incomplete stream (a missed print silently wrongs the high/low)."""
    agg = TickAggregator(15)
    # window [990, 1005): ticks at 1000/1002/1004
    agg.add(1000.0, 100.0, 1, cumulative=51)
    agg.add(1002.0, 101.0, 2, cumulative=53)
    # a 3-lot print at 99.0 is MISSED here (cumulative jumps to 58)
    agg.add(1004.0, 100.5, 2, cumulative=58)
    closed = agg.add(1006.0, 100.6, 1, cumulative=59)   # next window (1005)
    assert len(closed) == 1
    assert closed[0].suspect, (
        "cumulative-volume mismatch (missed print) must flag the bar")
    assert closed[0].volume == 5                        # what we saw
    complete = agg.flush()
    assert complete is not None and not complete.suspect


def __test_empty_windows_emit_nothing__():
    """#80 adjudication: a zero-print window emits NO bar (matches the
    venue's own bar-stream shape). Catches flat-bar padding."""
    agg = TickAggregator(15)
    agg.add(1000.0, 100.0, 1)
    closed = agg.add(1075.0, 101.0, 1)    # 4 empty windows skipped
    assert len(closed) == 1, f"expected 1 closed bar, got {len(closed)}"
    assert closed[0].time == 990
    flushed = agg.flush()
    assert flushed is not None and flushed.time == 1065


def __test_control_frames_and_junk_are_ignored__():
    assert parse_tick_frame({"action": "ping"}) is None
    assert parse_tick_frame({"symbol": "X"}) is None          # no matchPrice
    assert parse_tick_frame({"matchPrice": "x", "time": 1}) is None
